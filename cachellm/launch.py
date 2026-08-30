"""`cachellm launch` - start the proxy and hand it to an app in one command.

The point is that you shouldn't have to think about the proxy at all. Run

    cachellm launch hermes

and you get a Hermes session talking to your usual provider through the cache.
If the proxy isn't running yet, it gets started. If it doesn't know your
upstream, it reads it out of the Hermes config you already have.

There's also a generic form for anything that speaks OpenAI over an env var:

    cachellm launch -- aider --model gpt-4o
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .logging_utils import get_logger, mask_secret

log = get_logger("cachellm.launch")

DEFAULT_PROFILE = "cachellm"


# ---------------------------------------------------------------------------
# reading an existing Hermes config
# ---------------------------------------------------------------------------


@dataclass
class HermesUpstream:
    """What we managed to learn from a Hermes install."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    home: Path | None = None
    profile: str | None = None

    @property
    def usable(self) -> bool:
        return bool(self.base_url)

    def describe(self) -> str:
        return (
            f"base_url={self.base_url or '<none>'} model={self.model or '<none>'} "
            f"api_key={mask_secret(self.api_key) or '<none>'}"
        )


def hermes_home() -> Path | None:
    """Locate the active Hermes home, honouring $HERMES_HOME and profiles."""
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        path = Path(env_home).expanduser()
        if path.is_dir():
            return path
    for candidate in (
        Path.home() / ".hermes",
        Path(os.environ.get("LOCALAPPDATA", "")) / "hermes"
        if os.environ.get("LOCALAPPDATA")
        else None,
        Path(os.environ.get("APPDATA", "")) / "hermes" if os.environ.get("APPDATA") else None,
    ):
        if candidate and candidate.is_dir():
            return candidate
    return None


def _profile_dir(home: Path, profile: str | None) -> Path:
    if not profile or profile == "default":
        return home
    return home / "profiles" / profile


# A deliberately narrow YAML reader: we only want the top-level `model:` block,
# which is a flat two-space-indented mapping in every Hermes config. Using a real
# YAML parser would mean adding a dependency for six lines of data.
_TOP_KEY = re.compile(r"^([A-Za-z_][\w-]*):\s*(.*)$")
_CHILD_KEY = re.compile(r"^\s{2}([A-Za-z_][\w-]*):\s*(.*)$")


def read_model_block(config_path: Path) -> dict[str, str]:
    if not config_path.is_file():
        return {}
    out: dict[str, str] = {}
    inside = False
    try:
        lines = config_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        top = _TOP_KEY.match(line)
        if top:
            inside = top.group(1) == "model"
            continue
        if not inside:
            continue
        child = _CHILD_KEY.match(line)
        if child is None:
            if line[:1] not in (" ", "\t"):
                inside = False
            continue
        key, value = child.group(1), child.group(2).strip()
        if value:
            out[key] = value.strip().strip("'\"")
    return out


def read_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    except OSError:
        return {}
    return values


_VAR = re.compile(r"\$\{([A-Za-z_][\w]*)\}|\$([A-Za-z_][\w]*)")


def expand_vars(value: str, env: dict[str, str]) -> str:
    """Resolve ${VAR} / $VAR against a Hermes .env, then the process env."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return env.get(name) or os.environ.get(name, "")

    return _VAR.sub(replace, value)


def discover_hermes_upstream(profile: str | None = None) -> HermesUpstream:
    """Read base_url / api_key / model out of an existing Hermes profile."""
    home = hermes_home()
    if home is None:
        return HermesUpstream()
    directory = _profile_dir(home, profile)
    model = read_model_block(directory / "config.yaml")
    if not model and directory != home:
        # A cloned profile may inherit the model block from the default home.
        model = read_model_block(home / "config.yaml")
    env = read_env_file(directory / ".env")
    if directory != home:
        merged = read_env_file(home / ".env")
        merged.update(env)
        env = merged
    return HermesUpstream(
        base_url=expand_vars(model.get("base_url", ""), env).rstrip("/"),
        api_key=expand_vars(model.get("api_key", ""), env),
        model=model.get("default", ""),
        home=home,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# proxy lifecycle
# ---------------------------------------------------------------------------


def proxy_url(config: Config) -> str:
    host = config.server.host
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    return f"http://{host}:{config.server.port}"


def proxy_is_up(base: str, timeout: float = 1.5) -> bool:
    import httpx

    try:
        response = httpx.get(f"{base}/health", timeout=timeout)
        return response.status_code < 500
    except Exception:
        return False


def wait_for_proxy(base: str, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proxy_is_up(base):
            return True
        time.sleep(0.4)
    return False


def start_proxy_detached(
    config: Config,
    *,
    upstream: HermesUpstream | None = None,
    extra_env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Start `cachellm start` in the background, inheriting our settings."""
    env = dict(os.environ)
    env.setdefault("PORT", str(config.server.port))
    env.setdefault("HOST", config.server.host)
    if upstream is not None and upstream.usable:
        env.setdefault("UPSTREAM_BASE_URL", upstream.base_url)
        if upstream.api_key:
            env.setdefault("UPSTREAM_API_KEY", upstream.api_key)
    env.update(extra_env or {})

    command = [sys.executable, "-m", "cachellm", "start"]
    log_path = log_path or default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    handle.write(
        f"\n=== cachellm start (launched {time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n".encode()
    )

    kwargs: dict[str, object] = {
        "stdout": handle,
        "stderr": handle,
        "stdin": subprocess.DEVNULL,
        "env": env,
        "cwd": os.getcwd(),
    }
    if os.name == "nt":
        # Detach so closing the launching terminal doesn't kill the proxy.
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)  # type: ignore[arg-type]


def default_log_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
    if base:
        return Path(base) / "cachellm" / "proxy.log"
    return Path.home() / ".cachellm" / "proxy.log"


# ---------------------------------------------------------------------------
# launching Hermes
# ---------------------------------------------------------------------------


def hermes_executable() -> str | None:
    return shutil.which("hermes")


def hermes_profile_exists(profile: str) -> bool:
    home = hermes_home()
    if home is None:
        return False
    return (home / "profiles" / profile / "config.yaml").is_file()


def run_hermes_command(args: list[str], *, quiet: bool = True) -> tuple[int, str]:
    """Run a `hermes ...` command, returning (exit code, combined output)."""
    executable = hermes_executable()
    if executable is None:
        return 127, "hermes is not on PATH"
    try:
        result = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return 124, "hermes command timed out"
    except OSError as exc:
        return 1, str(exc)
    output = (result.stdout or "") + (result.stderr or "")
    if not quiet and output.strip():
        print(output.strip())
    return result.returncode, output


def ensure_hermes_profile(
    profile: str,
    proxy_base: str,
    *,
    source_profile: str | None = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Create (if needed) a Hermes profile whose model.base_url is the proxy.

    We never touch the user's default profile - the whole point is that the
    cached setup lives alongside it and can be thrown away.
    """
    created = False
    if not hermes_profile_exists(profile):
        args = ["profile", "create", profile, "--clone"]
        if source_profile and source_profile != "default":
            args = ["profile", "create", profile, "--clone-from", source_profile]
        code, output = run_hermes_command(args, quiet=not verbose)
        if code != 0 and not hermes_profile_exists(profile):
            return False, f"could not create the Hermes profile {profile!r}: {output.strip()[:300]}"
        created = True

    code, output = run_hermes_command(
        ["--profile", profile, "config", "set", "model.base_url", f"{proxy_base}/v1"],
        quiet=not verbose,
    )
    if code != 0:
        return False, f"could not point {profile!r} at the proxy: {output.strip()[:300]}"
    return True, ("created" if created else "reused")


def exec_hermes(profile: str | None, passthrough: list[str]) -> int:
    """Hand the terminal over to Hermes. Interactive, inherits stdin/stdout.

    ``profile=None`` means the default profile, so no --profile flag at all.
    """
    executable = hermes_executable()
    if executable is None:
        print(
            "hermes is not on PATH. Install it first:\n"
            "  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash",
            file=sys.stderr,
        )
        return 127
    command = [executable]
    if profile and profile != "default":
        command += ["--profile", profile]
    command += passthrough
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


# ---------------------------------------------------------------------------
# pointing a profile at the proxy (and putting it back)
# ---------------------------------------------------------------------------


def state_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
    return Path(base) / "cachellm" if base else Path.home() / ".cachellm"


def restore_file() -> Path:
    return state_dir() / "hermes-restore.json"


def read_base_url(profile: str | None) -> str:
    """What is this profile's model.base_url right now?"""
    home = hermes_home()
    if home is None:
        return ""
    model = read_model_block(_profile_dir(home, profile) / "config.yaml")
    return model.get("base_url", "")


def set_base_url(profile: str | None, url: str, *, verbose: bool = False) -> tuple[bool, str]:
    """Change a profile's base_url through the Hermes CLI.

    Deliberately shelling out rather than editing config.yaml ourselves - a
    stray indent in that file breaks Hermes, and `hermes config set` knows how
    to write it safely.
    """
    args: list[str] = []
    if profile and profile != "default":
        args += ["--profile", profile]
    args += ["config", "set", "model.base_url", url]
    code, output = run_hermes_command(args, quiet=not verbose)
    return code == 0, output


def remember_original(profile: str, original_url: str) -> None:
    """Write down where a profile used to point, so we can undo it."""
    import json

    path = restore_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        saved = {}
    # Never overwrite an existing restore point with a proxy URL - that would
    # lose the real provider if launch ran twice.
    if profile not in saved:
        saved[profile] = original_url
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def restore_points() -> dict[str, str]:
    import json

    path = restore_file()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def forget_original(profile: str) -> None:
    import json

    path = restore_file()
    saved = restore_points()
    if profile in saved:
        saved.pop(profile)
        try:
            if saved:
                path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            else:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def looks_like_proxy(url: str, proxy_base: str) -> bool:
    if not url:
        return False
    normalised = url.rstrip("/")
    return normalised in {proxy_base.rstrip("/"), f"{proxy_base.rstrip('/')}/v1"} or (
        "127.0.0.1" in normalised or "localhost" in normalised
    ) and "/v1" in normalised


def point_profile_at_proxy(
    profile: str,
    proxy_base: str,
    *,
    verbose: bool = False,
) -> tuple[bool, str, str]:
    """Repoint an existing profile, remembering where it used to go.

    Returns (ok, message, original_url).
    """
    original = read_base_url(profile)
    if looks_like_proxy(original, proxy_base):
        return True, "already pointed at the cache", original
    if original:
        remember_original(profile, original)
    ok, output = set_base_url(profile, f"{proxy_base}/v1", verbose=verbose)
    if not ok:
        return False, f"could not update the profile: {output.strip()[:300]}", original
    return True, "repointed", original


def restore_profile(profile: str, *, verbose: bool = False) -> tuple[bool, str]:
    """Put a profile's base_url back to whatever it was before launch."""
    saved = restore_points()
    original = saved.get(profile)
    if not original:
        return False, f"no saved provider URL for profile {profile!r}"
    ok, output = set_base_url(profile, original, verbose=verbose)
    if not ok:
        return False, f"could not restore {profile!r}: {output.strip()[:300]}"
    forget_original(profile)
    return True, original


# ---------------------------------------------------------------------------
# launching anything else
# ---------------------------------------------------------------------------

# Env var names different tools read their OpenAI base URL from. We set all of
# them - they don't conflict, and it saves the user looking up which one applies.
BASE_URL_VARS: tuple[str, ...] = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_API_BASE_URL",
    "LLM_BASE_URL",
)


def exec_with_proxy_env(
    command: list[str], proxy_base: str, *, api_key: str = "cachellm-local"
) -> int:
    """Run an arbitrary command with the OpenAI base URL pointed at the proxy."""
    if not command:
        print("nothing to run - pass a command after --", file=sys.stderr)
        return 2
    executable = shutil.which(command[0])
    if executable is None:
        print(f"{command[0]!r} is not on PATH", file=sys.stderr)
        return 127
    env = dict(os.environ)
    for name in BASE_URL_VARS:
        env[name] = f"{proxy_base}/v1"
    env.setdefault("OPENAI_API_KEY", api_key)
    try:
        return subprocess.call([executable, *command[1:]], env=env)
    except KeyboardInterrupt:
        return 130
