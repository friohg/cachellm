"""Tests for `cachellm launch` and the dashboard's accessibility contract.

The launch tests avoid actually spawning Hermes; they check the pieces that can
silently rot - reading an existing Hermes config, and the argument wiring.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cachellm.cli import build_parser
from cachellm.launch import (
    BASE_URL_VARS,
    discover_hermes_upstream,
    expand_vars,
    proxy_url,
    read_env_file,
    read_model_block,
)
from cachellm.config import Config
from cachellm.server import STATIC_DIR


# ---------------------------------------------------------------------------
# reading a Hermes install
# ---------------------------------------------------------------------------


HERMES_CONFIG = """\
model:
  default: claude-opus-5-thinking
  provider: custom
  base_url: https://api.example.com/v1
  api_key: ${MY_PROVIDER_KEY}
  api_mode: chat_completions
database:
  journal_mode: wal
agent:
  max_turns: 0
"""


def test_read_model_block(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(HERMES_CONFIG, encoding="utf-8")
    model = read_model_block(path)
    assert model["default"] == "claude-opus-5-thinking"
    assert model["base_url"] == "https://api.example.com/v1"
    assert model["api_key"] == "${MY_PROVIDER_KEY}"
    # Keys from other top-level blocks must not leak in.
    assert "journal_mode" not in model
    assert "max_turns" not in model


def test_read_model_block_handles_bom_and_missing_file(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("\ufeff" + HERMES_CONFIG, encoding="utf-8")
    assert read_model_block(path)["default"] == "claude-opus-5-thinking"
    assert read_model_block(tmp_path / "nope.yaml") == {}


def test_read_env_file(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\nMY_PROVIDER_KEY=secret-value\nQUOTED='also-secret'\nbroken-line\n",
        encoding="utf-8",
    )
    values = read_env_file(path)
    assert values["MY_PROVIDER_KEY"] == "secret-value"
    assert values["QUOTED"] == "also-secret"


def test_expand_vars_prefers_the_env_file():
    assert expand_vars("${A}", {"A": "from-file"}) == "from-file"
    assert expand_vars("$A", {"A": "from-file"}) == "from-file"
    assert expand_vars("plain", {}) == "plain"
    assert expand_vars("${MISSING}", {}) == ""


def test_discover_hermes_upstream(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(HERMES_CONFIG, encoding="utf-8")
    (home / ".env").write_text("MY_PROVIDER_KEY=live-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    info = discover_hermes_upstream()
    assert info.usable is True
    assert info.base_url == "https://api.example.com/v1"
    assert info.api_key == "live-key"
    assert info.model == "claude-opus-5-thinking"
    # The description is for humans and must not print the key.
    assert "live-key" not in info.describe()


def test_discover_hermes_upstream_profile_inherits_env(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "profiles" / "work").mkdir(parents=True)
    (home / ".env").write_text("MY_PROVIDER_KEY=root-key\n", encoding="utf-8")
    (home / "profiles" / "work" / "config.yaml").write_text(HERMES_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    info = discover_hermes_upstream("work")
    assert info.base_url == "https://api.example.com/v1"
    assert info.api_key == "root-key", "a profile should fall back to the root .env"


def test_discover_hermes_upstream_when_hermes_is_absent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "does-not-exist"))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
    info = discover_hermes_upstream()
    assert info.usable is False


def test_proxy_url_normalises_wildcard_hosts():
    config = Config()
    config.server.host = "0.0.0.0"
    config.server.port = 4123
    assert proxy_url(config) == "http://127.0.0.1:4123"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_launch_defaults_to_hermes():
    args = build_parser().parse_args(["launch"])
    assert args.command == "launch"
    assert args.target == "hermes"
    assert args.strict_tools is False, "agent caching must be on by default"
    assert args.tool_ttl == 1800


def test_launch_passes_extra_args_through():
    """`--` separation is handled in main(), not the parser, so test it there."""
    from cachellm.cli import build_parser as bp

    raw = ["launch", "hermes", "--", "-q", "hello"]
    index = raw.index("--")
    args = bp().parse_args(raw[:index])
    passthrough = raw[index + 1 :]
    assert args.target == "hermes"
    assert passthrough == ["-q", "hello"]


def test_flags_and_passthrough_can_coexist():
    """The case that broke first: a flag AND a `--` tail on the same line."""
    raw = ["launch", "hermes", "--use-profile", "work", "--", "-q", "hi"]
    index = raw.index("--")
    args = build_parser().parse_args(raw[:index])
    assert args.use_profile == "work"
    assert raw[index + 1 :] == ["-q", "hi"]


def test_launch_can_target_another_program():
    """Flags for the target go after `--`, since cachellm has flags of its own."""
    raw = ["launch", "aider", "--", "--model", "gpt-4o"]
    index = raw.index("--")
    args = build_parser().parse_args(raw[:index])
    assert args.target == "aider"
    assert raw[index + 1 :] == ["--model", "gpt-4o"]


def test_launch_flags():
    args = build_parser().parse_args(
        ["launch", "--port", "4500", "--upstream", "https://x/v1", "--semantic",
         "--strict-tools", "--ttl", "60", "--profile-name", "mine"]
    )
    assert args.port == 4500
    assert args.upstream == "https://x/v1"
    assert args.semantic is True
    assert args.strict_tools is True
    assert args.ttl == 60
    assert args.profile_name == "mine"


def test_base_url_vars_cover_the_common_tools():
    assert "OPENAI_BASE_URL" in BASE_URL_VARS
    assert "OPENAI_API_BASE" in BASE_URL_VARS


def test_use_profile_bare_means_the_default_profile():
    args = build_parser().parse_args(["launch", "--use-profile"])
    assert args.use_profile == "", "bare flag means the default profile"


def test_use_profile_can_name_a_profile():
    args = build_parser().parse_args(["launch", "--use-profile", "work"])
    assert args.use_profile == "work"


def test_use_profile_absent_by_default():
    args = build_parser().parse_args(["launch"])
    assert args.use_profile is None, "default is the throwaway profile"


def test_unlink_command_exists():
    args = build_parser().parse_args(["unlink"])
    assert args.command == "unlink"
    assert args.profile is None
    assert build_parser().parse_args(["unlink", "work"]).profile == "work"
    assert build_parser().parse_args(["unlink", "--list"]).list is True


# ---------------------------------------------------------------------------
# repointing an existing profile, and putting it back
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hermes(tmp_path, monkeypatch):
    """A pretend Hermes install plus a stubbed `hermes config set`."""
    from cachellm import launch as launch_mod

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(HERMES_CONFIG, encoding="utf-8")
    (home / ".env").write_text("MY_PROVIDER_KEY=live-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "state"))

    calls: list[list[str]] = []

    def fake_run(args, *, quiet=True):
        calls.append(list(args))
        # Emulate `hermes config set model.base_url <url>` writing the file.
        if "config" in args and "set" in args and "model.base_url" in args:
            url = args[args.index("model.base_url") + 1]
            profile = args[args.index("--profile") + 1] if "--profile" in args else None
            target = home if profile in (None, "default") else home / "profiles" / profile
            target.mkdir(parents=True, exist_ok=True)
            path = target / "config.yaml"
            existing = path.read_text(encoding="utf-8") if path.is_file() else HERMES_CONFIG
            path.write_text(
                re.sub(r"(?m)^(  base_url: ).*$", rf"\g<1>{url}", existing),
                encoding="utf-8",
            )
        return 0, "ok"

    monkeypatch.setattr(launch_mod, "run_hermes_command", fake_run)
    return {"home": home, "calls": calls}


def test_point_profile_at_proxy_and_restore(fake_hermes):
    from cachellm.launch import (
        point_profile_at_proxy,
        read_base_url,
        restore_points,
        restore_profile,
    )

    assert read_base_url("default") == "https://api.example.com/v1"

    ok, detail, original = point_profile_at_proxy("default", "http://127.0.0.1:4000")
    assert ok is True
    assert detail == "repointed"
    assert original == "https://api.example.com/v1"
    assert read_base_url("default") == "http://127.0.0.1:4000/v1"
    assert restore_points()["default"] == "https://api.example.com/v1"

    ok, restored = restore_profile("default")
    assert ok is True
    assert restored == "https://api.example.com/v1"
    assert read_base_url("default") == "https://api.example.com/v1"
    assert restore_points() == {}, "the restore point is cleared once used"


def test_repointing_twice_keeps_the_original_provider(fake_hermes):
    """The dangerous case: running launch twice must not save the proxy URL
    as the 'original', which would make unlink a no-op."""
    from cachellm.launch import point_profile_at_proxy, restore_points

    point_profile_at_proxy("default", "http://127.0.0.1:4000")
    point_profile_at_proxy("default", "http://127.0.0.1:4000")
    assert restore_points()["default"] == "https://api.example.com/v1"


def test_repointing_an_already_pointed_profile_is_a_noop(fake_hermes):
    from cachellm.launch import point_profile_at_proxy

    point_profile_at_proxy("default", "http://127.0.0.1:4000")
    ok, detail, _ = point_profile_at_proxy("default", "http://127.0.0.1:4000")
    assert ok is True
    assert detail == "already pointed at the cache"


def test_restore_without_a_saved_point_fails_cleanly(fake_hermes):
    from cachellm.launch import restore_profile

    ok, message = restore_profile("never-touched")
    assert ok is False
    assert "no saved provider URL" in message


def test_looks_like_proxy():
    from cachellm.launch import looks_like_proxy

    base = "http://127.0.0.1:4000"
    assert looks_like_proxy("http://127.0.0.1:4000/v1", base) is True
    assert looks_like_proxy("http://localhost:4000/v1", base) is True
    assert looks_like_proxy("https://api.example.com/v1", base) is False
    assert looks_like_proxy("", base) is False


# ---------------------------------------------------------------------------
# dashboard accessibility
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dashboard_html() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dashboard_js() -> str:
    return (STATIC_DIR / "dashboard.js").read_text(encoding="utf-8")


def test_page_declares_a_language(dashboard_html):
    assert re.search(r'<html[^>]+lang="en"', dashboard_html)


def test_page_has_a_skip_link_to_main(dashboard_html):
    assert 'class="sr-only skip-link" href="#main"' in dashboard_html
    assert 'id="main"' in dashboard_html


def test_tabs_use_the_aria_tabs_pattern(dashboard_html):
    assert 'role="tablist"' in dashboard_html
    assert dashboard_html.count('role="tab"') >= 6
    assert dashboard_html.count('role="tabpanel"') >= 6
    assert 'aria-selected="true"' in dashboard_html
    # Every tab must name the panel it controls.
    controls = re.findall(r'role="tab"[^>]*aria-controls="([^"]+)"', dashboard_html)
    assert len(controls) >= 6
    for panel_id in controls:
        assert f'id="{panel_id}"' in dashboard_html


def test_tabs_are_keyboard_navigable(dashboard_js):
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert key in dashboard_js


def test_live_regions_exist_for_announcements(dashboard_html):
    assert 'role="status"' in dashboard_html
    assert 'role="alert"' in dashboard_html
    assert 'aria-live="polite"' in dashboard_html


def test_every_input_has_a_label(dashboard_html):
    """No orphan controls: each id used by an input is referenced by a label."""
    input_ids = set(re.findall(r'<(?:input|select)[^>]*\bid="([^"]+)"', dashboard_html))
    labelled = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', dashboard_html))
    aria_labelled = set(
        re.findall(r'<(?:input|select)[^>]*\bid="([^"]+)"[^>]*aria-label=', dashboard_html)
    )
    missing = input_ids - labelled - aria_labelled
    assert not missing, f"inputs without a label: {sorted(missing)}"


def test_buttons_are_real_buttons_with_types(dashboard_html):
    buttons = re.findall(r"<button[^>]*>", dashboard_html)
    assert buttons
    for button in buttons:
        assert 'type="button"' in button, f"button missing type: {button}"
    # No fake buttons made of divs or links.
    assert 'onclick=' not in dashboard_html, "no inline handlers; keyboard support comes free with <button>"


def test_tables_have_captions_and_scoped_headers(dashboard_html):
    assert dashboard_html.count("<caption") >= 6
    assert 'scope="col"' in dashboard_html


def test_row_headers_are_used_for_identifying_columns(dashboard_js):
    assert "rowHeader" in dashboard_js
    assert "th.scope = 'row'" in dashboard_js


def test_outcomes_are_not_conveyed_by_colour_alone(dashboard_js):
    assert "OUTCOMES" in dashboard_js
    for label in ("hit (exact)", "hit (similar)", "miss", "not cacheable", "error"):
        assert label in dashboard_js
    # A glyph accompanies the colour.
    assert "\\u2713" in dashboard_js or "✓" in dashboard_js


def test_reduced_motion_and_contrast_preferences_are_honoured(dashboard_html):
    assert "prefers-reduced-motion" in dashboard_html
    assert "prefers-contrast" in dashboard_html
    assert "prefers-color-scheme" in dashboard_html


def test_autorefresh_can_be_switched_off(dashboard_html, dashboard_js):
    assert 'id="auto-toggle"' in dashboard_html
    assert 'aria-pressed' in dashboard_html
    assert "prefers-reduced-motion" in dashboard_js, "start paused when motion is reduced"


def test_focus_is_always_visible(dashboard_html):
    assert ":focus-visible" in dashboard_html
    assert "outline" in dashboard_html


def test_content_is_built_with_textcontent_not_innerhtml(dashboard_js):
    """innerHTML with user data would be both an injection and an a11y hazard."""
    assert "innerHTML" not in dashboard_js
    assert "textContent" in dashboard_js


async def test_dashboard_and_its_script_are_served(client):
    page = await client.get("/dashboard")
    assert page.status_code == 200
    assert 'role="tablist"' in page.text

    script = await client.get("/static/dashboard.js")
    assert script.status_code == 200
    assert "OUTCOMES" in script.text
