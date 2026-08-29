"""CacheLLM command line interface.

    cachellm start      run the proxy (+ dashboard)
    cachellm stats      show statistics
    cachellm clear      invalidate cache entries
    cachellm inspect    inspect a cache entry / list entries
    cachellm config     show effective configuration, env docs, or write a sample file
    cachellm health     check cache + upstream health
    cachellm pricing    view/set model pricing
    cachellm policy     ask the policy engine about a tool

Every command talks to a running proxy over HTTP when one is reachable, and
falls back to reading the local SQLite database directly when it is not - so
``cachellm stats`` works whether or not the server is up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config, ModelPricing, env_docs, load_config
from .logging_utils import configure_logging, get_logger

log = get_logger("cachellm.cli")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _base_url(config: Config, override: str | None = None) -> str:
    if override:
        return override.rstrip("/")
    host = config.server.host
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    return f"http://{host}:{config.server.port}"


def _http(method: str, url: str, payload: Any = None, timeout: float = 10.0) -> Any:
    """Minimal HTTP client via httpx (already a dependency)."""
    import httpx

    with httpx.Client(timeout=timeout) as client:
        response = client.request(method, url, json=payload)
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()


def _try_http(method: str, url: str, payload: Any = None) -> tuple[bool, Any]:
    try:
        return True, _http(method, url, payload)
    except Exception as exc:
        return False, str(exc)


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


def _state_for_offline(config: Config):
    """Build an AppState for direct (server-less) operations."""
    from .app import AppState

    return AppState(config)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.port:
        config.server.port = args.port
    if args.host:
        config.server.host = args.host
    if args.upstream:
        config.upstream.base_url = args.upstream.rstrip("/")
    if args.backend:
        config.cache.backend = args.backend
    if args.ttl is not None:
        config.cache.default_ttl = args.ttl
    if args.semantic:
        config.semantic.enabled = True
    if args.no_cache:
        config.cache.enabled = False
    if args.log_level:
        config.server.log_level = args.log_level.upper()

    configure_logging(config.server.log_level)

    if not config.upstream.base_url:
        print(
            "warning: UPSTREAM_BASE_URL is not set - cache lookups will work but "
            "misses will fail until you configure an upstream provider.\n"
            "         set it via environment variable, config file, or --upstream.",
            file=sys.stderr,
        )

    import uvicorn

    from .server import create_app

    app = create_app(config)
    banner = (
        f"\nCacheLLM {__version__}\n"
        f"  proxy      http://{config.server.host}:{config.server.port}/v1\n"
        f"  dashboard  http://{config.server.host}:{config.server.port}/dashboard\n"
        f"  cache      {config.cache.backend}"
        + (f" ({config.cache.sqlite_path})" if config.cache.backend == "sqlite" else "")
        + f"\n  ttl        {config.cache.default_ttl}s"
        f"\n  semantic   {'on' if config.semantic.enabled else 'off'}"
        f"\n  upstream   {config.upstream.base_url or '<unset>'}\n"
    )
    print(banner)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        access_log=args.access_log,
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ok, data = _try_http("GET", f"{_base_url(config, args.url)}/api/stats")
    if ok:
        if args.json:
            _print_json(data)
            return 0
        _render_stats(data)
        return 0

    print(f"proxy not reachable ({data}); reading the local database instead\n")
    state = _state_for_offline(config)
    try:
        if state.db is None:
            print("no database configured; nothing to report")
            return 1
        totals = state.db.usage_totals()
        _print_json({"lifetime": totals, "storage": state.db.stats_snapshot()})
        by_model = [dict(r) for r in state.db.usage_by_model()]
        if by_model:
            print("\nby model:")
            _table(
                by_model,
                ["model", "requests", "hits", "prompt_tokens", "completion_tokens", "cost"],
            )
        return 0
    finally:
        asyncio.run(state.shutdown())


def _render_stats(data: dict[str, Any]) -> None:
    c = data["counters"]
    cost = data["cost"]
    lat = data["latency"]
    print("CacheLLM statistics")
    print("-------------------")
    print(f"  requests            {c['total_requests']}")
    print(f"  cache hits          {c['cache_hits']}  ({data['cache_hit_rate_pct']}% hit rate)")
    print(f"    exact             {c['exact_hits']}")
    print(f"    semantic          {c['semantic_hits']}")
    print(f"    tool              {c['tool_cache_hits']}")
    print(f"    coalesced         {c['coalesced_requests']}")
    print(f"  cache misses        {c['cache_misses']}")
    print(f"  upstream requests   {c['upstream_requests']}")
    print(f"  upstream errors     {c['upstream_errors']}")
    print(f"  input tokens        {c['input_tokens']}")
    print(f"  output tokens       {c['output_tokens']}")
    print(f"  tokens saved        {data['estimated_tokens_saved']}")
    print(f"  actual cost         {cost['actual_cost']} {cost['currency']}")
    print(f"  cost without cache  {cost['cost_without_cache']} {cost['currency']}")
    print(f"  savings             {cost['savings']} {cost['currency']} ({cost['savings_pct']}%)")
    print(f"  avg latency         {lat['avg_total_ms']} ms")
    print(f"  avg cache lookup    {lat['avg_cache_lookup_ms']} ms")
    print(f"  avg upstream        {lat['avg_upstream_ms']} ms")
    storage = data.get("storage")
    if storage:
        print(
            f"  stored entries      {storage['cache_entries']} response, "
            f"{storage['tool_cache_entries']} tool, {storage['semantic_entries']} vectors"
        )


def cmd_clear(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    body: dict[str, Any] = {}
    if args.all:
        body["all"] = True
    if args.key:
        body["key"] = args.key
    if args.model:
        body["model"] = args.model
    if args.namespace:
        body["namespace"] = args.namespace
    if args.tool:
        body["tool"] = args.tool
    if not body:
        print("nothing selected; pass --all, --key, --model, --namespace or --tool")
        return 2

    ok, data = _try_http("POST", f"{_base_url(config, args.url)}/api/cache/invalidate", body)
    if ok:
        print(f"removed: {json.dumps(data['removed'])}")
        return 0

    print(f"proxy not reachable ({data}); operating on the local database")
    state = _state_for_offline(config)
    try:
        removed = asyncio.run(
            state.engine.invalidate(
                key=args.key,
                model=args.model,
                namespace=args.namespace,
                tool=args.tool,
                all_entries=args.all,
            )
        )
        print(f"removed: {json.dumps(removed)}")
        return 0
    finally:
        asyncio.run(state.shutdown())


def cmd_inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = _base_url(config, args.url)
    if args.key:
        ok, data = _try_http("GET", f"{base}/api/cache/{args.key}")
        if ok:
            _print_json(data)
            return 0
        state = _state_for_offline(config)
        try:
            if state.db is None:
                print("no database configured")
                return 1
            detail = state.db.cache_entry_detail(args.key)
            if detail is None:
                print("entry not found")
                return 1
            _print_json(detail)
            return 0
        finally:
            asyncio.run(state.shutdown())

    query = f"?limit={args.limit}" + (f"&search={args.search}" if args.search else "")
    ok, data = _try_http("GET", f"{base}/api/cache{query}")
    if ok:
        rows = [
            {
                "key": e["key"][:16],
                "model": e.get("model") or "",
                "category": e.get("category") or "",
                "hits": e.get("hits"),
                "ttl_left": e.get("ttl_remaining"),
                "prompt": (e.get("prompt_excerpt") or "")[:48],
            }
            for e in data["entries"]
        ]
        print(f"{data['total']} entries total")
        _table(rows, ["key", "model", "category", "hits", "ttl_left", "prompt"])
        return 0

    state = _state_for_offline(config)
    try:
        if state.db is None:
            print("no database configured")
            return 1
        rows = [
            dict(r)
            for r in state.db.list_cache_entries(search=args.search, limit=args.limit)
        ]
        _table(
            [
                {
                    "key": r["key"][:16],
                    "model": r["model"] or "",
                    "hits": r["hits"],
                    "prompt": (r["prompt_excerpt"] or "")[:48],
                }
                for r in rows
            ],
            ["key", "model", "hits", "prompt"],
        )
        return 0
    finally:
        asyncio.run(state.shutdown())


SAMPLE_CONFIG: dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 4000, "log_level": "INFO"},
    "upstream": {
        "base_url": "https://api.example.com/v1",
        "api_key": "",
        "provider": "openai_compatible",
        "forward_client_key": False,
    },
    "cache": {
        "backend": "sqlite",
        "sqlite_path": "./data/cache.db",
        "default_ttl": 3600,
        "namespace": "default",
        "fail_open": True,
    },
    "semantic": {"enabled": False, "threshold": 0.92, "backend": "hash"},
    "policy": {
        "category_ttl": {
            "static": 86400,
            "general": 3600,
            "read_only_tool": 300,
            "search": 60,
            "current_information": 0,
            "mutation": 0,
        },
        "tool_policies": {
            "weather": {"cacheable": True, "ttl": 300},
            "list_repositories": {"cacheable": True, "ttl": 30},
            "delete_repository": {"cacheable": False, "ttl": 0},
        },
        "unsafe_tool_deny_list": ["*payment*", "*transfer*"],
    },
    "pricing": {
        "currency": "USD",
        "models": {
            "your-model-name": {"input_per_1m": 0.15, "output_per_1m": 0.60},
        },
    },
    "privacy": {"store_prompts": True, "store_responses": True, "retention_days": 30},
    "routing": {"model_map": {}},
}


def cmd_config(args: argparse.Namespace) -> int:
    if args.env:
        print("CacheLLM environment variables")
        print("------------------------------")
        for name, description in env_docs():
            print(f"  {name:<24} {description}")
        return 0

    if args.init:
        target = Path(args.init).expanduser()
        if target.exists() and not args.force:
            print(f"{target} already exists; pass --force to overwrite")
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(SAMPLE_CONFIG, indent=2) + "\n", encoding="utf-8")
        print(f"wrote sample configuration to {target}")
        return 0

    config = load_config(args.config)
    ok, data = _try_http("GET", f"{_base_url(config, args.url)}/api/config")
    if ok:
        _print_json(data["config"])
        return 0
    _print_json(config.redacted())
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ok, data = _try_http("GET", f"{_base_url(config, args.url)}/health")
    if ok:
        if args.json:
            _print_json(data)
        else:
            up = data["upstream"]
            cache = data["cache"]
            print(f"status      {data['status']}")
            print(f"version     {data['version']}")
            print(f"uptime      {data['uptime_seconds']}s")
            print(f"cache       {cache['exact'].get('backend')} "
                  f"entries={cache['exact'].get('entries')} errors={cache['exact'].get('errors')}")
            print(f"semantic    enabled={cache['semantic']['enabled']} "
                  f"backend={cache['semantic']['backend']} entries={cache['semantic']['entries']}")
            print(f"tools       entries={cache['tools']['entries']}")
            print(f"upstream    {'ok' if up.get('ok') else 'FAILED'} {up.get('base_url', '')}"
                  + (f" ({up.get('error')})" if not up.get("ok") else ""))
        return 0 if data.get("status") == "ok" else 1

    print(f"proxy not reachable at {_base_url(config, args.url)}: {data}")
    state = _state_for_offline(config)
    try:
        health = asyncio.run(state.health())
        _print_json(health)
        return 1
    finally:
        asyncio.run(state.shutdown())


def cmd_pricing(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = _base_url(config, args.url)
    if args.model and (args.input is not None or args.output is not None):
        payload = {
            "model": args.model,
            "input_per_1m": args.input or 0.0,
            "output_per_1m": args.output or 0.0,
        }
        ok, data = _try_http("POST", f"{base}/api/pricing", payload)
        if ok:
            print(f"pricing saved for {args.model}")
            return 0
        state = _state_for_offline(config)
        try:
            state.set_pricing(
                args.model,
                ModelPricing(input_per_1m=args.input or 0.0, output_per_1m=args.output or 0.0),
            )
            print(f"pricing saved for {args.model} (local database)")
            return 0
        finally:
            asyncio.run(state.shutdown())

    ok, data = _try_http("GET", f"{base}/api/pricing")
    if ok:
        print(f"currency: {data['currency']}  estimator: {data['token_estimator']}")
        _table(data["models"], ["model", "input_per_1m", "output_per_1m"])
        return 0
    state = _state_for_offline(config)
    try:
        _table(state.costs.table(), ["model", "input_per_1m", "output_per_1m"])
        return 0
    finally:
        asyncio.run(state.shutdown())


def cmd_policy(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    from .policy import PolicyEngine

    engine = PolicyEngine(config.policy, default_ttl=config.cache.default_ttl)
    if args.tool:
        decision = engine.decide_tool(args.tool)
        print(
            f"{args.tool}: cacheable={decision.cacheable} ttl={decision.ttl}s "
            f"category={decision.category} reason={decision.reason}"
        )
        return 0
    _print_json(engine.describe())
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cachellm",
        description="Local caching proxy for OpenAI-compatible LLM APIs",
    )
    parser.add_argument("--version", action="version", version=f"cachellm {__version__}")
    parser.add_argument("--config", help="path to a JSON config file")
    parser.add_argument("--url", help="base URL of a running CacheLLM proxy")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="run the proxy and dashboard")
    start.add_argument("--host")
    start.add_argument("--port", type=int)
    start.add_argument("--upstream", help="UPSTREAM_BASE_URL override")
    start.add_argument("--backend", choices=["memory", "sqlite", "redis"])
    start.add_argument("--ttl", type=int, help="default TTL in seconds")
    start.add_argument("--semantic", action="store_true", help="enable the semantic cache")
    start.add_argument("--no-cache", action="store_true", help="run as a pure passthrough proxy")
    start.add_argument("--log-level")
    start.add_argument("--access-log", action="store_true", help="enable uvicorn access logs")
    start.set_defaults(func=cmd_start)

    stats = sub.add_parser("stats", help="show statistics and savings")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=cmd_stats)

    clear = sub.add_parser("clear", help="invalidate cache entries")
    clear.add_argument("--all", action="store_true", help="clear everything")
    clear.add_argument("--key")
    clear.add_argument("--model")
    clear.add_argument("--namespace")
    clear.add_argument("--tool")
    clear.set_defaults(func=cmd_clear)

    inspect = sub.add_parser("inspect", help="inspect cache entries")
    inspect.add_argument("key", nargs="?", help="cache key (full SHA-256)")
    inspect.add_argument("--search", help="substring search over keys/prompts")
    inspect.add_argument("--limit", type=int, default=25)
    inspect.set_defaults(func=cmd_inspect)

    config_cmd = sub.add_parser("config", help="show or create configuration")
    config_cmd.add_argument("--env", action="store_true", help="list environment variables")
    config_cmd.add_argument("--init", metavar="PATH", nargs="?",
                            const="./cachellm.config.json",
                            help="write a sample config file")
    config_cmd.add_argument("--force", action="store_true")
    config_cmd.set_defaults(func=cmd_config)

    health = sub.add_parser("health", help="check proxy, cache and upstream health")
    health.add_argument("--json", action="store_true")
    health.set_defaults(func=cmd_health)

    pricing = sub.add_parser("pricing", help="view or set model pricing")
    pricing.add_argument("model", nargs="?")
    pricing.add_argument("--input", type=float, help="input price per 1M tokens")
    pricing.add_argument("--output", type=float, help="output price per 1M tokens")
    pricing.set_defaults(func=cmd_pricing)

    policy = sub.add_parser("policy", help="show policy engine settings / test a tool")
    policy.add_argument("tool", nargs="?", help="tool name to evaluate")
    policy.set_defaults(func=cmd_policy)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if os.environ.get("CACHELLM_DEBUG"):
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
