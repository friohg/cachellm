# CacheLLM

A local-first caching proxy for OpenAI-compatible LLM APIs. Point your agent at
`http://localhost:4000/v1` instead of your provider and CacheLLM removes the
repeated upstream calls that dominate agent bills — even when the provider has
no prompt caching of its own.

It is not just "detect the same prompt twice". The goal is fewer expensive
upstream calls **without ever returning an answer that was generated for a
different context**, in this priority order: correctness, cache safety, request
coalescing, exact caching, tool-result caching, semantic caching, observability,
easy deployment.

```
Agent / OpenAI SDK
        |
        v
http://localhost:4000/v1        <- CacheLLM proxy (FastAPI)
        |
        |-- exact cache      (SHA-256 of the full request context)
        |-- semantic cache   (optional, local embeddings, context-scoped)
        |-- tool cache       (per-tool TTL policies)
        |-- single-flight    (10 identical concurrent calls -> 1 upstream call)
        |
        v
Upstream OpenAI-compatible provider
```

## What it does

| Feature | Notes |
|---|---|
| Exact response cache | SHA-256 over canonical JSON of model + messages + system/developer instructions + tools & schemas + `response_format` + generation params. Configurable TTL. |
| Backends | in-memory LRU, **SQLite (default)**, optional Redis. An in-process L1 LRU always sits in front of the durable backend. |
| Semantic cache | Opt-in. Local embeddings, cosine similarity, and a **context fingerprint** so a paraphrase can only match inside an identical system prompt / model / tools / history. |
| Agent-aware policies | Categories: `static`, `general`, `read_only_tool`, `search`, `current_information`, `mutation`. Mutating and unknown tools are never cached. |
| Tool-result cache | Separate subsystem keyed on tool name + canonical arguments + declared context, with per-tool TTLs. |
| Streaming | Real SSE passthrough while collecting chunks; only a cleanly finished stream is cached. Cache hits are replayed as SSE. |
| Request coalescing | Single-flight: duplicate concurrent requests wait for the leader and share its result. |
| Statistics & cost | Real provider usage when available, configurable estimation otherwise. Per-model pricing you set — no provider prices are hardcoded. |
| Dashboard | `/dashboard`: hit rate, tokens/money saved, latency, recent requests, cache browser, pricing and runtime config pages. |
| Privacy | Local-only. Prompt/response/tool-argument/tool-result storage individually disableable; aggregate stats keep working. API keys are never logged, never returned, never stored in the DB. |
| Failure handling | Errors are never cached; `fail_open` keeps the agent working when the cache backend breaks; no stale single-flight locks. |

## Install

Requires Python 3.10+. Nothing else — SQLite is stdlib and the default semantic
backend needs no model download.

### Linux / macOS / WSL

```bash
git clone <your-repo> cachellm && cd cachellm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Windows (PowerShell)

```powershell
git clone <your-repo> cachellm; cd cachellm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Windows (Git Bash)

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
```

Optional extras:

```bash
pip install -e ".[semantic]"   # sentence-transformers embedding backend
pip install -e ".[redis]"      # Redis cache backend
pip install -e ".[tokens]"     # tiktoken token counting
pip install -e ".[dev]"        # pytest + pytest-asyncio (to run the tests)
```

If the `cachellm` script is not on your PATH, `python -m cachellm ...` is
equivalent everywhere.

## Run it

Set the upstream and start:

```bash
export UPSTREAM_BASE_URL="https://api.your-provider.com/v1"
export UPSTREAM_API_KEY="your-upstream-key"
cachellm start
```

PowerShell:

```powershell
$env:UPSTREAM_BASE_URL="https://api.your-provider.com/v1"
$env:UPSTREAM_API_KEY="your-upstream-key"
cachellm start
```

```
CacheLLM 0.1.0
  proxy      http://127.0.0.1:4000/v1
  dashboard  http://127.0.0.1:4000/dashboard
  cache      sqlite (./data/cache.db)
  ttl        3600s
  semantic   off
  upstream   https://api.your-provider.com/v1
```

Any OpenAI-compatible upstream works — OpenAI, OpenRouter, Together, Groq,
DeepSeek, Mistral, vLLM, Ollama (`http://localhost:11434/v1`), LM Studio,
llama.cpp. `UPSTREAM_PROVIDER` selects the adapter; they all speak the same wire
format, and new shapes plug in via `cachellm/providers/`.

### Try it with no provider at all

```bash
python examples/mock_upstream.py --port 8099          # terminal 1
UPSTREAM_BASE_URL=http://127.0.0.1:8099/v1 cachellm start   # terminal 2
python examples/e2e_check.py                           # terminal 3
```

`e2e_check.py` asserts exact hits, semantic hits, streaming replay, coalescing,
tool caching, redaction and savings against the running proxy.

## Connect a client

Only `base_url` changes. Full runnable versions: `examples/python_client.py`,
`examples/js_client.mjs`.

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="not-needed-when-the-proxy-holds-the-key",  # SDK requires a value
)

completion = client.chat.completions.create(
    model="your-model",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain what an LLM response cache does."},
    ],
    temperature=0,
)
print(completion.choices[0].message.content)
```

### JavaScript / TypeScript

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:4000/v1",
  apiKey: "not-needed-when-the-proxy-holds-the-key",
});

const completion = await client.chat.completions.create({
  model: "your-model",
  messages: [
    { role: "system", content: "You are a concise assistant." },
    { role: "user", content: "Explain what an LLM response cache does." },
  ],
  temperature: 0,
});
console.log(completion.choices[0].message.content);
```

Streaming works normally in both — a cache hit is replayed as SSE, so
`for await (const chunk of stream)` behaves exactly as before.

By default the proxy holds the upstream key and **discards** any client
`Authorization` header. Set `FORWARD_CLIENT_KEY=true` if each agent should bring
its own key instead.

### Supported endpoints

| Endpoint | Behaviour |
|---|---|
| `POST /v1/chat/completions` | Full cache pipeline, streaming and non-streaming |
| `POST /v1/responses` | Same pipeline, `/v1/responses` shape (separate keyspace) |
| `GET /v1/models` | Passthrough |
| `POST /v1/embeddings` | Passthrough (not cached by default) |
| `POST /v1/tools/cache/{lookup,store,policy}` | Tool-result cache API |

## Using it with Hermes Agent

Verified working end-to-end with Hermes Agent (`claude-opus-5-thinking` over a
custom OpenAI-compatible provider).

Point a throwaway Hermes profile at the proxy so your default profile is
untouched:

```bash
hermes profile create cachellm-test --clone
hermes --profile cachellm-test config set model.base_url http://127.0.0.1:4000/v1
```

Start the proxy with the same upstream your Hermes profile normally uses, and
**turn on `CACHE_RESPONSES_WITH_TOOLS`**:

```bash
UPSTREAM_BASE_URL=https://your-provider/v1 \
UPSTREAM_API_KEY=*** \
CACHE_RESPONSES_WITH_TOOLS=true \
AGENT_TOOL_CALL_TTL=1800 \
cachellm start
```

Then use it normally: `hermes --profile cachellm-test`.

### Why that flag matters for agents

Hermes attaches its **entire toolset** — `terminal`, `write_file`, `delete_*` —
to every model call. The default conservative rule sees mutating tool schemas and
refuses to cache, so the agent's largest and most expensive request (~29k prompt
tokens in the measured run) is never cached:

```
CACHE SKIP reason=mutation_or_unsafe_tool category=mutation
```

`CACHE_RESPONSES_WITH_TOOLS=true` reclassifies those as `agent_tool_call` and
caches them. This is safe because **the cached artefact is the model's decision,
not the effect of a tool**: replaying a hit executes nothing, Hermes still runs
the tool itself. The tool-result cache is a separate subsystem and still refuses
mutations unconditionally — `cachellm policy terminal` reports `cacheable=False`
either way.

### Measured result

The same prompt run three times through Hermes:

| Run | Wall time | Proxy outcome |
|---|---|---|
| A (cold cache) | 8.5 s | `miss` ×2 (agent_tool_call + general) |
| B | 4.7 s | `exact_hit` ×2 |
| C | 4.9 s | `exact_hit` ×2 |

```
requests            6
cache hits          4  (66% hit rate)
upstream requests   2
tokens saved        73454
cost without cache  0.557025 USD
savings             0.557025 USD (100.0%)
```

A different prompt correctly produced a fresh `miss` with different keys — the
cache did not answer a new question with an old answer.

Two Hermes calls per turn is expected: one streamed tool-capable call plus one
non-streamed follow-up. Both are cached independently.

## Verifying a CACHE HIT

Three independent ways.

**1. Response header.** Every response carries `X-CacheLLM-Cache`, which is one
of `miss`, `exact`, `semantic`, `coalesced` or `error`:

```bash
curl -s -D - -o /dev/null http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model","messages":[{"role":"user","content":"hello"}],"temperature":0}' \
  | grep -i x-cachellm
```

First call:
```
x-cachellm-cache: miss
x-cachellm-key: 3b63f7ccf0eca8261d586e49b8abc7e1
```
Run the identical command again:
```
x-cachellm-cache: exact
x-cachellm-key: 3b63f7ccf0eca8261d586e49b8abc7e1
x-cachellm-age: 4.2
```
Same key, `exact` instead of `miss`, and `x-cachellm-age` = seconds since it was
stored. Semantic hits also include `x-cachellm-similarity`.

**2. Server logs.** The proxy prints structured events with request ids:

```
18:10:31 INFO  cachellm CACHE MISS rid=8c1f2a key=3b63f7ccf0eca826 lookup_ms=1.400
18:10:31 INFO  cachellm UPSTREAM REQUEST rid=8c1f2a model=your-model stream=False
18:10:32 INFO  cachellm CACHE WRITE rid=8c1f2a key=3b63f7ccf0eca826 ttl=3600 category=static
18:10:36 INFO  cachellm CACHE HIT rid=91ade4 key=3b63f7ccf0eca826 tier=l1 age_s=4.2 lookup_ms=0.010
```

Event vocabulary: `CACHE HIT`, `CACHE MISS`, `SEMANTIC HIT`, `TOOL CACHE HIT`,
`TOOL CACHE MISS`, `TOOL CACHE WRITE`, `UPSTREAM REQUEST`, `UPSTREAM ERROR`,
`CACHE WRITE`, `CACHE SKIP`, `CACHE ERROR`, `CACHE INVALIDATE`, `COALESCED`,
`STREAM ABORT`. Authorization headers and key-shaped strings are redacted before
formatting.

**3. Latency and counters.** A hit is served locally, typically 5–20 ms versus
hundreds or thousands for upstream:

```bash
cachellm stats
```
```
  requests            23
  cache hits          17  (73.91% hit rate)
    exact             9
    semantic          2
    tool              2
    coalesced         4
  cache misses        6
  upstream requests   6
  tokens saved        645
  savings             3.9e-05 USD (100.0%)
  avg latency         10.76 ms
  avg cache lookup    4.0 ms
  avg upstream        6.34 ms
```

The dashboard's Recent activity table shows the same outcome per request, and
the Cache browser lets you find the entry by key or prompt text.

To prove *cache safety* rather than hit rate: send the same user message with a
different `system` prompt and confirm you get `miss` and a different
`x-cachellm-key`. That is exercised by
`tests/test_exact_cache.py::test_different_system_prompt_never_shares_cache`.

## Dashboard and savings

Open <http://localhost:4000/dashboard>.

**Overview** — request count, cache hit rate, exact / semantic / tool / coalesced
hit counts, tokens saved, money saved, upstream calls, average total / cache
lookup / upstream latency, per-model usage, and a live recent-activity feed.

**Savings card** — the money numbers:

- *Actual cost* — what you really spent upstream.
- *Cost without cache* — what the same traffic would have cost with every
  request going upstream.
- *Saved* and *Percent saved* — the difference, with a progress bar.

Money saved is `0` until you configure prices, because CacheLLM ships **no**
provider prices. Set them on the **Pricing** tab (model, input per 1M tokens,
output per 1M tokens) or from the CLI:

```bash
cachellm pricing your-model --input 0.15 --output 0.60
cachellm pricing                       # list what is configured
```

Prices are stored in SQLite and survive restarts. Token counts come from the
provider's `usage` field when present; otherwise they are estimated
(`TOKEN_ESTIMATOR=heuristic` with `chars_per_token`, or `tiktoken` if installed).
Rows flagged as estimated are marked in `/api/requests`.

Other tabs: **Requests** (filter the log by outcome), **Cache browser**
(search / inspect / delete entries, invalidate by model, tool or namespace, clear
all), **Tool cache** (entries plus a policy checker), **Configuration** (runtime
settings, the policy engine dump, the effective config with secrets redacted, and
the environment-variable reference).

Everything the UI does is also plain JSON: `/api/stats`, `/api/requests`,
`/api/cache`, `/api/cache/{key}`, `/api/cache/invalidate`, `/api/tools/cache`,
`/api/config`, `/api/pricing`, `/api/policy`, `/api/maintenance/purge`, `/health`.

## Configuration

Layered: **defaults → JSON config file → environment variables** (env always
wins). The config file is searched at `$CACHELLM_CONFIG`,
`./cachellm.config.json`, `./config/cachellm.json`, `~/.cachellm/config.json`.

```bash
cachellm config --init          # writes ./cachellm.config.json
cachellm config                 # show the effective config (secrets redacted)
cachellm config --env           # list every environment variable
```

Minimal `.env`-style setup (see `.env.example` for all of it):

```ini
CACHE_BACKEND=sqlite
SQLITE_PATH=./data/cache.db
SEMANTIC_CACHE=false
SEMANTIC_THRESHOLD=0.92
DEFAULT_TTL=3600
UPSTREAM_BASE_URL=https://api.your-provider.com/v1
UPSTREAM_API_KEY=your-upstream-key
PORT=4000
```

### Environment variables

| Variable | Meaning |
|---|---|
| `PORT`, `HOST`, `LOG_LEVEL` | Proxy listen address (default `127.0.0.1:4000`) and log level |
| `UPSTREAM_BASE_URL` | OpenAI-compatible base URL, e.g. `https://api.example.com/v1` |
| `UPSTREAM_API_KEY` | Upstream key — never returned to clients, logged, or stored in the DB |
| `UPSTREAM_PROVIDER` | Adapter name (default `openai_compatible`) |
| `FORWARD_CLIENT_KEY` | `true` forwards each client's own `Authorization` header |
| `CACHE_BACKEND` | `memory` \| `sqlite` (default) \| `redis` |
| `SQLITE_PATH` / `REDIS_URL` | Backend location |
| `DEFAULT_TTL` | Default TTL in seconds (`3600`) |
| `CACHE_NAMESPACE` | Namespace prefix isolating entries (`default`) |
| `CACHE_ENABLED` | `false` = pure passthrough proxy |
| `CACHE_FAIL_OPEN` | `true` = on cache backend failure still serve from upstream |
| `CACHE_STREAMING` | `false` disables caching of streamed responses |
| `CACHE_RESPONSES_WITH_TOOLS` | `true` caches LLM replies for requests carrying mutating tool schemas — needed for agent frameworks like Hermes |
| `AGENT_TOOL_CALL_TTL` | TTL for the `agent_tool_call` category (default `300`) |
| `SEMANTIC_CACHE` | `true` enables the semantic cache (default `false`) |
| `SEMANTIC_THRESHOLD` | Cosine similarity threshold, 0–1 (`0.92`) |
| `SEMANTIC_BACKEND` | `hash` \| `sentence_transformers` \| `openai` |
| `SEMANTIC_MODEL`, `EMBED_BASE_URL`, `EMBED_API_KEY` | Embedding backend settings |
| `SINGLE_FLIGHT` | `false` disables request coalescing (not recommended) |
| `STORE_PROMPTS`, `STORE_RESPONSES`, `STORE_TOOL_ARGUMENTS`, `STORE_TOOL_RESULTS` | Privacy switches |
| `RETENTION_DAYS` | History retention; `0` = keep forever |
| `DASHBOARD_ENABLED` | `false` disables `/dashboard` |
| `TOKEN_ESTIMATOR` | `heuristic` \| `tiktoken` |
| `CACHELLM_CONFIG` | Path to the JSON config file |

### Per-request overrides

| Header / field | Effect |
|---|---|
| `Cache-Control: no-store` | Bypass the cache for this request |
| `X-CacheLLM-Cache: off` | Same |
| `X-CacheLLM-TTL: 300` | Override the TTL (`0` = do not cache) |
| `X-CacheLLM-Namespace: tenant-b` | Use a different cache namespace |
| body `cachellm_no_cache`, `cachellm_ttl`, `cachellm_namespace` | Same three knobs in JSON; stripped before forwarding upstream |

### Model routing

```json
{ "routing": { "model_map": { "fast": "upstream-small", "smart": "upstream-large" },
               "allow_unlisted_models": true } }
```

The agent asks for `fast`; the proxy checks the cache under the resolved upstream
model and, on a miss, calls upstream with `upstream-small`. Set
`allow_unlisted_models: false` to reject models that are not in the table.

## How caching decides

### Cache key (correctness)

The key is `SHA-256` over canonical JSON of a document containing the namespace,
the endpoint, and the normalized request: resolved model, all messages, top-level
`instructions`/`system`, every tool and its full JSON schema, `tool_choice`,
`response_format`/`text`, and the generation parameters (`temperature`, `top_p`,
`top_k`, `n`, `max_tokens`, `stop`, `seed`, penalties, `logit_bias`, `logprobs`,
`reasoning_effort`, …). Unrecognised fields are kept too — an unknown knob might
matter.

Excluded because they cannot change the content: `stream`, `stream_options`,
`user`, `metadata`, `store`, and the `cachellm_*` control fields. That is why a
streamed and non-streamed version of the same request share one entry.

Normalization (in `cachellm/normalize.py`) is deliberately conservative:
canonical JSON with sorted keys, trimmed leading/trailing whitespace of message
text, and re-serialized tool-call arguments so `{"b":2,"a":1}` and
`{"a": 1, "b": 2}` hash alike. Casing, punctuation, interior whitespace and
message order are **never** touched.

### Policy engine

| Category | Default TTL | Triggered by |
|---|---|---|
| `static` | 86400s | `temperature == 0` or a `seed` is set |
| `general` | 3600s | ordinary Q&A (semantic cache eligible) |
| `read_only_tool` | 300s | every tool in the request is a read verb |
| `search` | 60s | tool names containing search/browse/web/crawl/news |
| `current_information` | 0 (never) | "today", "right now", "current price", "latest news", … |
| `mutation` | 0 (never) | any create/delete/update/send/purchase/execute/deploy… tool |
| `agent_tool_call` | 300s, **opt-in** | as `mutation`, but only when `CACHE_RESPONSES_WITH_TOOLS=true` — see the Hermes section |

Hard rules that override everything: any mutation-capable or deny-listed tool
makes the request uncacheable; unknown tool verbs are treated as unsafe; audio
modalities and streamed `logprobs` are not cached; `no-store` is honoured.

Verb matching is word-boundary aware, so `get_updates` is read-only while
`update_settings` is a mutation. Override anything in `policy` in the config file
and inspect the result with `cachellm policy <tool>` or `GET /api/policy`.

### Tool-result cache

Key = `SHA-256(namespace | tool name | canonical arguments | declared context)`.
Only the context keys a tool's policy declares are folded in, so noise like a
trace id cannot fragment the cache.

```json
{
  "policy": {
    "tool_policies": {
      "get_weather":       { "cacheable": true,  "ttl": 300 },
      "list_repositories": { "cacheable": true,  "ttl": 30 },
      "get_file":          { "cacheable": true,  "ttl": 60,
                             "include_context_keys": ["repo", "ref"] },
      "delete_repository": { "cacheable": false, "ttl": 0 },
      "internal_*":        { "cacheable": true,  "ttl": 15 }
    },
    "unsafe_tool_deny_list": ["*payment*", "*transfer*", "*refund*"]
  }
}
```

Usage from an agent — ask before executing, offer the result afterwards
(`examples/tool_cache_client.py` is a working harness):

```bash
curl -s localhost:4000/v1/tools/cache/lookup -H 'Content-Type: application/json' \
  -d '{"tool":"get_weather","arguments":{"city":"Delhi"}}'
# -> {"hit":false,"cacheable":true,"ttl":300,"category":"read_only_tool", ...}

curl -s localhost:4000/v1/tools/cache/store -H 'Content-Type: application/json' \
  -d '{"tool":"get_weather","arguments":{"city":"Delhi"},"result":{"temp_c":41}}'
```

`delete_repository` returns `{"stored": false, "cacheable": false}` — mutations
are refused even if you ask nicely. `POST /v1/tools/cache/policy` answers "what
would you do with this tool?" without touching the cache.

### Semantic cache

Opt-in (`SEMANTIC_CACHE=true`). Flow: extract the final user turn → embed locally
→ compare against vectors that share the **same context fingerprint** → return
the cached response if similarity ≥ threshold.

The context fingerprint covers model, system/developer instructions, tool schemas,
`tool_choice`, `response_format`, generation params and the prior conversation.
Two users asking the same question under different system instructions therefore
have different fingerprints and cannot collide. Additional guards: tool-bearing
requests are excluded (`allow_tools`), multi-turn conversations are excluded
(`require_single_turn`), and very short or very long queries are skipped.

Backends:

- `hash` (default) — dependency-free local hashing embedder; deterministic,
  offline, no downloads. Good for paraphrase/typo/casing differences.
- `sentence_transformers` — a real local model (`pip install "cachellm[semantic]"`).
- `openai` — any OpenAI-compatible `/embeddings` endpoint via `EMBED_BASE_URL`.

Start around `0.95` and lower it deliberately; a low threshold is how you get
wrong answers.

## Invalidation

```bash
cachellm clear --key <full-sha256>      # one entry (also drops its vector)
cachellm clear --model your-model       # everything for a model
cachellm clear --tool get_weather       # a tool's results
cachellm clear --namespace tenant-b     # a namespace
cachellm clear --all                    # responses + tool results
```

Or over HTTP:

```bash
curl -s localhost:4000/api/cache/invalidate -H 'Content-Type: application/json' \
  -d '{"model":"your-model"}'
# -> {"removed":{"cache_entries":6,"tool_entries":0,"semantic_entries":0}}
```

TTL handles the rest: expired entries are skipped on read and swept by a
background maintenance pass every 5 minutes (`POST /api/maintenance/purge` to
force it), which also applies `RETENTION_DAYS` to the request history.

## CLI

```bash
cachellm start      # run the proxy + dashboard
cachellm stats      # statistics and savings
cachellm clear      # invalidate entries
cachellm inspect    # list entries, or inspect one by key
cachellm config     # show / --init / --env
cachellm health     # proxy, cache and upstream health
cachellm pricing    # view or set per-model prices
cachellm policy     # dump the policy engine, or test one tool name
```

`start` flags: `--host --port --upstream --backend --ttl --semantic --no-cache
--log-level --access-log`.

The read-only commands talk to a running proxy over HTTP and **fall back to
reading the local SQLite file** when it is not up, so `cachellm stats` and
`cachellm inspect` work either way.

## Concurrency and failure behaviour

**Coalescing.** Ten identical requests arriving together produce one upstream
call: the first becomes the leader, the others await its result and are reported
as `coalesced`. The response is written to the cache *inside* the flight slot, and
a late arrival re-checks the cache before becoming a new leader — so there is no
window where a second upstream call can slip through. Verified by
`test_concurrent_identical_requests_are_coalesced` and by `examples/e2e_check.py`
against the live proxy.

**Failures.** Upstream errors are returned with the upstream status code and a
clean JSON body (`error.source = "cachellm.upstream"`), never cached, and shared
with every coalesced waiter so one failure does not become ten. Slots are released
in a `finally`, so an exception cannot leave a stale lock. Timeouts map to `504`,
connection failures to `502`.

**Streams.** Chunks are forwarded as they arrive and accumulated in parallel. The
reconstructed response is cached only if the stream ended with `data: [DONE]`,
carried no error frame, and produced content. A truncated or errored stream leaves
the cache untouched — the next identical request is a fresh `miss`.

**Cache backend down.** With `CACHE_FAIL_OPEN=true` (default) a broken backend is
logged as `CACHE ERROR`, counted, and the request proceeds upstream: caching never
becomes a single point of failure. With `fail_open=false` you get an explicit
`503 cache_backend_error` instead of a silent degradation.

## Security and privacy

CacheLLM runs on your machine and binds `127.0.0.1` by default. It has **no
authentication** — anyone who can reach the port can use your upstream key and
read cached prompts. If you change `HOST` to `0.0.0.0` (including the Docker
setup), put it behind a reverse proxy with auth or restrict it at the firewall.

- The upstream API key lives in memory only. It is never written to the database,
  never logged, never included in a response body or header. Tests assert this.
- `configuration` rows whose key looks like a credential are rejected at the DB
  layer, so the dashboard cannot persist a secret even by accident.
- Log output passes through a redactor that masks `Authorization`, `api-key`,
  `x-api-key`, cookies, and inline `Bearer …` / `sk-…` shapes.
- `/api/config` returns the effective config with keys masked (`sk-t...alue`).
- Privacy switches: `STORE_PROMPTS`, `STORE_RESPONSES`, `STORE_TOOL_ARGUMENTS`,
  `STORE_TOOL_RESULTS`. Turning them off keeps the cache working (the payload the
  cache needs is still stored) but stops persisting prompt text, response
  excerpts, or tool arguments for browsing — and aggregate statistics still work.
- `RETENTION_DAYS` prunes request history; `0` keeps it forever.
- Client `Authorization` headers are dropped by default rather than forwarded.
- Cached data is not encrypted at rest. Treat `data/cache.db` as sensitive.

## Deployment

### Linux / macOS / WSL

```bash
UPSTREAM_BASE_URL=... UPSTREAM_API_KEY=... cachellm start
```

For a background service, systemd user unit:

```ini
[Unit]
Description=CacheLLM caching proxy
[Service]
WorkingDirectory=/home/you/cachellm
Environment=UPSTREAM_BASE_URL=https://api.your-provider.com/v1
Environment=UPSTREAM_API_KEY=your-key
ExecStart=/home/you/cachellm/.venv/bin/cachellm start
Restart=on-failure
[Install]
WantedBy=default.target
```

Under WSL, `localhost:4000` is reachable from Windows apps too. If your agent runs
on Windows and CacheLLM in WSL, start it with `HOST=0.0.0.0`.

### Windows

```powershell
$env:UPSTREAM_BASE_URL="https://api.your-provider.com/v1"
$env:UPSTREAM_API_KEY="your-key"
cachellm start
```

The SQLite file works fine on NTFS; use forward slashes or escaped backslashes in
`SQLITE_PATH`. `%LOCALAPPDATA%\cachellm\cache.db` is a reasonable location.

### Docker

```bash
docker build -t cachellm .
docker run -p 4000:4000 -v cachellm-data:/data \
  -e UPSTREAM_BASE_URL=https://api.your-provider.com/v1 \
  -e UPSTREAM_API_KEY=your-key \
  cachellm
```

Or `docker compose up -d` (reads `UPSTREAM_BASE_URL` / `UPSTREAM_API_KEY` from
your environment or `.env`). Redis is available as an optional profile:

```bash
docker compose --profile redis up -d
# then set CACHE_BACKEND=redis and REDIS_URL=redis://redis:6379/0
```

No Kubernetes, no orchestration, no external services required.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
# 147 passed
```

The suite runs the real proxy against a real in-process fake upstream wired
together with `httpx.ASGITransport`, so requests traverse the actual HTTP layer,
streaming code and cache engine — no monkeypatched internals.

Covered: exact hit / miss, key-order and whitespace insensitivity, **different
system prompts never sharing a cached response**, developer-role isolation,
different models, different generation params, different `response_format`,
different tool schemas, conversation-history isolation, namespace isolation,
`no-store`, key redaction, malformed bodies, `/v1/responses`, TTL expiry,
invalidation by key / model / namespace / tool / all, semantic hits plus every
semantic safety boundary, streaming (replay, truncated, in-band error, usage
frames, stream-flag key equivalence), concurrent identical requests, shared
failures, single-flight lock release, upstream failures, missing upstream config,
cache-backend failure with fail-open on and off, unsafe tools, per-tool TTLs,
tool context keys, statistics, real-vs-estimated tokens, cost math, pricing
persistence, dashboard endpoints, privacy switches, and config layering.

`examples/e2e_check.py` is the end-to-end complement: 28 assertions against a
running proxy.

## Project layout

```
cachellm/
  server/            proxy + admin routes, FastAPI app, dashboard HTML
    proxy.py         /v1/chat/completions, /v1/responses, tool cache API
    admin.py         /health, /api/* (stats, cache browser, config, pricing)
  cache/
    engine.py        orchestrates policy + exact + semantic + coalescing
    keys.py          cache key & context fingerprint construction
    exact.py         two-tier exact cache with fail-open handling
    semantic.py      context-scoped vector lookup
    tools.py         tool-result cache
    backends.py      memory / sqlite / redis
  policy.py          category + per-tool caching decisions
  normalize.py       canonical JSON and conservative normalization
  extract.py         message/system/tool extraction helpers
  embeddings.py      hash / sentence-transformers / OpenAI embedders
  providers/         provider adapter protocol + OpenAI-compatible adapter
  pricing.py         token counting and cost calculation
  stats.py           counters, latency windows, request log
  db.py              SQLite schema and queries
  singleflight.py    request coalescing
  config.py          layered configuration
  logging_utils.py   structured logging + redaction
  cli.py             cachellm command line
examples/            python/js clients, tool-cache harness, mock upstream, e2e check
tests/               147 tests
```

Database tables: `cache_entries`, `semantic_entries`, `tool_cache_entries`,
`requests`, `usage`, `model_pricing`, `configuration`, `schema_meta`.

## Adding a provider

```python
from cachellm.providers import register_provider
from cachellm.providers.base import ProviderResponse, StreamChunk

class MyProvider:
    name = "my_provider"
    async def post_json(self, path, payload, *, headers=None) -> ProviderResponse: ...
    def stream_json(self, path, payload, *, headers=None): ...   # yields StreamChunk
    async def list_models(self) -> ProviderResponse: ...
    async def health(self) -> dict: ...
    async def close(self) -> None: ...

register_provider("my_provider", lambda cfg, rt, ct: MyProvider(cfg))
```

Then `UPSTREAM_PROVIDER=my_provider`. Nothing in the cache engine or HTTP layer
needs to change; adapters only translate the wire format.

## License

MIT.

