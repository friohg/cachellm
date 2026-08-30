# CacheLLM

If you're paying OpenAI or Anthropic directly, prompt caching is handled for you and you can stop reading. This is for the rest of us.

The cheap and free OpenAI-compatible providers — the reverse proxies, the aggregators, the community endpoints — almost never implement prompt caching. Meanwhile a coding agent will happily re-send the same 30,000-token system prompt and tool schema forty times in an afternoon, and every one of those is billed at full price or burns another credit off your quota. That's the gap this fills. CacheLLM sits between your agent and whatever provider you're using, keeps a local copy of answers it has already seen, and stops the duplicates from ever leaving your machine.

It runs on your laptop. SQLite file, one Python process, no accounts, nothing phoning home.

```
your agent  ->  localhost:4000/v1  ->  CacheLLM  ->  whatever provider you use
                                          |
                                    already seen it?
                                    then don't ask again
```

The one thing that matters more than saving credits is not giving you a wrong answer. A cache that occasionally returns the response from someone else's system prompt is worse than no cache at all, so the cache key covers everything that could change the reply — model, every message, system and developer instructions, tool schemas, response format, temperature, top_p, seed, the lot. Same question with a different system prompt is a different key. Always.

## What it actually does

**Remembers exact repeats.** The bread and butter. Identical request, identical answer, served from disk in about 10ms instead of a few seconds. Key ordering and trailing whitespace don't count as different; casing and wording do.

**Merges duplicate calls in flight.** If ten identical requests land at once — which happens constantly with parallel agent workers — one goes to the provider and the other nine wait for it. Without this you pay ten times for one answer.

**Recognises reworded questions, if you ask it to.** Off by default. Turn it on and a paraphrase of something you already asked can hit the cache, but only inside an identical context: same model, same system prompt, same tools, same conversation so far. Embeddings run locally, no model download needed.

**Caches tool results separately.** `get_weather("Delhi")` for five minutes, `list_repositories()` for thirty seconds, `delete_repository(...)` never. Agents call tools far more often than they call the model, and each tool has its own idea of "stale".

**Refuses to cache things it shouldn't.** Anything that creates, deletes, sends, pays or executes is never cached. Neither is "what's the price right now". Unknown tool names are treated as dangerous until you say otherwise.

**Streams properly.** A cache miss streams through untouched while quietly being recorded. A cache hit is replayed as SSE, so your client's `for chunk in stream` loop doesn't know the difference. A stream that dies halfway is thrown away, never stored.

**Tells you what you saved.** Requests, hit rate, tokens saved, money saved, latency split between cache and provider. You supply the prices — the tool ships none, since every provider charges differently.

**Doesn't fall over.** If the cache backend breaks, requests go straight to the provider instead of your agent dying. Failed responses are never stored. Nothing is left half-locked.

Storage is SQLite by default because this is a local tool. Plain in-memory and Redis are there if you want them.

## Getting it

Python 3.10 or newer. That's the only requirement — SQLite comes with Python and the default similarity matching needs no downloaded model.

```bash
git clone https://github.com/beyondsighttech/cachellm
cd cachellm
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

On Windows with Git Bash, `.venv/Scripts/python.exe -m pip install -e .` works too. If the `cachellm` command doesn't end up on your PATH, `python -m cachellm` does the same thing everywhere.

Optional extras, only if you want them:

```bash
pip install -e ".[semantic]"   # real local embedding model instead of the built-in one
pip install -e ".[redis]"      # Redis instead of SQLite
pip install -e ".[tokens]"     # tiktoken for exact token counts
pip install -e ".[dev]"        # pytest, to run the test suite
```

## Using it with Hermes Agent

One command:

```bash
cachellm launch hermes
```

That brings the cache up if it isn't already running, works out which provider you're using by reading your existing Hermes config, creates a Hermes profile called `cachellm` pointed at the proxy, and drops you into a normal Hermes session. Your usual profile is never touched — if you decide you hate it, `hermes profile delete cachellm` and you're back to exactly where you were.

```
  using the upstream from your Hermes config: https://api.your-provider.com/v1
  starting the cache at http://127.0.0.1:4000  (log: ~/.cachellm/proxy.log)
  cache is up
  Hermes profile 'cachellm' (created) points at the cache
  dashboard: http://127.0.0.1:4000/dashboard
```

Run it again later and it reuses both:

```
  cache already running at http://127.0.0.1:4000
  Hermes profile 'cachellm' (reused) points at the cache
```

Anything after `--` is handed straight to Hermes:

```bash
cachellm launch hermes -- -q "summarise this repo"
cachellm launch hermes -- --tui
```

Useful flags: `--use-profile` uses a profile you already work in instead (see below), `--semantic` turns on reworded-question matching, `--ttl 7200` changes how long answers live, `--from-profile work` reads the provider out of a different Hermes profile, `--upstream https://...` overrides the provider entirely.

### One thing you need to know about agents

Hermes attaches its whole toolset — `terminal`, `write_file`, `delete_*` — to every single call. The cautious default sees those and refuses to cache, which means the biggest, most expensive request your agent makes never gets cached at all. You'd see this in the log:

```
CACHE SKIP reason=mutation_or_unsafe_tool category=mutation
```

`cachellm launch` turns this on for you, because otherwise the tool is pointless for agents. If you're running `cachellm start` by hand, set it yourself:

```bash
CACHE_RESPONSES_WITH_TOOLS=true cachellm start
```

This is safe, and it's worth understanding why: what gets cached is the model's *decision* to call a tool, not the result of running it. A cache hit executes nothing — Hermes still runs the tool itself, fresh, every time. The tool-result cache is a completely separate thing and still refuses mutations no matter what; `cachellm policy terminal` will tell you `cacheable=False` either way.

If you'd rather stay cautious, `cachellm launch --strict-tools` keeps the old behaviour.

### What it looked like here

Same prompt, three runs through Hermes against a live provider:

| Run | Wall clock | What the proxy did |
|---|---|---|
| first | 8.5 s | miss (went to the provider) |
| second | 4.7 s | hit |
| third | 4.9 s | hit |

```
requests            6
served from cache   4  (66%)
went to provider    2
tokens saved        73,454
would have cost     0.557 USD
saved               0.557 USD
```

A different question correctly missed and got a different key, which is the behaviour that matters. Two proxy requests per Hermes turn is normal — one streamed tool-capable call plus a follow-up — and both are cached independently.

## Using it with anything else

Point the client's base URL at the proxy. That's the whole integration.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="anything",           # the proxy holds the real key
)
```

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:4000/v1",
  apiKey: "anything",
});
```

Runnable versions of both are in `examples/`. For CLI tools that read `OPENAI_BASE_URL`, `cachellm launch` can wrap them:

```bash
cachellm launch aider --model gpt-4o
cachellm launch -- some-other-tool --flag
```

By default the proxy holds your provider key and throws away whatever `Authorization` header the client sent. If you'd rather each client bring its own key, set `FORWARD_CLIENT_KEY=true`.

Endpoints: `POST /v1/chat/completions` and `POST /v1/responses` go through the full cache; `GET /v1/models` and `POST /v1/embeddings` pass straight through.

## Two things people ask

**"If the system prompt is cached, does my agent lose context?"**

No, and it's worth being precise about why, because the word "caching" means two different things here.

What OpenAI and Anthropic do is *server-side prompt caching*: they keep the KV state of your prompt prefix on their GPUs so they don't have to recompute it. You still send the whole prompt every time; they just charge you less for the part they recognise.

This does something completely different. It's a lookup table sitting in front of the provider. Your agent sends the full request — system prompt, every message, tool schemas, all of it — exactly as it always did. Nothing is stripped out, summarised or held back. Then one of two things happens:

- The request is identical to one seen before, so the stored answer is handed back and the provider is never contacted.
- It isn't, so the entire request goes to the provider untouched.

The model never sees a partial conversation, because the cache never *edits* a request. It only ever answers it or forwards it whole.

And the moment your agent adds a turn, the request is different, so it's a miss and goes upstream with the full history. That's the normal case in a conversation: turn one might hit, turn two is new and can't. The hits you actually get are the genuinely repeated calls — the same file read twice, the same classification step, a retry after a tool error, parallel workers doing identical setup.

Demonstrated rather than argued. Here's the model recalling something only reachable through the full thread, on a cache miss:

```
system:    "SYSTEM_MARKER_ABC. Answer with one word only."
user:      "pineapple"
assistant: "one"
user:      "Repeat the exact marker string from your system prompt."
assistant: "SYSTEM_MARKER_ABC"
user:      "What was my very first message? One word."

reply:     pineapple
```

Both the system prompt and the first user message arrived intact. And on a hit you get the *whole* response object back — same content, `finish_reason`, `usage`, everything:

```
x-cachellm-cache: exact
x-cachellm-age: 34.9
reply: SYSTEM_MARKER_ABC   usage present: True   finish_reason: stop
```

There are tests pinning this down — `test_full_conversation_is_forwarded_on_a_miss`, `test_second_turn_of_a_conversation_still_sends_everything`, `test_tool_schemas_are_forwarded_intact`, `test_cache_hit_returns_the_whole_response_not_a_fragment` — so it can't quietly break later.

The only fields deliberately left out of the cache key are ones that can't change the reply: `stream`, `stream_options`, `user`, `metadata`. They're still forwarded; they just don't split the cache, which is why a streamed and a non-streamed version of the same question share one entry.

**"Can I use my own profile instead of a new one?"**

Yes:

```bash
cachellm launch hermes --use-profile          # your default profile
cachellm launch hermes --use-profile work     # a named one
```

This changes that profile's `model.base_url` to point at the cache, and writes down where it used to point. Everything else — sessions, memory, skills, keys — is untouched. When you want out:

```bash
cachellm unlink            # restore everything cachellm repointed
cachellm unlink work       # just that one
cachellm unlink --list     # what's repointed right now, and where from
```

A real round trip:

```
BEFORE: https://api.justwoker.icu/v1

  cachellm-test now goes through the cache
  its provider was https://api.justwoker.icu/v1
  put it back any time with:  cachellm unlink cachellm-test

AFTER launch: http://127.0.0.1:4000/v1
AFTER unlink: https://api.justwoker.icu/v1
```

Two things worth knowing. Running `launch --use-profile` twice won't clobber the saved original — there's a test for that, because getting it wrong would make `unlink` silently useless. And a repointed profile has nowhere to send requests if the proxy isn't running; `cachellm launch` starts it for you, but if you run `hermes` directly, either have the proxy up or `cachellm unlink` first.

The default is still the throwaway profile, because that's the safer thing to hand someone who's only trying this out.

## Running it by hand

If you don't want `launch` to manage things:

```bash
UPSTREAM_BASE_URL=https://api.your-provider.com/v1 \
UPSTREAM_API_KEY=your-key \
cachellm start
```

Windows PowerShell:

```powershell
$env:UPSTREAM_BASE_URL="https://api.your-provider.com/v1"
$env:UPSTREAM_API_KEY="your-key"
cachellm start
```

Any OpenAI-compatible endpoint works — that's rather the point. OpenRouter, Together, Groq, DeepSeek, a reverse proxy someone posted on Discord, or something local like Ollama (`http://localhost:11434/v1`), vLLM, LM Studio, llama.cpp.

To try it without any provider at all, there's a fake one included:

```bash
python examples/mock_upstream.py --port 8099              # terminal 1
UPSTREAM_BASE_URL=http://127.0.0.1:8099/v1 cachellm start # terminal 2
python examples/e2e_check.py                              # terminal 3
```

`e2e_check.py` hammers the running proxy and checks the lot — hits, misses, streaming replay, coalescing, tool caching, key redaction, savings.

## Telling whether it worked

Three ways, pick whichever suits.

**The response header.** Every reply carries `X-CacheLLM-Cache`, one of `miss`, `exact`, `semantic`, `coalesced` or `error`.

```bash
curl -s -D - -o /dev/null http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model","messages":[{"role":"user","content":"hello"}],"temperature":0}' \
  | grep -i x-cachellm
```

First time:
```
x-cachellm-cache: miss
x-cachellm-key: 3b63f7ccf0eca8261d586e49b8abc7e1
```

Run it again, same command:
```
x-cachellm-cache: exact
x-cachellm-key: 3b63f7ccf0eca8261d586e49b8abc7e1
x-cachellm-age: 4.2
```

Same key, and now a hit. `x-cachellm-age` is how many seconds ago it was stored. Reworded matches also get `x-cachellm-similarity`.

**The log.** The proxy narrates what it's doing, one line per event, with a request id you can grep for:

```
18:10:31 CACHE MISS       rid=8c1f2a key=3b63f7cc lookup_ms=1.400
18:10:31 UPSTREAM REQUEST rid=8c1f2a model=your-model stream=False
18:10:32 CACHE WRITE      rid=8c1f2a key=3b63f7cc ttl=3600 category=static
18:10:36 CACHE HIT        rid=91ade4 key=3b63f7cc tier=l1 age_s=4.2 lookup_ms=0.010
```

The vocabulary is small and greppable: `CACHE HIT`, `CACHE MISS`, `SEMANTIC HIT`, `TOOL CACHE HIT`, `UPSTREAM REQUEST`, `UPSTREAM ERROR`, `CACHE WRITE`, `CACHE SKIP`, `CACHE ERROR`, `CACHE INVALIDATE`, `COALESCED`, `STREAM ABORT`. Authorization headers and anything shaped like a key are stripped before anything is printed.

**The numbers.**

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
  avg upstream        3874 ms
```

That last gap — 4ms versus 3.8 seconds — is the whole story.

If you want to convince yourself it's *safe* rather than just fast, send the same user message twice with different system prompts. You should get two misses and two different keys. There's a test for exactly that, `test_different_system_prompt_never_shares_cache`.

## The dashboard

<http://localhost:4000/dashboard>. Six tabs: overview, request log, cache browser, tool cache, pricing, configuration.

The overview has the savings panel — what you actually spent, what it would have cost with everything going upstream, the difference, and the percentage. Money reads zero until you tell it your prices, because nobody's prices are the same:

```bash
cachellm pricing your-model --input 0.15 --output 0.60
cachellm pricing                       # see what's set
```

Prices go in SQLite and survive restarts. Token counts come from the provider's own `usage` field when it sends one; otherwise they're estimated and marked as such.

The cache browser lets you search by key, prompt or response text, look inside an entry, delete one, or wipe a model, tool or namespace. The configuration tab shows the caching rules, the effective config with secrets masked, and every environment variable with a plain-language description.

Everything the page does is a JSON endpoint you can hit yourself: `/api/stats`, `/api/requests`, `/api/cache`, `/api/cache/{key}`, `/api/cache/invalidate`, `/api/tools/cache`, `/api/config`, `/api/pricing`, `/api/policy`, `/api/maintenance/purge`, `/health`.

### Accessibility

Treated as a requirement, not a nice-to-have, and checked rather than asserted:

- Proper semantic HTML throughout — real `<button>` elements, real `<table>` with `<caption>` and scoped `<th>`, real landmarks, one `<h1>`, no skipped heading levels.
- The tab strip implements the ARIA tabs pattern properly: left/right arrows move between tabs, Home and End jump to the ends, and only the current tab is in the tab order.
- A skip link as the first focusable thing, visible when focused.
- Every input has a real `<label>`. Buttons generated in JavaScript get accessible names describing what they act on, so "Delete" doesn't announce as just "Delete".
- Actions you take are announced through a polite live region; errors go to an assertive one. Auto-refresh deliberately does *not* announce anything, because a table quietly rewriting itself every five seconds would make the page unusable with a screen reader. You can switch auto-refresh off entirely, and it starts off if your system asks for reduced motion.
- No information carried by colour alone. Every result has a text label — "hit (exact)", "miss", "not cacheable" — plus a shape.
- Focus is always visible, with a 3px outline. Focus is moved deliberately (inspecting an entry moves you to its heading) and never stolen otherwise.
- Light and dark themes, both meeting WCAG AA. Measured: the worst pair is 6.54:1 in light mode and 8.85:1 in dark, against a 4.5:1 requirement. `prefers-contrast: more` switches borders to `currentColor`.
- Everything sizes in `rem`, so browser and OS text scaling work.
- Content is built with `textContent`, never `innerHTML` — safer, and no stray markup confusing a reader.

There's an audit script that enforces all of it:

```bash
python tools/a11y_audit.py                                    # the shipped files
python tools/a11y_audit.py --url http://localhost:4000/dashboard   # a running one
# 36 checks passed, 0 failed
```

The same checks run in the test suite (`tests/test_launch_and_accessibility.py`), so a regression fails CI rather than quietly shipping.

If something is still awkward with your screen reader, that's a bug — please say so, and be specific about which reader and which part.

## Configuring it

Defaults, then a JSON config file, then environment variables. Env always wins. The file is looked for at `$CACHELLM_CONFIG`, `./cachellm.config.json`, `./config/cachellm.json`, `~/.cachellm/config.json`.

```bash
cachellm config --init      # write a starter cachellm.config.json
cachellm config             # what's actually in effect (secrets masked)
cachellm config --env       # every variable, explained
```

A minimal setup:

```ini
UPSTREAM_BASE_URL=https://api.your-provider.com/v1
UPSTREAM_API_KEY=your-key
PORT=4000
CACHE_BACKEND=sqlite
SQLITE_PATH=./data/cache.db
DEFAULT_TTL=3600
CACHE_RESPONSES_WITH_TOOLS=true    # if an agent is the client
SEMANTIC_CACHE=false
SEMANTIC_THRESHOLD=0.92
```

`.env.example` has the full list with comments. The ones you're most likely to touch:

| Variable | What it does |
|---|---|
| `UPSTREAM_BASE_URL` | Where to forward on a miss |
| `UPSTREAM_API_KEY` | Your provider key. Never logged, never returned, never written to the database |
| `PORT`, `HOST` | Where the proxy listens. Defaults to `127.0.0.1:4000` |
| `CACHE_BACKEND` | `sqlite` (default), `memory`, or `redis` |
| `DEFAULT_TTL` | How long answers live, in seconds |
| `CACHE_RESPONSES_WITH_TOOLS` | Cache replies for requests carrying tool schemas. Needed for agents |
| `AGENT_TOOL_CALL_TTL` | Lifetime for those, default 300 |
| `SEMANTIC_CACHE` | Turn on reworded-question matching |
| `SEMANTIC_THRESHOLD` | How similar counts as a match, 0 to 1 |
| `CACHE_NAMESPACE` | Keeps separate agents' caches apart |
| `CACHE_FAIL_OPEN` | On by default: if the cache breaks, keep serving from the provider |
| `FORWARD_CLIENT_KEY` | Let each client bring its own key instead of using yours |
| `STORE_PROMPTS`, `STORE_RESPONSES`, `STORE_TOOL_ARGUMENTS`, `STORE_TOOL_RESULTS` | Privacy switches |
| `RETENTION_DAYS` | How long history is kept. `0` keeps it forever |

Per-request overrides, if you want fine control from the client side:

| Header | Effect |
|---|---|
| `Cache-Control: no-store` | Skip the cache this once |
| `X-CacheLLM-TTL: 300` | Different lifetime for this answer. `0` means don't store it |
| `X-CacheLLM-Namespace: tenant-b` | Use a separate cache |

Model routing, if you want short names or want to swap the real model without touching the client:

```json
{ "routing": { "model_map": { "fast": "upstream-small", "smart": "upstream-large" } } }
```

## What gets cached, and for how long

| Category | Default lifetime | When it applies |
|---|---|---|
| `static` | 24 hours | `temperature: 0`, or a seed is set — deterministic, so cache it hard |
| `general` | 1 hour | Ordinary questions. Eligible for reworded matching |
| `read_only_tool` | 5 minutes | Every tool in the request is a read verb |
| `search` | 1 minute | Tool names with search, browse, web, crawl, news in them |
| `current_information` | never | "today", "right now", "current price", "latest news" |
| `mutation` | never | Anything that creates, deletes, sends, pays, deploys, executes |
| `agent_tool_call` | 5 minutes, opt-in | Like `mutation`, but only when `CACHE_RESPONSES_WITH_TOOLS=true` |

Verb matching looks at whole words, so `get_updates` is a read but `update_settings` is a mutation. Unknown verbs are assumed dangerous. Check any tool with `cachellm policy <name>`.

Per-tool rules go in the config file:

```json
{
  "policy": {
    "tool_policies": {
      "get_weather":       { "cacheable": true,  "ttl": 300 },
      "list_repositories": { "cacheable": true,  "ttl": 30 },
      "get_file":          { "cacheable": true,  "ttl": 60,
                             "include_context_keys": ["repo", "ref"] },
      "delete_repository": { "cacheable": false, "ttl": 0 }
    },
    "unsafe_tool_deny_list": ["*payment*", "*transfer*", "*refund*"]
  }
}
```

`include_context_keys` is how you say "this tool's answer depends on which repo we're in" without letting unrelated context fragment the cache.

To use the tool cache from an agent, ask before running and offer the result afterwards:

```bash
curl -s localhost:4000/v1/tools/cache/lookup -H 'Content-Type: application/json' \
  -d '{"tool":"get_weather","arguments":{"city":"Delhi"}}'

curl -s localhost:4000/v1/tools/cache/store -H 'Content-Type: application/json' \
  -d '{"tool":"get_weather","arguments":{"city":"Delhi"},"result":{"temp_c":41}}'
```

`examples/tool_cache_client.py` is a working version of that pattern.

## Clearing things out

```bash
cachellm clear --key <full-sha256>
cachellm clear --model your-model
cachellm clear --tool get_weather
cachellm clear --namespace tenant-b
cachellm clear --all
```

TTLs handle the rest — expired entries are skipped on read and swept every five minutes.

## The rest of the commands

```
cachellm launch     start the cache and hand it to an app (hermes by default)
cachellm unlink     put a repointed Hermes profile back
cachellm start      just run the proxy and dashboard
cachellm stats      hit rate, tokens saved, money saved, latency
cachellm clear      throw entries away
cachellm inspect    list entries, or look inside one
cachellm config     show / --init / --env
cachellm health     is the proxy up, is the provider reachable
cachellm pricing    set or list model prices
cachellm policy     what would you do with this tool?
```

The read-only ones talk to a running proxy if there is one and read the SQLite file directly if there isn't, so `cachellm stats` works either way.

## Privacy and a security warning

Everything stays on your machine. Nothing is uploaded anywhere.

**There is no authentication on the proxy.** On `127.0.0.1` that's fine. If you change `HOST` to `0.0.0.0` — including if you run the Docker image — then anyone who can reach that port can spend your provider credits and read your cached prompts. Put it behind something, or firewall it.

Beyond that:

- Your provider key lives in memory only. It's never written to the database, never logged, never included in a response. The tests check this.
- Log output goes through a redactor that masks `Authorization`, `api-key`, `x-api-key`, cookies, and anything shaped like `Bearer …` or `sk-…`.
- The database layer refuses to store a config value whose key looks like a credential, so the dashboard can't accidentally persist a secret.
- You can turn off storing prompts, responses, tool arguments and tool results independently. The cache keeps working; you just lose the ability to browse the text. Aggregate stats keep working too.
- Cached data isn't encrypted at rest. Treat `data/cache.db` as sensitive — it contains your prompts.

## Running it as a service

Linux or WSL, systemd user unit:

```ini
[Unit]
Description=CacheLLM
[Service]
WorkingDirectory=/home/you/cachellm
Environment=UPSTREAM_BASE_URL=https://api.your-provider.com/v1
Environment=UPSTREAM_API_KEY=your-key
Environment=CACHE_RESPONSES_WITH_TOOLS=true
ExecStart=/home/you/cachellm/.venv/bin/cachellm start
Restart=on-failure
[Install]
WantedBy=default.target
```

If your agent runs on Windows and CacheLLM in WSL, start it with `HOST=0.0.0.0` so Windows can reach it — and see the warning above.

Docker, if you'd rather:

```bash
docker build -t cachellm .
docker run -p 4000:4000 -v cachellm-data:/data \
  -e UPSTREAM_BASE_URL=https://api.your-provider.com/v1 \
  -e UPSTREAM_API_KEY=your-key \
  cachellm
```

`docker compose up -d` works too, and there's an optional Redis profile (`--profile redis`). No Kubernetes, no cluster, nothing you have to learn.

## When things go wrong

**Upstream errors** come back with the provider's status code and a clean JSON body, are never cached, and are shared with everyone waiting on the same coalesced request — so one failure doesn't turn into ten retries. Timeouts become 504, connection failures 502.

**Interrupted streams** are dropped. If a stream doesn't finish with `data: [DONE]`, or carries an error frame, nothing is stored and the next identical request is a fresh miss.

**A broken cache backend** is logged as `CACHE ERROR`, counted, and stepped around — requests go straight to the provider. Caching should never be the thing that takes your agent down. If you'd rather know loudly, set `fail_open: false` and you'll get a `503` instead.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
# 191 passed
```

They run the real proxy against a real in-process fake provider wired together with `httpx.ASGITransport`, so requests go through the actual HTTP layer, the actual streaming code and the actual cache engine. Nothing important is mocked out.

The interesting ones: different system prompts never sharing an answer, tool schema changes producing different keys, ten concurrent identical requests producing exactly one upstream call, truncated streams not being cached, single-flight slots being released after an exception, the cache backend failing both open and closed, mutating tools being refused, the full conversation and all tool schemas arriving at the provider unmodified on a miss, profile repointing surviving being run twice, and the dashboard's accessibility contract.

## How it's put together

```
cachellm/
  server/          the HTTP layer, admin API, and dashboard
  cache/
    engine.py      ties policy, exact, semantic and coalescing together
    keys.py        cache keys and context fingerprints
    exact.py       the two-tier exact cache
    semantic.py    context-scoped similarity lookup
    tools.py       tool-result cache
    backends.py    memory / sqlite / redis
  policy.py        what may be cached, and for how long
  normalize.py     canonical JSON, conservative normalisation
  extract.py       pulling messages, system prompts and tools out of payloads
  embeddings.py    local hashing embedder, sentence-transformers, OpenAI
  providers/       provider adapters
  pricing.py       token counting and cost maths
  stats.py         counters and the request log
  db.py            SQLite schema and queries
  singleflight.py  request coalescing
  launch.py        cachellm launch and unlink
  config.py        layered configuration
  logging_utils.py structured logs and redaction
  cli.py           the command line
examples/          clients, tool-cache harness, fake provider, end-to-end check
tools/             the accessibility audit
tests/             191 tests
```

Adding another provider means writing one adapter class and registering it:

```python
from cachellm.providers import register_provider

class MyProvider:
    name = "my_provider"
    async def post_json(self, path, payload, *, headers=None): ...
    def stream_json(self, path, payload, *, headers=None): ...   # yields StreamChunk
    async def list_models(self): ...
    async def health(self): ...
    async def close(self): ...

register_provider("my_provider", lambda cfg, rt, ct: MyProvider(cfg))
```

Then `UPSTREAM_PROVIDER=my_provider`. Nothing in the cache engine changes — adapters only translate the wire format.

## License

MIT. Do what you like with it.

