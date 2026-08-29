"""End-to-end smoke test against a RUNNING CacheLLM proxy + mock upstream.

    python examples/mock_upstream.py --port 8099 &
    UPSTREAM_BASE_URL=http://127.0.0.1:8099/v1 cachellm start &
    python examples/e2e_check.py

Exercises the real HTTP path: exact hit, semantic hit, streaming replay,
request coalescing, tool cache, pricing and savings.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time

import httpx

PROXY = "http://127.0.0.1:4000"
MODEL = "mock-model"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {label}{('  -> ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


def chat(messages, **extra):
    body = {"model": MODEL, "messages": messages, "temperature": 0}
    body.update(extra)
    return httpx.post(f"{PROXY}/v1/chat/completions", json=body, timeout=60)


def content(response):
    return response.json()["choices"][0]["message"]["content"]


def main() -> int:
    httpx.post(f"{PROXY}/api/cache/invalidate", json={"all": True}, timeout=30)

    sys_msg = {"role": "system", "content": "You are a concise assistant."}
    q1 = [sys_msg, {"role": "user", "content": "What does an LLM response cache do?"}]

    # 1. exact miss -> hit
    first = chat(q1)
    second = chat(q1)
    check("exact cache: first call is a miss", first.headers["x-cachellm-cache"] == "miss")
    check("exact cache: second call is a hit", second.headers["x-cachellm-cache"] == "exact")
    check("exact cache: identical body returned", content(first) == content(second),
          content(second)[:60])

    # 2. different system prompt must NOT share
    other_sys = [
        {"role": "system", "content": "You are a pirate who answers in verse."},
        q1[1],
    ]
    third = chat(other_sys)
    check("correctness: different system prompt is a miss",
          third.headers["x-cachellm-cache"] == "miss")
    check("correctness: different key", third.headers["x-cachellm-key"] != first.headers["x-cachellm-key"])

    # 3. latency improvement
    t0 = time.perf_counter(); chat(q1); hit_ms = (time.perf_counter() - t0) * 1000
    check("cache hit is fast", hit_ms < 200, f"{hit_ms:.1f} ms")

    # 4. semantic hit (proxy started with SEMANTIC_CACHE=true, threshold 0.85)
    paraphrase = [sys_msg, {"role": "user", "content": "What does an LLM response cache do?  "}]
    sem = chat(paraphrase)
    check("semantic/exact hit on near-identical question",
          sem.headers["x-cachellm-cache"] in {"exact", "semantic"},
          sem.headers["x-cachellm-cache"])

    semq = [sys_msg, {"role": "user", "content": "Tell me how request coalescing helps agents."}]
    chat(semq)
    semq2 = [sys_msg, {"role": "user", "content": "tell me how request coalescing helps agents"}]
    sem2 = chat(semq2)
    check("semantic cache catches a paraphrase",
          sem2.headers["x-cachellm-cache"] == "semantic",
          f"{sem2.headers['x-cachellm-cache']} sim={sem2.headers.get('x-cachellm-similarity')}")

    unrelated = [sys_msg, {"role": "user", "content": "How do I pickle a mango at home?"}]
    unrelated_response = chat(unrelated)
    check("semantic cache does not match unrelated questions",
          unrelated_response.headers["x-cachellm-cache"] == "miss")

    # 5. streaming
    stream_q = [sys_msg, {"role": "user", "content": "Stream a short sentence about caching."}]
    with httpx.stream("POST", f"{PROXY}/v1/chat/completions",
                      json={"model": MODEL, "messages": stream_q, "temperature": 0, "stream": True},
                      timeout=60) as response:
        frames = [line for line in response.iter_lines() if line.startswith("data:")]
        first_cache_header = response.headers["x-cachellm-cache"]
    check("streaming: miss streams from upstream", first_cache_header == "miss")
    check("streaming: terminated with [DONE]", frames[-1].strip() == "data: [DONE]")

    with httpx.stream("POST", f"{PROXY}/v1/chat/completions",
                      json={"model": MODEL, "messages": stream_q, "temperature": 0, "stream": True},
                      timeout=60) as response:
        replay = [line for line in response.iter_lines() if line.startswith("data:")]
        replay_header = response.headers["x-cachellm-cache"]
    check("streaming: second call is replayed from cache", replay_header == "exact")
    check("streaming: replay is valid SSE ending in [DONE]", replay[-1].strip() == "data: [DONE]")

    # 6. request coalescing: 10 identical concurrent requests
    before = httpx.get(f"{PROXY}/api/stats", timeout=30).json()["counters"]["upstream_requests"]
    burst_q = [sys_msg, {"role": "user", "content": f"Coalesce me {time.time()}"}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: chat(burst_q), range(10)))
    after = httpx.get(f"{PROXY}/api/stats", timeout=30).json()["counters"]["upstream_requests"]
    outcomes = [r.headers["x-cachellm-cache"] for r in results]
    bodies = {content(r) for r in results}
    check("coalescing: only one upstream call for 10 identical requests",
          after - before == 1, f"upstream calls={after - before}, outcomes={sorted(set(outcomes))}")
    check("coalescing: all 10 got the same answer", len(bodies) == 1, str(len(bodies)))
    check("coalescing: exactly one request went to upstream",
          outcomes.count("miss") == 1, str(sorted(set(outcomes))))

    # 7. tool cache
    #    'get_weather' is auto-classified read-only; a bare 'weather' name needs an
    #    explicit policy entry (unknown verbs are treated as unsafe by design).
    httpx.post(f"{PROXY}/v1/tools/cache/store",
               json={"tool": "get_weather", "arguments": {"city": "Delhi"},
                     "result": {"temp_c": 41}, "ttl": 300}, timeout=30)
    hit = httpx.post(f"{PROXY}/v1/tools/cache/lookup",
                     json={"tool": "get_weather", "arguments": {"city": "Delhi"}},
                     timeout=30).json()
    check("tool cache: read-only tool hits", hit["hit"] is True and hit["result"]["temp_c"] == 41,
          str(hit.get("reason")))
    reordered = httpx.post(f"{PROXY}/v1/tools/cache/lookup",
                           json={"tool": "get_weather", "arguments": {"city": "Delhi"}},
                           timeout=30).json()
    check("tool cache: repeat lookup still hits", reordered["hit"] is True)
    miss = httpx.post(f"{PROXY}/v1/tools/cache/lookup",
                      json={"tool": "get_weather", "arguments": {"city": "Mumbai"}},
                      timeout=30).json()
    check("tool cache: different arguments miss", miss["hit"] is False)
    unknown = httpx.post(f"{PROXY}/v1/tools/cache/store",
                         json={"tool": "weather", "arguments": {"city": "Delhi"},
                               "result": {}}, timeout=30).json()
    check("tool cache: unknown tool name refused without a policy",
          unknown["stored"] is False, unknown["reason"])
    unsafe = httpx.post(f"{PROXY}/v1/tools/cache/store",
                        json={"tool": "delete_repository", "arguments": {"n": "x"},
                              "result": "gone"}, timeout=30).json()
    check("tool cache: mutation tool refused", unsafe["stored"] is False, unsafe["reason"])

    # 8. pricing + savings
    httpx.post(f"{PROXY}/api/pricing",
               json={"model": MODEL, "input_per_1m": 0.5, "output_per_1m": 1.5}, timeout=30)
    chat(q1)  # a hit that now has a price attached
    stats = httpx.get(f"{PROXY}/api/stats", timeout=30).json()
    check("stats: hits recorded", stats["counters"]["cache_hits"] > 0)
    check("stats: tokens saved recorded", stats["estimated_tokens_saved"] > 0,
          str(stats["estimated_tokens_saved"]))
    check("stats: hit rate computed", stats["cache_hit_rate_pct"] > 0,
          f"{stats['cache_hit_rate_pct']}%")

    # 9. secrets never leak
    config = httpx.get(f"{PROXY}/api/config", timeout=30).text
    check("privacy: upstream key redacted in /api/config", "sk-fake-local-key" not in config)
    check("privacy: no key in response headers/body", "sk-fake-local-key" not in first.text)

    # 10. malformed request
    bad = httpx.post(f"{PROXY}/v1/chat/completions", content=b"{bad json",
                     headers={"Content-Type": "application/json"}, timeout=30)
    check("malformed JSON rejected with 400", bad.status_code == 400)

    # 11. dashboard
    dash = httpx.get(f"{PROXY}/dashboard", timeout=30)
    check("dashboard served", dash.status_code == 200 and "CacheLLM" in dash.text)

    print("\n--- final stats ---")
    print(json.dumps({
        "requests": stats["counters"]["total_requests"],
        "exact_hits": stats["counters"]["exact_hits"],
        "semantic_hits": stats["counters"]["semantic_hits"],
        "tool_hits": stats["counters"]["tool_cache_hits"],
        "coalesced": stats["counters"]["coalesced_requests"],
        "misses": stats["counters"]["cache_misses"],
        "upstream_requests": stats["counters"]["upstream_requests"],
        "hit_rate_pct": stats["cache_hit_rate_pct"],
        "tokens_saved": stats["estimated_tokens_saved"],
        "cost": stats["cost"],
        "latency": stats["latency"],
    }, indent=2))

    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
