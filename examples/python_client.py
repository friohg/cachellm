"""Python OpenAI SDK against CacheLLM.

    pip install openai
    python examples/python_client.py

Only the base_url changes - everything else is a normal OpenAI SDK program.
"""

from __future__ import annotations

import time

import httpx
from openai import OpenAI

PROXY = "http://localhost:4000/v1"
MODEL = "mock-model"  # whatever your upstream serves

client = OpenAI(
    base_url=PROXY,
    api_key="not-needed-when-the-proxy-holds-the-key",  # SDK requires a value
)

MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Explain what an LLM response cache does, in one sentence."},
]


def timed_call(label: str) -> None:
    started = time.perf_counter()
    completion = client.chat.completions.create(
        model=MODEL, messages=MESSAGES, temperature=0
    )
    elapsed = (time.perf_counter() - started) * 1000
    print(f"{label:<12} {elapsed:7.1f} ms  {completion.choices[0].message.content}")


def show_cache_header() -> None:
    """The raw header tells you exactly which layer answered."""
    response = httpx.post(
        f"{PROXY}/chat/completions",
        json={"model": MODEL, "messages": MESSAGES, "temperature": 0},
        timeout=60,
    )
    print("X-CacheLLM-Cache:", response.headers.get("x-cachellm-cache"))
    print("X-CacheLLM-Key:  ", response.headers.get("x-cachellm-key"))


def streaming_demo() -> None:
    print("\nstreaming (cache replay keeps normal streaming behaviour):")
    stream = client.chat.completions.create(
        model=MODEL, messages=MESSAGES, temperature=0, stream=True
    )
    for chunk in stream:
        piece = chunk.choices[0].delta.content if chunk.choices else None
        if piece:
            print(piece, end="", flush=True)
    print()


def stats() -> None:
    data = httpx.get("http://localhost:4000/api/stats", timeout=10).json()
    counters = data["counters"]
    print(
        f"\nrequests={counters['total_requests']} "
        f"hits={counters['cache_hits']} misses={counters['cache_misses']} "
        f"hit_rate={data['cache_hit_rate_pct']}% "
        f"saved={data['cost']['savings']} {data['cost']['currency']}"
    )


if __name__ == "__main__":
    timed_call("first call")   # CACHE MISS -> upstream
    timed_call("second call")  # CACHE HIT  -> local, much faster
    show_cache_header()
    streaming_demo()
    stats()
