"""Streaming support: collect SSE chunks and replay cached responses as SSE.

Two directions:

*collect*   While proxying a live stream we forward every chunk untouched to the
            client and simultaneously accumulate the deltas.  Only when the
            stream terminates cleanly (``data: [DONE]`` and no error chunk) do we
            reconstruct a complete non-streaming response and hand it to the
            cache.  An interrupted stream is never cached.

*replay*    A cache hit for a streaming request is turned back into SSE chunks so
            the client keeps its normal streaming experience.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from .logging_utils import get_logger

log = get_logger("cachellm.streaming")

DONE = "data: [DONE]\n\n"


def sse(data: str) -> str:
    return f"data: {data}\n\n"


def sse_json(payload: Any) -> str:
    return sse(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


# ---------------------------------------------------------------------------
# collecting a live stream
# ---------------------------------------------------------------------------


@dataclass
class _ToolCallAccumulator:
    id: str | None = None
    type: str = "function"
    name: str = ""
    arguments: str = ""

    def to_dict(self, index: int) -> dict[str, Any]:
        return {
            "id": self.id or f"call_{index}",
            "type": self.type,
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class _ChoiceAccumulator:
    role: str = "assistant"
    content: str = ""
    reasoning: str = ""
    refusal: str = ""
    finish_reason: str | None = None
    tool_calls: dict[int, _ToolCallAccumulator] = field(default_factory=dict)
    logprobs: Any = None

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                acc.to_dict(index) for index, acc in sorted(self.tool_calls.items())
            ]
        if self.refusal:
            message["refusal"] = self.refusal
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        return message


class StreamCollector:
    """Accumulates chat-completion SSE deltas into a complete response object."""

    def __init__(self) -> None:
        self.id: str | None = None
        self.model: str | None = None
        self.created: int | None = None
        self.system_fingerprint: str | None = None
        self.service_tier: str | None = None
        self.usage: dict[str, Any] | None = None
        self.choices: dict[int, _ChoiceAccumulator] = {}
        self.completed = False
        self.failed = False
        self.error: dict[str, Any] | None = None
        self.chunk_count = 0
        self.raw_object: str = "chat.completion.chunk"

    # -- ingestion --------------------------------------------------------
    def feed(self, data: str) -> None:
        """Feed one raw SSE ``data:`` payload."""
        text = data.strip()
        if not text:
            return
        if text == "[DONE]":
            self.completed = True
            return
        try:
            chunk = json.loads(text)
        except json.JSONDecodeError:
            log.debug("skipping non-JSON stream chunk")
            return
        if not isinstance(chunk, dict):
            return
        self.chunk_count += 1

        if chunk.get("error"):
            self.failed = True
            self.error = chunk["error"] if isinstance(chunk["error"], dict) else {
                "message": str(chunk["error"])
            }
            return

        self.id = chunk.get("id") or self.id
        self.model = chunk.get("model") or self.model
        self.created = chunk.get("created") or self.created
        self.system_fingerprint = chunk.get("system_fingerprint") or self.system_fingerprint
        self.service_tier = chunk.get("service_tier") or self.service_tier
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]

        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = int(choice.get("index") or 0)
            acc = self.choices.setdefault(index, _ChoiceAccumulator())
            if choice.get("finish_reason"):
                acc.finish_reason = choice["finish_reason"]
            if choice.get("logprobs") is not None:
                acc.logprobs = choice["logprobs"]
            delta = choice.get("delta")
            if delta is None:
                delta = choice.get("message")  # some servers send full messages
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("role"), str):
                acc.role = delta["role"]
            content = delta.get("content")
            if isinstance(content, str):
                acc.content += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        acc.content += part["text"]
            if isinstance(delta.get("reasoning_content"), str):
                acc.reasoning += delta["reasoning_content"]
            if isinstance(delta.get("refusal"), str):
                acc.refusal += delta["refusal"]
            for tool_delta in delta.get("tool_calls") or []:
                if not isinstance(tool_delta, dict):
                    continue
                tc_index = int(tool_delta.get("index") or 0)
                tool_acc = acc.tool_calls.setdefault(tc_index, _ToolCallAccumulator())
                if tool_delta.get("id"):
                    tool_acc.id = tool_delta["id"]
                if tool_delta.get("type"):
                    tool_acc.type = tool_delta["type"]
                function = tool_delta.get("function") or {}
                if isinstance(function, dict):
                    if isinstance(function.get("name"), str):
                        tool_acc.name += function["name"]
                    if isinstance(function.get("arguments"), str):
                        tool_acc.arguments += function["arguments"]

    # -- output -----------------------------------------------------------
    @property
    def cacheable(self) -> bool:
        """Only a cleanly finished, non-empty stream may be cached."""
        if self.failed or not self.completed or not self.choices:
            return False
        return any(
            acc.content or acc.tool_calls or acc.finish_reason
            for acc in self.choices.values()
        )

    def to_response(self, *, model_hint: str = "") -> dict[str, Any]:
        response: dict[str, Any] = {
            "id": self.id or f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": self.created or int(time.time()),
            "model": self.model or model_hint,
            "choices": [],
        }
        for index, acc in sorted(self.choices.items()):
            choice: dict[str, Any] = {
                "index": index,
                "message": acc.to_message(),
                "finish_reason": acc.finish_reason or "stop",
            }
            if acc.logprobs is not None:
                choice["logprobs"] = acc.logprobs
            response["choices"].append(choice)
        if self.usage:
            response["usage"] = self.usage
        if self.system_fingerprint:
            response["system_fingerprint"] = self.system_fingerprint
        if self.service_tier:
            response["service_tier"] = self.service_tier
        return response


# ---------------------------------------------------------------------------
# replaying a cached response as SSE
# ---------------------------------------------------------------------------


def chunks_from_response(
    response: dict[str, Any],
    *,
    include_usage: bool = False,
    split_content: int = 0,
) -> Iterator[str]:
    """Yield SSE frames that reproduce ``response`` as a chat-completion stream.

    ``split_content`` > 0 splits the assistant text into chunks of that many
    characters so clients that render progressively still see incremental output.
    """
    base = {
        "id": response.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": response.get("created") or int(time.time()),
        "model": response.get("model", ""),
    }
    if response.get("system_fingerprint"):
        base["system_fingerprint"] = response["system_fingerprint"]

    choices = response.get("choices") or []
    for choice in choices:
        index = int(choice.get("index") or 0)
        message = choice.get("message") or {}
        yield sse_json(
            {
                **base,
                "choices": [
                    {"index": index, "delta": {"role": message.get("role", "assistant")},
                     "finish_reason": None}
                ],
            }
        )

        content = message.get("content")
        if isinstance(content, str) and content:
            pieces = (
                [content]
                if split_content <= 0
                else [content[i : i + split_content] for i in range(0, len(content), split_content)]
            )
            for piece in pieces:
                yield sse_json(
                    {
                        **base,
                        "choices": [
                            {"index": index, "delta": {"content": piece}, "finish_reason": None}
                        ],
                    }
                )

        for tc_index, tool_call in enumerate(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            yield sse_json(
                {
                    **base,
                    "choices": [
                        {
                            "index": index,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tc_index,
                                        "id": tool_call.get("id"),
                                        "type": tool_call.get("type", "function"),
                                        "function": {
                                            "name": function.get("name", ""),
                                            "arguments": function.get("arguments", ""),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

        yield sse_json(
            {
                **base,
                "choices": [
                    {
                        "index": index,
                        "delta": {},
                        "finish_reason": choice.get("finish_reason") or "stop",
                    }
                ],
            }
        )

    if include_usage and isinstance(response.get("usage"), dict):
        yield sse_json({**base, "choices": [], "usage": response["usage"]})

    yield DONE


async def replay_stream(
    response: dict[str, Any], *, include_usage: bool = False, split_content: int = 0
) -> AsyncIterator[str]:
    for frame in chunks_from_response(
        response, include_usage=include_usage, split_content=split_content
    ):
        yield frame


def error_sse(payload: dict[str, Any]) -> str:
    """Emit an in-band error frame followed by [DONE] (what OpenAI clients expect)."""
    return sse_json(payload) + DONE
