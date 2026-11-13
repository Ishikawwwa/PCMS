"""Bounded SSE assembly for Polza Chat Completions.

The reader consumes the entire HTTP response so a verified persistent connection
can be reused. An incomplete stream never becomes a completion or inferred bill.
No transport, retry, model call, or filesystem operation is performed here.
"""
from __future__ import annotations

import json
import math
import re
from typing import Callable

MAX_LINE_BYTES = 1024 * 1024
MAX_EVENT_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_CHUNKS = 100_000
MAX_CHOICES = 16
MAX_TOOLS_PER_CHOICE = 128


class StreamError(RuntimeError):
    """Failure with an opaque request ID and any already verified final usage.

    ``usage`` is None when billing is unknown, never a fabricated zero. Partial
    generated text or tool calls are deliberately not attached to the exception.
    """
    def __init__(self, message: str, *, request_id=None, usage=None, code="invalid_stream"):
        super().__init__(message)
        self.request_id = request_id
        self.usage = dict(usage) if usage is not None else None
        self.code = code


def _strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = item
        return result

    def constant(value):
        raise ValueError("Nonfinite JSON number")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def _progress_identity(value):
    # Only public identifiers are exposed to callbacks, with token-like strings
    # redacted and arbitrary provider text omitted rather than echoed.
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", value):
        return "[OMITTED]"
    if re.search(r"pza_|sk-|bearer", value, re.IGNORECASE):
        return "[REDACTED]"
    return value


def read_completion_stream(response, on_progress: Callable[[dict], None] | None = None) -> dict:
    """Return one normal completion dict only after valid usage and ``[DONE]``.

    ``response`` must provide bounded ``readline(size)`` like HTTPResponse or
    BytesIO. SSE comments and multiline data fields are supported. Final usage
    must include provider token totals and ``cost_rub``; it follows finished
    choices. A callback sees only sanitized identifiers and numeric counts.
    """
    metadata: dict = {}
    choices: dict[int, dict] = {}
    usage = None
    total_bytes = 0
    chunks = 0
    content_chars = reasoning_chars = tool_argument_chars = 0
    done = False
    event_data: list[str] = []
    event_bytes = 0
    event_name = ""
    first_line = True

    def fail(message, code="invalid_stream"):
        raise StreamError(message, request_id=metadata.get("id"), usage=usage, code=code)

    def identity(chunk):
        for key in ("id", "model", "provider"):
            value = chunk.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value or len(value) > 1024:
                fail("Polza stream contains invalid completion metadata")
            if key in metadata and metadata[key] != value:
                fail(f"Polza stream {key} metadata changed")
            metadata[key] = value
        if chunk.get("created") is not None:
            value = chunk["created"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                fail("Polza stream contains an invalid creation timestamp")
            # The aggregation layer can timestamp each chunk independently.
            # Preserve the first timestamp; it is not a request/model identity.
            metadata.setdefault("created", value)

    def progress():
        if on_progress is None:
            return
        values = {"request_id": _progress_identity(metadata.get("id")),
                  "model": _progress_identity(metadata.get("model")),
                  "provider": _progress_identity(metadata.get("provider")),
                  "chunks": chunks, "bytes_received": total_bytes,
                  "content_chars": content_chars, "reasoning_chars": reasoning_chars,
                  "tool_argument_chars": tool_argument_chars}
        try:
            on_progress(values)
        except Exception:
            fail("Polza stream progress callback failed", "callback_failed")

    def text_delta(delta, key, state):
        nonlocal content_chars, reasoning_chars
        value = delta.get(key)
        if value is None:
            return
        if not isinstance(value, str):
            fail("Polza stream contains a nontext message delta")
        if state["finish_reason"] is not None and value:
            fail("Polza stream emitted text after the choice finished")
        state[key].append(value)
        if key == "content":
            content_chars += len(value)
        else:
            reasoning_chars += len(value)

    def tool_delta(values, state):
        nonlocal tool_argument_chars
        if values is None:
            return
        if not isinstance(values, list):
            fail("Polza stream contains invalid tool-call fragments")
        if values and state["finish_reason"] is not None:
            fail("Polza stream emitted tool calls after the choice finished")
        for value in values:
            if not isinstance(value, dict):
                fail("Polza stream contains invalid tool-call fragments")
            index = value.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < MAX_TOOLS_PER_CHOICE:
                fail("Polza stream contains an invalid tool-call index")
            tool = state["tools"].setdefault(index, {"id": None, "type": "function", "name": [], "arguments": []})
            identifier = value.get("id")
            if identifier is not None:
                if not isinstance(identifier, str) or not identifier or len(identifier) > 1024:
                    fail("Polza stream contains an invalid tool-call identifier")
                if tool["id"] is not None and tool["id"] != identifier:
                    fail("Polza stream tool-call identifier changed")
                tool["id"] = identifier
            if value.get("type") not in (None, "function"):
                fail("Polza stream contains an unsupported tool-call type")
            function = value.get("function", {})
            if not isinstance(function, dict):
                fail("Polza stream contains an invalid tool function")
            for key in ("name", "arguments"):
                fragment = function.get(key)
                if fragment is None:
                    continue
                if not isinstance(fragment, str):
                    fail("Polza stream contains a nontext tool fragment")
                tool[key].append(fragment)
                if key == "arguments":
                    tool_argument_chars += len(fragment)

    def process_event(data, name):
        nonlocal chunks, usage, done
        if name == "error":
            # Some providers identify a request only in their terminal error
            # event. Recover just trusted scalar metadata, never echo its body.
            try:
                error_chunk = _strict_json(data)
            except (ValueError, RecursionError):
                error_chunk = None
            if isinstance(error_chunk, dict):
                identity(error_chunk)
            fail("Polza stream returned an error event", "provider_error")
        if data == "[DONE]":
            if done:
                fail("Polza stream returned duplicate completion markers")
            done = True
            return
        if done:
            fail("Polza stream contains data after its completion marker")
        if usage is not None:
            fail("Polza stream contains completion chunks after final usage")
        try:
            chunk = _strict_json(data)
        except (ValueError, RecursionError):
            fail("Polza stream contains malformed JSON")
        if not isinstance(chunk, dict):
            fail("Polza stream JSON event is not an object")
        identity(chunk)
        if chunk.get("error") is not None:
            fail("Polza stream returned a provider error", "provider_error")
        chunks += 1
        if chunks > MAX_CHUNKS:
            fail("Polza stream exceeded its chunk limit", "size_limit")
        values = chunk.get("choices")
        if not isinstance(values, list):
            fail("Polza stream event lacks a choices list")
        for value in values:
            if not isinstance(value, dict):
                fail("Polza stream contains an invalid choice")
            index = value.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < MAX_CHOICES:
                fail("Polza stream contains an invalid choice index")
            state = choices.setdefault(index, {"content": [], "reasoning_content": [], "reasoning": [],
                                               "tools": {}, "finish_reason": None})
            delta = value.get("delta", {})
            if not isinstance(delta, dict):
                fail("Polza stream contains an invalid delta")
            if delta.get("role") not in (None, "assistant"):
                fail("Polza stream contains an invalid assistant role")
            for key in ("content", "reasoning_content", "reasoning"):
                text_delta(delta, key, state)
            tool_delta(delta.get("tool_calls"), state)
            finished = value.get("finish_reason")
            if finished is not None:
                if not isinstance(finished, str) or not finished:
                    fail("Polza stream contains an invalid finish reason")
                if finished in {"error", "failed", "cancelled", "timeout"}:
                    fail("Polza stream returned a failed completion", "provider_error")
                if state["finish_reason"] is not None and state["finish_reason"] != finished:
                    fail("Polza stream finish reason changed")
                state["finish_reason"] = finished
        final_usage = chunk.get("usage")
        if final_usage is not None:
            if not isinstance(final_usage, dict):
                fail("Polza stream returned invalid usage", "missing_usage")
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                count = final_usage.get(key)
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    fail("Polza stream omitted valid final token totals", "missing_usage")
            if final_usage["total_tokens"] < final_usage["prompt_tokens"] + final_usage["completion_tokens"]:
                fail("Polza stream understated final token totals", "invalid_usage")
            cost = final_usage.get("cost_rub")
            if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost < 0:
                fail("Polza stream omitted valid final RUB billing", "missing_usage")
            if not choices or any(state["finish_reason"] is None for state in choices.values()):
                fail("Polza stream returned usage before every choice finished")
            usage = final_usage
        progress()

    while True:
        try:
            raw_line = response.readline(MAX_LINE_BYTES + 1)
        except Exception:
            fail("Polza completion stream read failed", "read_failed")
        if raw_line in (b"", ""):
            break
        if isinstance(raw_line, str):
            try:
                encoded = raw_line.encode("utf-8")
            except UnicodeError:
                fail("Polza stream contains invalid Unicode")
        elif isinstance(raw_line, bytes):
            encoded = raw_line
        else:
            fail("Polza stream reader returned an invalid line")
        total_bytes += len(encoded)
        if len(encoded) > MAX_LINE_BYTES or total_bytes > MAX_TOTAL_BYTES:
            fail("Polza stream exceeded its byte limit", "size_limit")
        try:
            line = encoded.decode("utf-8-sig" if first_line else "utf-8").rstrip("\r\n")
        except UnicodeError:
            fail("Polza stream contains invalid UTF-8")
        first_line = False
        if not line:
            if event_data:
                process_event("\n".join(event_data), event_name)
            elif event_name == "error":
                fail("Polza stream returned an error event", "provider_error")
            event_data = []
            event_bytes = 0
            event_name = ""
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            event_bytes += len(value.encode("utf-8")) + 1
            if event_bytes > MAX_EVENT_BYTES:
                fail("Polza stream exceeded its event limit", "size_limit")
            event_data.append(value)
        elif field == "event":
            event_name = value
    if event_data or event_name:
        fail("Polza stream ended inside an SSE event", "truncated")
    if not done:
        fail("Polza stream ended without [DONE]", "truncated")
    if not choices or any(state["finish_reason"] is None for state in choices.values()):
        fail("Polza stream ended without a finish reason", "truncated")
    if usage is None:
        fail("Polza stream omitted final usage; billing is unknown", "missing_usage")
    if sorted(choices) != list(range(len(choices))):
        fail("Polza stream contains an incomplete choice index set")
    assembled = []
    for index, state in sorted(choices.items()):
        message = {"role": "assistant", "content": "".join(state["content"]) if state["content"] else None}
        for key in ("reasoning_content", "reasoning"):
            if state[key]:
                message[key] = "".join(state[key])
        if state["tools"]:
            if sorted(state["tools"]) != list(range(len(state["tools"]))):
                fail("Polza stream contains an incomplete tool-call index set")
            tool_calls = []
            identifiers = set()
            for _, tool in sorted(state["tools"].items()):
                name = "".join(tool["name"])
                arguments = "".join(tool["arguments"])
                if not tool["id"] or tool["id"] in identifiers or not name:
                    fail("Polza stream ended with an incomplete tool call")
                try:
                    parsed_arguments = _strict_json(arguments)
                except (ValueError, RecursionError):
                    fail("Polza stream ended with malformed tool arguments")
                if not isinstance(parsed_arguments, dict):
                    fail("Polza stream tool arguments are not an object")
                identifiers.add(tool["id"])
                tool_calls.append({"id": tool["id"], "type": "function", "function": {"name": name, "arguments": arguments}})
            message["tool_calls"] = tool_calls
        assembled.append({"index": index, "message": message, "finish_reason": state["finish_reason"]})
    return {**metadata, "object": "chat.completion", "choices": assembled, "usage": usage}
