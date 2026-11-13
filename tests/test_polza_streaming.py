"""Synthetic protocol fixtures; no credentials, network, or model calls."""
import io
import json

import pytest

import constraint_memory.streaming as module
from constraint_memory.streaming import StreamError, read_completion_stream

USAGE = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20, "cost_rub": 0.031,
         "completion_tokens_details": {"reasoning_tokens": 3}}


def event(value):
    return "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"


def chunk(delta=None, finish=None, **extra):
    return {"id": "gen-fixture", "model": "qwen/qwen3.6-27b", "provider": "chutes/fp8",
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **extra}


def usage_event(**extra):
    return {"id": "gen-fixture", "model": "qwen/qwen3.6-27b", "provider": "chutes/fp8",
            "choices": [], "usage": USAGE, **extra}


def response(*values, done=True):
    content = "".join(event(value) for value in values)
    if done:
        content += event("[DONE]")
    return io.BytesIO(content.encode())


def test_content_reasoning_usage_and_safe_progress_are_preserved():
    progress = []
    stream = response(chunk({"role": "assistant", "reasoning_content": "private reasoning ", "reasoning": "other "}),
                      chunk({"reasoning_content": "continued", "reasoning": "thought", "content": "Hello "}),
                      chunk({"content": "world"}, "stop"), usage_event())
    raw = read_completion_stream(stream, progress.append)
    message = raw["choices"][0]["message"]
    assert message["content"] == "Hello world"
    assert message["reasoning_content"] == "private reasoning continued"
    assert message["reasoning"] == "other thought"
    assert raw["id"] == "gen-fixture" and raw["provider"] == "chutes/fp8"
    assert raw["usage"] == USAGE and raw["choices"][0]["finish_reason"] == "stop"
    assert len(progress) == 4 and progress[-1]["content_chars"] == 11
    assert progress[-1]["reasoning_chars"] == len("private reasoning continuedother thought")
    assert "Hello" not in json.dumps(progress) and "private reasoning" not in json.dumps(progress)
    assert stream.read() == b""  # response drained for connection reuse


def test_per_chunk_creation_timestamps_do_not_change_request_identity():
    raw=read_completion_stream(response(chunk({"content":"ok"},created=10),
        chunk({},"stop",created=11),usage_event(created=12)))
    assert raw["id"]=="gen-fixture" and raw["created"]==10
    assert raw["choices"][0]["message"]["content"]=="ok"


def test_parallel_tool_calls_aggregate_name_and_argument_fragments_by_index():
    first = [{"index": 1, "id": "call-b", "type": "function", "function": {"name": "Accommodation", "arguments": '{"city":'}},
             {"index": 0, "id": "call-a", "type": "function", "function": {"name": "Restaurant", "arguments": '{"city":"Rock'}}]
    second = [{"index": 0, "function": {"name": "Search", "arguments": 'ford"}'}},
              {"index": 1, "id": "call-b", "function": {"name": "Search", "arguments": '"Rockford"}'}}]
    raw = read_completion_stream(response(chunk({"tool_calls": first}), chunk({"tool_calls": second}, "tool_calls"), usage_event()))
    message = raw["choices"][0]["message"]
    assert message["content"] is None
    assert [call["id"] for call in message["tool_calls"]] == ["call-a", "call-b"]
    assert [call["function"]["name"] for call in message["tool_calls"]] == ["RestaurantSearch", "AccommodationSearch"]
    assert all(json.loads(call["function"]["arguments"]) == {"city": "Rockford"} for call in message["tool_calls"])


def test_sse_comments_multiline_data_bom_and_crlf():
    text = '\ufeff: heartbeat\r\n\r\ndata: {"id":"gen-fixture",\r\ndata: "choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\r\n\r\n'
    text += event(usage_event()) + event("[DONE]") + ": trailing comment\n\n"
    assert read_completion_stream(io.BytesIO(text.encode()))["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize("stream", [
    response(chunk({"content": "partial"}), done=False),
    response(chunk({"content": "partial"}), done=True),
    response(chunk({"content": "partial"}, "stop"), done=True),
    io.BytesIO((event(chunk({"content": "ok"}, "stop")) + event(usage_event()) + "data: [DONE]\n").encode()),
])
def test_incomplete_stream_never_returns_partial_completion_or_zero_billing(stream):
    with pytest.raises(StreamError) as caught:
        read_completion_stream(stream)
    error = caught.value
    assert error.request_id == "gen-fixture"
    assert not hasattr(error, "content") and not hasattr(error, "choices")
    assert error.usage is None or error.usage == USAGE
    assert "partial" not in str(error)


@pytest.mark.parametrize("value", [
    "{malformed json}",
    {"id": "gen-fixture", "error": {"message": "secret prompt must not leak"}},
    chunk({"content": ["not text"]}),
    chunk({}, "error"),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": 12}}]}),
    {"id": "gen-fixture", "choices": "wrong type"},
])
def test_invalid_or_provider_error_events_are_rejected_without_body_echo(value):
    with pytest.raises(StreamError) as caught:
        read_completion_stream(response(value))
    assert "secret prompt" not in str(caught.value)


@pytest.mark.parametrize("key,value", [("id", "different-request"), ("model", "wrong/model"), ("provider", "changed")])
def test_contradictory_completion_metadata_is_rejected(key, value):
    changed = chunk({"content": "second"}, "stop")
    changed[key] = value
    with pytest.raises(StreamError, match="metadata changed") as caught:
        read_completion_stream(response(chunk({"content": "first"}), changed, usage_event()))
    assert caught.value.request_id == "gen-fixture"


@pytest.mark.parametrize("changes", [{"cost_rub": None}, {"cost_rub": -1}, {"prompt_tokens": True},
                                       {"total_tokens": 1}, {"completion_tokens": None}])
def test_missing_or_invalid_final_usage_is_never_fabricated(changes):
    bad_usage = {**USAGE, **changes}
    with pytest.raises(StreamError) as caught:
        read_completion_stream(response(chunk({"content": "ok"}, "stop"), usage_event(usage=bad_usage)))
    assert caught.value.usage is None


def test_malformed_finished_tool_arguments_are_rejected_with_known_billing():
    tool = {"index": 0, "id": "call-a", "function": {"name": "Search", "arguments": '{"city":'}}
    with pytest.raises(StreamError, match="malformed tool arguments") as caught:
        read_completion_stream(response(chunk({"tool_calls": [tool]}, "tool_calls"), usage_event()))
    assert caught.value.usage == USAGE


def test_read_failure_preserves_request_id_and_does_not_expose_socket_error():
    class Interrupted:
        def __init__(self):
            self.prefix = io.BytesIO(event(chunk({"content": "partial"})).encode())

        def readline(self, limit):
            line = self.prefix.readline(limit)
            if not line:
                raise TimeoutError("secret socket details")
            return line

    with pytest.raises(StreamError) as caught:
        read_completion_stream(Interrupted())
    assert caught.value.request_id == "gen-fixture" and caught.value.code == "read_failed"
    assert "secret socket details" not in str(caught.value)


def test_first_error_event_can_supply_the_request_id_without_echoing_its_body():
    payload = {"id": "gen-failed", "error": {"message": "secret provider body"}}
    stream = io.BytesIO(("event: error\n" + event(payload)).encode())
    with pytest.raises(StreamError) as caught:
        read_completion_stream(stream)
    assert caught.value.request_id == "gen-failed"
    assert caught.value.usage is None and "secret provider body" not in str(caught.value)


def test_line_event_and_total_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(module, "MAX_LINE_BYTES", 80)
    with pytest.raises(StreamError, match="byte limit"):
        read_completion_stream(io.BytesIO(b"data: " + b"a" * 100 + b"\n\n"))
    monkeypatch.setattr(module, "MAX_LINE_BYTES", 1000)
    monkeypatch.setattr(module, "MAX_EVENT_BYTES", 20)
    with pytest.raises(StreamError, match="event limit"):
        read_completion_stream(io.BytesIO(b"data: " + b"a" * 21 + b"\n\n"))
    monkeypatch.setattr(module, "MAX_EVENT_BYTES", 1000)
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", 30)
    with pytest.raises(StreamError, match="byte limit"):
        read_completion_stream(io.BytesIO(b": heartbeat\n" * 10))


def test_data_after_done_or_final_usage_is_rejected():
    stream = event(chunk({"content": "ok"}, "stop")) + event(usage_event()) + event("[DONE]") + event(chunk())
    with pytest.raises(StreamError, match="after its completion marker"):
        read_completion_stream(io.BytesIO(stream.encode()))
    with pytest.raises(StreamError, match="after final usage"):
        read_completion_stream(response(chunk({}, "stop"), usage_event(), chunk()))


def test_callback_identifiers_are_sanitized_and_callback_failures_preserve_billing():
    values = []
    first = chunk({}, "stop", provider="pza_not_a_real_key")
    final = usage_event(provider="pza_not_a_real_key")
    read_completion_stream(response(first, final), values.append)
    assert all(value["provider"] == "[REDACTED]" for value in values)
    def callback(value):
        if value["chunks"] == 2:
            raise RuntimeError("secret callback text")
    with pytest.raises(StreamError) as caught:
        read_completion_stream(response(first, final), callback)
    assert caught.value.usage == USAGE and "secret callback text" not in str(caught.value)
