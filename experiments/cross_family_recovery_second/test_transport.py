"""Offline transport fixtures: no sockets, credentials, models or provider calls."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.parse

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from constraint_memory.polza import ProviderError
from constraint_memory.streaming import StreamError, read_completion_stream
from experiments.cross_family_recovery_second import transport


def event(value):
    return ("data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n").encode()


def chunk(delta=None, *, identifier="gen_fixture", finished=None, usage=None):
    result = {"id": identifier, "model": "qwen/qwen3.6-27b", "provider": "Chutes", "created": 123,
              "object": "chat.completion.chunk", "choices": []}
    if delta is not None or finished is not None:
        result["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finished}]
    if usage is not None:
        result["usage"] = usage
    return result


USAGE = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "cost_rub": .25}


def tool_stream():
    return b": processing\n\n" + b"".join([
        event(chunk({"role": "assistant", "reasoning_content": "think"})),
        event(chunk({"tool_calls": [{"index": 0, "id": "call_fixture", "type": "function",
                                     "function": {"name": "lookup", "arguments": '{"city":'}}]})),
        event(chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Oxford"}'}}]}, finished="tool_calls")),
        event(chunk(usage=USAGE)), event("[DONE]"),
    ])


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_tool_success_is_exact_native_output_and_exact_tape(tmp_path):
    wire = tool_stream()
    expected = read_completion_stream(io.BytesIO(wire))
    actual = transport.read_recorded_stream(io.BytesIO(wire), tmp_path)
    assert actual == expected
    assert (tmp_path / "response.sse").read_bytes() == wire
    capture = read_json(tmp_path / "capture.json")
    assert capture["full_http_body_captured"] is True
    assert capture["sha256"] == hashlib.sha256(wire).hexdigest()


@pytest.mark.parametrize("variant", ["different_error_id", "event_error", "finish_error", "malformed_error"])
def test_terminal_error_is_saved_and_rejected_before_native_metadata(tmp_path, variant):
    prefix = event(chunk({"reasoning_content": "Partial reasoning must never become an action"}))
    error = {"id": "error_fixture", "error": {"code": "SERVICE_UNAVAILABLE", "message": "Provider failed"},
             "trace_id": "trace_fixture", "choices": [{"index": 0, "delta": {"content": "unsafe"}, "finish_reason": "error"}]}
    if variant == "event_error":
        body = b"event: error\n" + event(error)
    elif variant == "finish_error":
        error.pop("error")
        body = event(error)
    elif variant == "malformed_error":
        body = b"event: error\ndata: {broken\n\n"
    else:
        body = event(error)
    wire = prefix + body + event("[DONE]")
    with pytest.raises(StreamError) as caught:
        transport.read_recorded_stream(io.BytesIO(wire), tmp_path)
    assert caught.value.code == "provider_error"
    assert caught.value.request_id == "gen_fixture"
    assert caught.value.usage is None
    assert not hasattr(caught.value, "content")
    detail = read_json(tmp_path / "provider_error.json")
    assert detail["partial_output_accepted"] is False
    assert detail["initial_request_id"] == "gen_fixture"
    if variant != "malformed_error":
        assert detail["error_id"] == "error_fixture"
    capture = read_json(tmp_path / "capture.json")
    assert capture["full_http_body_captured"] is False
    assert (tmp_path / "response.sse").read_bytes() == prefix + body


def test_unverified_error_usage_does_not_become_bill(tmp_path):
    error = {"id": "error_fixture", "error": {"message": "failed"}, "choices": [], "usage": USAGE}
    with pytest.raises(StreamError) as caught:
        transport.read_recorded_stream(io.BytesIO(event(error)), tmp_path)
    assert caught.value.usage is None
    detail = read_json(tmp_path / "provider_error.json")
    assert detail["received_error_usage"] == USAGE
    assert detail["received_error_usage_verified"] is False


def test_successful_mixed_id_still_fails_strict_parser(tmp_path):
    wire = event(chunk({"content": "one"})) + event(chunk({"content": "two"}, identifier="gen_other"))
    with pytest.raises(StreamError, match="id metadata changed") as caught:
        transport.read_recorded_stream(io.BytesIO(wire), tmp_path)
    assert caught.value.code == "invalid_stream"
    assert not (tmp_path / "provider_error.json").exists()


@pytest.mark.parametrize("wire,code", [
    (event(chunk({"content": "unfinished"})), "truncated"),
    (event(chunk({"content": "done"}, finished="stop")) + event("[DONE]"), "missing_usage"),
    (event(chunk({"content": "done"}, finished="stop")) + event(chunk(usage=USAGE)), "truncated"),
])
def test_incomplete_stream_never_becomes_completion_and_preserves_tape(tmp_path, wire, code):
    with pytest.raises(StreamError) as caught:
        transport.read_recorded_stream(io.BytesIO(wire), tmp_path)
    assert caught.value.code == code
    assert (tmp_path / "response.sse").read_bytes() == wire
    assert read_json(tmp_path / "capture.json")["status"] == "failed"


def test_capture_bound_is_exact_and_reports_omitted_overflow_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "MAX_TOTAL_BYTES", 32)
    wire = b":" + b"x" * 80 + b"\n\n"
    with pytest.raises(StreamError) as caught:
        transport.read_recorded_stream(io.BytesIO(wire), tmp_path)
    assert caught.value.code == "size_limit"
    assert (tmp_path / "response.sse").read_bytes() == wire[:32]
    capture = read_json(tmp_path / "capture.json")
    assert capture["bytes_captured"] == 32 and capture["bytes_received"] == 33
    assert capture["capture_limit_exceeded"] is True
    assert capture["full_http_body_captured"] is False


def test_native_line_bound_is_retained(tmp_path):
    wire = b":" + b"x" * (transport.MAX_LINE_BYTES + 4) + b"\n\n"
    with pytest.raises(StreamError) as caught:
        transport.read_recorded_stream(io.BytesIO(wire), tmp_path)
    assert caught.value.code == "size_limit"
    assert len((tmp_path / "response.sse").read_bytes()) == transport.MAX_LINE_BYTES + 1


def test_tape_cannot_be_overwritten(tmp_path):
    transport.read_recorded_stream(io.BytesIO(tool_stream()), tmp_path)
    with pytest.raises(FileExistsError):
        transport.read_recorded_stream(io.BytesIO(tool_stream()), tmp_path)


class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"Content-Type": "text/event-stream; charset=utf-8"}

    def getheader(self, name):
        return self.headers.get(name)


class FakeConnection:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.calls = []
        self.sock = SimpleNamespace(settimeout=lambda value: None)
        self.timeout = None

    def request(self, method, target, body=None, headers=None):
        self.calls.append((method, target, body, headers))
        if self.error is not None:
            raise self.error

    def getresponse(self):
        return self.response


def fake_client(connection):
    client = SimpleNamespace(base_url="https://polza.ai/api/v1", _closed=False,
        api_key="pza_invented_test_credential", timeout=1800, group_id="E2:69", traveler="Olivia", operation="agent",
        current_attempt="request-00156", _origin=urllib.parse.urlsplit("https://polza.ai/api/v1"),
        discards=0, warmups=0)
    client._connect = lambda: connection
    def warmup(value):
        assert value is connection
        client.warmups += 1
    def discard():
        client.discards += 1
    client._warm_catalog_connection, client._discard_connection = warmup, discard
    return client


def logical_payload():
    return {"model": "qwen/qwen3.6-27b", "messages": [{"role": "user", "content": "Fixture"}],
            "temperature": 0, "seed": 20261013, "max_tokens": 16384,
            "provider": {"only": ["chutes/fp8"], "allow_fallbacks": False},
            "reasoning": {"enabled": True}, "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto"}


def test_wire_adds_only_stream_and_never_logs_authentication_headers(tmp_path):
    response = FakeResponse(tool_stream(), headers={"Content-Type": "text/event-stream", "Set-Cookie": "bad-secret",
                                                    "Authorization": "bad-secret", "X-Trace-ID": "trace_fixture"})
    connection = FakeConnection(response)
    client = fake_client(connection)
    original = logical_payload()
    untouched = copy.deepcopy(original)
    actual = transport.RecordingChatTransport(client, tmp_path)("POST", "/chat/completions", original)
    assert actual == read_completion_stream(io.BytesIO(tool_stream()))
    assert original == untouched
    assert len(connection.calls) == 1 and client.warmups == 1 and client.discards == 0
    method, target, body, headers = connection.calls[0]
    assert method == "POST" and target == "/api/v1/chat/completions"
    assert json.loads(body) == {**original, "stream": True}
    assert headers["Authorization"].endswith(client.api_key)
    directory = tmp_path / "new_requests/attempt00001"
    assert (directory / "wire_request_body.json").read_bytes() == body
    saved = "\n".join(path.read_text(encoding="utf-8") for path in directory.iterdir() if path.suffix in {".json", ".jsonl"})
    assert client.api_key not in saved and "bad-secret" not in saved
    assert "Set-Cookie" not in saved and "Authorization" not in saved


def test_remote_disconnect_is_one_ambiguous_dispatch_without_retry(tmp_path):
    import http.client
    connection = FakeConnection(error=http.client.RemoteDisconnected("Invented sensitive details"))
    client = fake_client(connection)
    with pytest.raises(ProviderError) as caught:
        transport.RecordingChatTransport(client, tmp_path)("POST", "/chat/completions", logical_payload())
    assert caught.value.request_sent is True
    assert caught.value.request_id is None and caught.value.usage is None
    assert len(connection.calls) == 1 and client.discards == 1
    assert "sensitive" not in str(caught.value)
    receipt = read_json(tmp_path / "new_requests/attempt00001/dispatch.json")
    assert receipt["request_sent"] is True and receipt["automatic_paid_retries"] == 0


def test_preconnect_error_is_unsent_and_does_not_dispatch(tmp_path):
    connection = FakeConnection(FakeResponse(tool_stream()))
    client = fake_client(connection)
    def fail():
        raise OSError("Invented connection failure")
    client._connect = fail
    with pytest.raises(ProviderError) as caught:
        transport.RecordingChatTransport(client, tmp_path)("POST", "/chat/completions", logical_payload())
    assert caught.value.request_sent is False
    assert connection.calls == []


def test_documented_provider_error_preserves_generation_id_through_transport(tmp_path):
    wire = event(chunk({"content": "partial"})) + event({"id": "error_fixture", "error": {"code": "TIMEOUT", "message": "late"},
                                                             "trace_id": "trace_fixture", "choices": []})
    connection = FakeConnection(FakeResponse(wire))
    client = fake_client(connection)
    with pytest.raises(ProviderError) as caught:
        transport.RecordingChatTransport(client, tmp_path)("POST", "/chat/completions", logical_payload())
    assert caught.value.request_sent is True
    assert caught.value.request_id == "gen_fixture" and caught.value.usage is None
    assert caught.value.stream_error_code == "provider_error"
    assert caught.value.provider_error["error_id"] == "error_fixture"
    assert len(connection.calls) == 1 and client.discards == 1


@pytest.mark.parametrize("status,content_type", [(502, "application/json"), (200, "application/json")])
def test_http_or_content_type_failure_is_not_accepted_or_retried(tmp_path, status, content_type):
    connection = FakeConnection(FakeResponse(b'{"error":"fixture"}', status=status, headers={"Content-Type": content_type}))
    client = fake_client(connection)
    with pytest.raises(ProviderError) as caught:
        transport.RecordingChatTransport(client, tmp_path)("POST", "/chat/completions", logical_payload())
    assert caught.value.request_sent is True
    assert len(connection.calls) == 1 and client.discards == 1


def test_progress_has_only_safe_native_metadata_and_is_throttled(tmp_path):
    progress = []
    transport.read_recorded_stream(io.BytesIO(tool_stream()), tmp_path, on_progress=progress.append)
    assert len(progress) == 1
    assert set(progress[0]) == {"request_id", "model", "provider", "chunks", "bytes_received", "content_chars", "reasoning_chars", "tool_argument_chars"}
    assert "think" not in json.dumps(progress)
