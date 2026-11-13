from concurrent.futures import ThreadPoolExecutor
import json
import math
import time

import pytest

from constraint_memory.polza import ProviderError

from .lab_server import Runs, event_batch
from .lab_trace import LabPolzaClient, LabStopped, LabTrace, atomic_json, read_json


def wait_for_state(path, status, operation):
    until = time.monotonic() + 4
    while time.monotonic() < until:
        state = read_json(path, {})
        if state.get("status") == status and state.get("operation") == operation:
            return
        time.sleep(0.03)
    pytest.fail(f"Worker did not reach {status} at {operation}")


def test_pause_step_resume_and_stop_at_boundaries(tmp_path):
    control = tmp_path / "control.json"
    trace = LabTrace(tmp_path, control)
    actions = []
    atomic_json(control, {"mode": "pause", "revision": 0})

    def execute():
        for action in ("one", "two", "three"):
            trace.before(action)
            actions.append(action)
            if action == "two":
                while read_json(control)["mode"] == "run":
                    time.sleep(0.01)

    with ThreadPoolExecutor() as pool:
        worker = pool.submit(execute)
        try:
            wait_for_state(trace.state_path, "paused", "one")
            assert actions == []
            atomic_json(control, {"mode": "step", "revision": 1})
            wait_for_state(trace.state_path, "paused", "two")
            assert actions == ["one"]
            atomic_json(control, {"mode": "run", "revision": 2})
            wait_for_state(trace.state_path, "running", "two")
            atomic_json(control, {"mode": "pause", "revision": 3})
            wait_for_state(trace.state_path, "paused", "three")
            assert actions == ["one", "two"]
        finally:
            atomic_json(control, {"mode": "stop", "revision": 4})
        with pytest.raises(LabStopped):
            worker.result(timeout=4)


def test_trace_correlates_actual_model_payload_response_and_usage(tmp_path):
    trace = LabTrace(tmp_path)
    calls = []

    def transport(method, path, payload):
        calls.append(payload)
        return {"id": "call1", "model": "qwen/qwen3.6-27b", "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2}}

    with LabPolzaClient(trace=trace, api_key="pza_demo_secret", transport=transport) as client:
        client.complete([{"role": "user", "content": "visible prompt"}])
    raw = trace.timeline.read_text(encoding="utf-8")
    rows = list(map(json.loads, raw.splitlines()))
    assert [row["kind"] for row in rows] == ["prompt", "response", "usage"]
    assert rows[0]["data"]["payload"] == calls[0]
    assert rows[1]["data"]["request"] == rows[2]["data"]["request"] == rows[0]["seq"]
    assert "pza_demo_secret" not in raw


def test_interrupted_stream_retry_keeps_failed_attempt_and_separate_request(tmp_path):
    trace = LabTrace(tmp_path)
    attempts = []

    def transport(method, path, payload):
        attempts.append(payload)
        if len(attempts) == 1:
            raise ProviderError("Polza completion stream failed (truncated)", request_sent=True, request_id="lost")
        return {"id": "new", "model": "qwen/qwen3.6-27b", "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2}}

    with LabPolzaClient(trace=trace, api_key="pza_demo_secret", transport=transport,
                        interrupted_stream_retries=1, max_total_tokens=math.inf, max_cost_rub=None) as client:
        response = client.complete([{"role": "user", "content": "visible prompt"}])
        assert response["id"] == "new"
        assert [record["request_status"] for record in client.records] == ["failed", "completed"]
        assert client.records[0]["billing_status"] == "unknown_after_failed_request"
    assert attempts[0] == attempts[1]
    rows = list(map(json.loads, trace.timeline.read_text(encoding="utf-8").splitlines()))
    assert len([row for row in rows if row["kind"] == "prompt"]) == 2
    assert any(row["title"] == "Retrying interrupted model request" for row in rows)


def test_breakpoint_skips_current_match_and_pauses_before_next_match(tmp_path):
    control = tmp_path / "control.json"
    trace = LabTrace(tmp_path, control)
    actions = []
    atomic_json(control, {"mode": "pause", "revision": 0})

    def execute():
        for operation, category in (("first", "model"), ("middle", "memory"), ("second", "model")):
            trace.before(operation, category=category)
            actions.append(operation)

    with ThreadPoolExecutor() as pool:
        worker = pool.submit(execute)
        try:
            wait_for_state(trace.state_path, "paused", "first")
            skip = read_json(trace.state_path)["checkpoint"]
            atomic_json(control, {"mode": "run_to", "revision": 1, "target": "model", "skip": skip})
            wait_for_state(trace.state_path, "paused", "second")
            assert actions == ["first", "middle"]
        finally:
            atomic_json(control, {"mode": "stop", "revision": 2})
        with pytest.raises(LabStopped):
            worker.result(timeout=4)


def test_partial_timeline_line_is_read_after_flush(tmp_path):
    trace = LabTrace(tmp_path)
    trace.emit("run", "Ready")
    first = event_batch(trace.timeline, 0)
    with trace.timeline.open("ab") as stream:
        stream.write(b'{"seq":2,"at":"now","kind":"run","title":"Done","data":{}}')
    assert event_batch(trace.timeline, first["next"]) == {"events": [], "next": first["next"]}
    with trace.timeline.open("ab") as stream:
        stream.write(b"\n")
    assert event_batch(trace.timeline, first["next"])["events"][0]["title"] == "Done"


def test_previous_run_files_are_read_only_and_confined_to_run_directory(tmp_path):
    runs = Runs(tmp_path)
    name = "concept-subspaces-previous"
    folder = tmp_path / name
    folder.mkdir()
    (folder / "manifest.json").write_text('{"status":"running"}')
    (tmp_path / "external.json").write_text("{}")
    assert runs.describe(name)["managed"] is False
    with pytest.raises(ValueError, match="read-only"):
        runs.command(name, "stop")
    with pytest.raises(FileNotFoundError):
        runs.file(name, "../external.json")
