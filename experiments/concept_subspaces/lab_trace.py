"""Append-only experiment events and cooperative control at operation boundaries."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from constraint_memory.polza import PolzaClient, ProviderError
from constraint_memory.secrets import redact


class LabStopped(Exception):
    pass


def json_default(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, set):
        return sorted(value, key=str)
    return str(value)


def json_object(value):
    return redact(json.loads(json.dumps(value, ensure_ascii=False, default=json_default)))


def read_json(path, default=None):
    for attempt in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02)


def atomic_json(path, value):
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    for attempt in range(5):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02)


class LabTrace:
    def __init__(self, directory: Path, control: Path | None = None, state_path: Path | None = None):
        self.timeline = directory / "timeline.jsonl"
        self.control = control
        self.state_path = state_path or (control.with_suffix(".state.json") if control else None)
        self.sequence = 0
        self.last_step = None

    def emit(self, kind, title, **data):
        self.sequence += 1
        row = {"seq": self.sequence, "at": datetime.now(timezone.utc).isoformat(), "kind": kind,
               "title": title, "data": json_object(data)}
        with self.timeline.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.sequence

    def before(self, operation, *, category=None, **details):
        if self.control is None:
            return
        sequence = self.emit("checkpoint", operation, category=category, **details)
        waiting = None
        while True:
            command = read_json(self.control, {"mode": "pause", "revision": 0})
            mode = command["mode"]
            revision = command["revision"]
            if mode == "stop":
                self.emit("control", "Stopped at checkpoint", operation=operation, checkpoint=sequence)
                self._status("stopped", operation, sequence, revision)
                raise LabStopped(operation)
            passes_breakpoint = mode == "run_to" and (category != command["target"] or sequence == command.get("skip"))
            if mode == "run" or passes_breakpoint or (mode == "step" and revision != self.last_step):
                if mode == "step":
                    self.last_step = revision
                if waiting is not None:
                    self.emit("control", "Continued from checkpoint", operation=operation, checkpoint=sequence)
                self._status("executing_step" if mode == "step" else "running", operation, sequence, revision)
                return
            if waiting != revision:
                self._status("paused", operation, sequence, revision)
                self.emit("control", "Paused at checkpoint", operation=operation, checkpoint=sequence)
                waiting = revision
            time.sleep(0.15)

    def _status(self, status, operation, sequence, revision):
        atomic_json(self.state_path, {"status": status, "operation": operation,
                                      "checkpoint": sequence, "revision": revision})


class LabPolzaClient(PolzaClient):
    """Trace the exact completion payload and returned data without HTTP headers."""

    def __init__(self, *, trace, interrupted_stream_retries=0, **kwargs):
        super().__init__(**kwargs)
        self.trace = trace
        self.call_sequence = None
        self.interrupted_stream_retries = interrupted_stream_retries
        self._recovered_attempts = set()

    def _check_budget(self, messages, output_cap, tools=None):
        if self.max_total_tokens == math.inf and self.max_cost_rub is None and self._recovered_attempts:
            unknown = [record for record in self.records if record.get("group_id") == self.group_id
                       and record.get("total_tokens") is None]
            if unknown and all(id(record) in self._recovered_attempts for record in unknown):
                return  # No aggregate cap is configured. Preserve unknown usage in the ledger.
        super()._check_budget(messages, output_cap, tools)

    def complete(self, *args, **kwargs):
        self.call_sequence = None
        try:
            return super().complete(*args, **kwargs)
        except ProviderError as error:
            interrupted = any(f"stream failed ({code})" in str(error) for code in ("truncated", "read_failed"))
            if not (self.interrupted_stream_retries and interrupted and error.request_sent is True):
                raise
            self.interrupted_stream_retries -= 1
            self._recovered_attempts.add(id(self.records[-1]))
            self.trace.emit("error", "Retrying interrupted model request", request=self.call_sequence,
                            request_id=error.request_id, previous_billing_known=error.usage is not None,
                            remaining_retries=self.interrupted_stream_retries)
            self.call_sequence = None
            return super().complete(*args, **kwargs)

    def _request(self, method, path, payload=None, authenticated=True):
        if method != "POST" or path != "/chat/completions":
            return super()._request(method, path, payload, authenticated)
        scope = {"group": self.group_id, "turn": self.traveler, "operation": self.operation}
        self.call_sequence = self.trace.emit("prompt", "Polza completion request", scope=scope, payload=payload)
        self.trace.before("Send model request", category="model", request=self.call_sequence, scope=scope)
        try:
            response = super()._request(method, path, payload, authenticated)
        except Exception as error:
            self.trace.emit("error", "Polza request failed", request=self.call_sequence,
                            scope=scope, error=str(error))
            raise
        self.trace.emit("response", "Polza completion response", request=self.call_sequence,
                        scope=scope, response=response)
        return response

    def _record_usage(self, record):
        super()._record_usage(record)
        self.trace.emit("usage", "Provider usage", request=self.call_sequence, **record)

    def _transport_event(self, values):
        super()._transport_event(values)
        self.trace.emit("transport", "Polza transport", request=self.call_sequence, **values)
