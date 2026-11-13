"""Recorded Chat SSE transport; successful streams retain the frozen parser.

Only ``stream=True`` is added to the logical Chat payload. Error events are
observed before the native parser's successful-generation identity check. They
remain failures, never partial completions or inferred zero-cost receipts.
"""
from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import time

from constraint_memory.polza import ProviderError
from constraint_memory.secrets import redact
from constraint_memory.streaming import (
    MAX_EVENT_BYTES, MAX_LINE_BYTES, MAX_TOTAL_BYTES, StreamError,
    _progress_identity, _strict_json, read_completion_stream,
)

FSYNC_BYTES = 64 * 1024
FSYNC_SECONDS = 1.0
PROGRESS_SECONDS = 10.0
SAFE_RESPONSE_HEADERS = ("Content-Type", "Content-Length", "Date", "X-Request-ID", "X-Trace-ID", "Retry-After")


def _json(path, value):
    """Atomically replace small diagnostic metadata, flushing its contents."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = (json.dumps(redact(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _safe_scalar(value):
    if not isinstance(value, str):
        return None
    return redact(value[:4096])


class _RecordedReader:
    def __init__(self, response, directory):
        self.response, self.directory = response, Path(directory)
        self.stream = (self.directory / "response.sse").open("xb")
        self.digest = hashlib.sha256()
        self.captured = self.received = self.since_sync = 0
        self.last_sync = time.monotonic()
        self.eof = False
        self.capture_limit_exceeded = False
        self.observer_error = None
        self.initial_request_id = None
        self.event_name = ""
        self.event_data = []
        self.event_bytes = 0
        self.first_line = True
        self.provider_error = None

    def _sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.since_sync = 0
        self.last_sync = time.monotonic()

    def _abort(self, message, code):
        self.observer_error = StreamError(message, request_id=self.initial_request_id, code=code)
        raise self.observer_error

    def _event(self):
        data = "\n".join(self.event_data)
        name = self.event_name
        self.event_name, self.event_data, self.event_bytes = "", [], 0
        if data == "[DONE]":
            return
        try:
            value = _strict_json(data) if data else None
        except (ValueError, RecursionError):
            value = None
        failed_finish = False
        if isinstance(value, dict):
            choices = value.get("choices")
            if isinstance(choices, list):
                failed_finish = any(isinstance(choice, dict) and choice.get("finish_reason") in
                                    ("error", "failed", "cancelled", "timeout") for choice in choices)
        error = value.get("error") if isinstance(value, dict) else None
        terminal_error = name == "error" or error is not None or failed_finish
        if terminal_error:
            detail = error if isinstance(error, dict) else {}
            self.provider_error = {
                "initial_request_id": self.initial_request_id,
                "error_id": _safe_scalar(value.get("id")) if isinstance(value, dict) else None,
                "trace_id": _safe_scalar(value.get("trace_id") or detail.get("trace_id")) if isinstance(value, dict) else None,
                "code": _safe_scalar(detail.get("code")),
                "message": _safe_scalar(detail.get("message")),
                "event_name": _safe_scalar(name),
                "error_shape_valid": isinstance(error, dict),
                "received_error_usage": value.get("usage") if isinstance(value, dict) else None,
                "received_error_usage_verified": False,
                "partial_output_accepted": False,
            }
            _json(self.directory / "provider_error.json", self.provider_error)
            self._abort("Polza stream returned a terminal provider error; partial output rejected", "provider_error")
        # Observe only the first ID. The frozen reader validates every success
        # frame, including all later identity changes, types and usage totals.
        if self.initial_request_id is None and isinstance(value, dict):
            identifier = value.get("id")
            if isinstance(identifier, str) and identifier and len(identifier) <= 1024:
                self.initial_request_id = identifier

    def _observe(self, raw):
        try:
            line = raw.decode("utf-8-sig" if self.first_line else "utf-8").rstrip("\r\n")
        except UnicodeError:
            return  # The frozen parser emits the precise UTF-8 failure.
        self.first_line = False
        if not line:
            if self.event_data or self.event_name:
                self._event()
            return
        if line.startswith(":"):
            return
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self.event_bytes += len(value.encode("utf-8")) + 1
            if self.event_bytes > MAX_EVENT_BYTES:
                self._abort("Polza stream exceeded its event limit", "size_limit")
            self.event_data.append(value)
        elif field == "event":
            self.event_name = value

    def readline(self, size):
        if type(size) is not int or size < 1:
            self._abort("Recorded stream requires bounded reads", "size_limit")
        remaining = MAX_TOTAL_BYTES - self.captured
        raw = self.response.readline(min(size, MAX_LINE_BYTES + 1, remaining + 1))
        if raw == b"":
            self.eof = True
            return raw
        if not isinstance(raw, bytes):
            self._abort("HTTP stream returned non-byte data", "invalid_stream")
        self.received += len(raw)
        kept = raw[:remaining]
        self.stream.write(kept)
        self.digest.update(kept)
        self.captured += len(kept)
        self.since_sync += len(kept)
        if self.since_sync >= FSYNC_BYTES or time.monotonic() - self.last_sync >= FSYNC_SECONDS:
            self._sync()
        if len(raw) > remaining:
            self.capture_limit_exceeded = True
            self._abort("Polza stream exceeded its bounded raw capture", "size_limit")
        if len(raw) <= MAX_LINE_BYTES:
            self._observe(raw)
        return raw

    def close(self, status, error=None):
        try:
            self._sync()
        finally:
            self.stream.close()
        _json(self.directory / "capture.json", {
            "status": status, "bytes_captured": self.captured, "bytes_received": self.received,
            "sha256": self.digest.hexdigest(), "limit_bytes": MAX_TOTAL_BYTES,
            "http_body_eof_observed": self.eof, "capture_limit_exceeded": self.capture_limit_exceeded,
            "full_http_body_captured": self.eof and not self.capture_limit_exceeded,
            "initial_request_id": _progress_identity(self.initial_request_id),
            "error_code": getattr(error, "code", None),
            "capture_note": "Exact received body prefix; a parser abort does not prove the remaining remote body was captured.",
        })


def read_recorded_stream(response, destination, on_progress=None):
    """Assemble using the unchanged strict reader and preserve a bounded tape."""
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    observer = _RecordedReader(response, directory)
    last_progress = None

    def progress(values):
        nonlocal last_progress
        now = time.monotonic()
        if last_progress is not None and now - last_progress < PROGRESS_SECONDS:
            return
        last_progress = now
        safe = {key: _progress_identity(value) if key in {"request_id", "model", "provider"} else value
                for key, value in values.items()}
        if on_progress is not None:
            on_progress(safe)

    error = None
    try:
        return read_completion_stream(observer, on_progress=progress)
    except StreamError as native_error:
        error = observer.observer_error or native_error
        if observer.observer_error is not None:
            # Only usage already accepted by the frozen reader may become a
            # billing receipt. Error-frame usage stays separate and unverified.
            error.usage = native_error.usage
        error.provider_error = observer.provider_error
        raise error from None
    except BaseException as unexpected:
        error = unexpected
        raise
    finally:
        observer.close("failed" if error is not None else "accepted_by_strict_parser", error)


class RecordingChatTransport:
    """One network POST per invocation, with no ambiguous paid retry loop."""
    def __init__(self, client, destination):
        self.client, self.destination = client, Path(destination)
        self.attempts = 0

    def __call__(self, method, path, payload):
        if method != "POST" or path != "/chat/completions" or not isinstance(payload, dict):
            raise ProviderError("Recorded transport requires the native Chat completion endpoint", request_sent=False)
        client = self.client
        if client.base_url != "https://polza.ai/api/v1" or client._closed:
            raise ProviderError("Recorded transport requires the open fixed Polza client", request_sent=False)
        if not client.api_key:
            raise ProviderError("POLZA_API_KEY is missing", request_sent=False)
        wire = copy.deepcopy(payload)
        if wire.get("stream") not in (None, False):
            raise ProviderError("Logical recovery payload unexpectedly enables streaming", request_sent=False)
        wire["stream"] = True
        body = json.dumps(wire, allow_nan=False).encode("utf-8")
        self.attempts += 1
        directory = self.destination / "new_requests" / f"attempt{self.attempts:05d}"
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "wire_request_body.json").open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        common = {"local_attempt": self.attempts, "logical_attempt_id": getattr(client, "current_attempt", None),
                  "scope": client.group_id, "traveler": client.traveler, "operation": client.operation,
                  "method": method, "path": path, "wire_body_sha256": hashlib.sha256(body).hexdigest(),
                  "wire_change": {"stream": True}, "automatic_paid_retries": 0}
        sent = False
        _json(directory / "dispatch.json", {**common, "phase": "prepared", "request_sent": False})
        try:
            connection = client._connect()
            client._warm_catalog_connection(connection)
            connection.timeout = client.timeout
            if connection.sock is not None:
                connection.sock.settimeout(client.timeout)
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
                       "User-Agent": "constraint-memory-lab/0.1", "Connection": "keep-alive",
                       "Authorization": "Bearer " + client.api_key}
            sent = True  # Request bytes may leave during the following call.
            _json(directory / "dispatch.json", {**common, "phase": "POST_MAY_HAVE_BEEN_SENT", "request_sent": True})
            connection.request("POST", client._origin.path.rstrip("/") + path, body=body, headers=headers)
            response = connection.getresponse()
            safe_headers = {name: redact(str(response.getheader(name))[:1024])
                            for name in SAFE_RESPONSE_HEADERS if response.getheader(name) is not None}
            _json(directory / "response_headers.json", {"status": response.status, "headers": safe_headers})
            if type(response.status) is not int or not 200 <= response.status < 300:
                raise ProviderError("Polza returned HTTP " + str(response.status), request_sent=True)
            content_type = str(response.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
            if content_type != "text/event-stream":
                raise ProviderError("Polza did not return the requested SSE content type", request_sent=True)
            last_event = None

            def progress(values):
                nonlocal last_event
                last_event = values
                with (directory / "progress.jsonl").open("ab") as stream:
                    stream.write((json.dumps(redact(values), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())

            raw = read_recorded_stream(response, directory, on_progress=progress)
            _json(directory / "dispatch.json", {**common, "phase": "strict_stream_complete", "request_sent": True,
                                               "request_id": _progress_identity(raw.get("id")), "last_progress": last_event})
            return raw
        except BaseException as error:
            client._discard_connection()
            _json(directory / "dispatch.json", {**common, "phase": "failed", "request_sent": sent,
                "error_type": type(error).__name__, "stream_error_code": getattr(error, "code", None),
                "request_id": _progress_identity(getattr(error, "request_id", None)),
                "usage": getattr(error, "usage", None), "provider_error": getattr(error, "provider_error", None)})
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(error, StreamError):
                failure = ProviderError("Polza recorded completion stream failed (" + error.code + "); no automatic paid retry",
                                        request_sent=sent, request_id=error.request_id, usage=error.usage)
                failure.stream_error_code = error.code
                failure.provider_error = getattr(error, "provider_error", None)
                failure.recording_directory = str(directory)
                raise failure from None
            if isinstance(error, ProviderError):
                error.request_sent = sent
                raise
            # Remote error text and exception reprs can echo credentials.
            raise ProviderError("Polza recorded request failed (" + type(error).__name__ + "); no automatic paid retry",
                                request_sent=sent) from None
