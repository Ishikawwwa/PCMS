"""Space new network requests without changing inference or retrying them."""
from __future__ import annotations

import datetime as dt
import http.client
import math
from pathlib import Path
import ssl
import time

from experiments.cross_family_recovery.run import frozen
from constraint_memory.polza import ProviderError

MINIMUM_START_INTERVAL_SECONDS = 10.0


def _progress(callback, event, **fields):
    """Reporting cannot change the request or its retry policy."""
    if callback is not None:
        try:
            callback(event, **fields)
        except Exception:
            # A broken console or optional observer must not turn a completed
            # POST into another transport attempt or hide its original error.
            pass


def _timeout(client):
    value = getattr(client, "connect_timeout", 10.0)
    return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


class PacedTransport:
    def __init__(self, transport, destination, *, clock=time.monotonic, sleep=time.sleep, progress=None):
        self.transport = transport
        self.destination = Path(destination)
        self.clock, self.sleep = clock, sleep
        self.progress = progress
        self.last_start = None

    def __call__(self, method, path, payload):
        now = self.clock()
        wait = (MINIMUM_START_INTERVAL_SECONDS if self.last_start is None else
                max(0.0, MINIMUM_START_INTERVAL_SECONDS - (now - self.last_start)))
        waiting_started = self.clock()
        if wait:
            _progress(self.progress, "pacing_wait", wait_seconds=wait,
                      minimum_start_interval_seconds=MINIMUM_START_INTERVAL_SECONDS)
            if self.progress is None:
                print(f"REQUEST PACING: waiting {wait:.1f}s before a new request", flush=True)
            self.sleep(wait)
        _progress(self.progress, "pacing_done", wait_seconds=wait,
                  elapsed_seconds=max(0.0, self.clock() - waiting_started),
                  minimum_start_interval_seconds=MINIMUM_START_INTERVAL_SECONDS)
        self.last_start = self.clock()
        frozen.durable(self.destination / "request_pacing.jsonl", {
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "wait_seconds": wait,
            "minimum_start_interval_seconds": MINIMUM_START_INTERVAL_SECONDS,
            "policy": "One request only; this wrapper never retries a rejected or ambiguous POST",
        }, append=True)
        return self.transport(method, path, payload)


class FreshConnectionTransport:
    """Prepare a new connection; retry only CONNECT/public HEAD, never POST."""
    def __init__(self, transport, client, destination, *, sleep=time.sleep, clock=time.monotonic, progress=None):
        self.transport, self.client = transport, client
        self.destination, self.sleep = Path(destination), sleep
        self.clock, self.progress = clock, progress

    def __call__(self, method, path, payload):
        if method != "POST" or path != "/chat/completions" or not isinstance(payload, dict):
            raise ProviderError("Fresh connection requires the fixed completion endpoint", request_sent=False)
        if self.client._closed or self.client.base_url != "https://polza.ai/api/v1" or not self.client.api_key:
            raise ProviderError("Fresh connection requires the open configured Polza client", request_sent=False)
        for attempt in range(1, 4):
            self.client._discard_connection()
            started = self.clock()
            stage = "connect"
            _progress(self.progress, "preflight_connect", attempt=attempt, max_attempts=3,
                      stage=stage, timeout_seconds=_timeout(self.client))
            try:
                connection = self.client._connect()
                stage = "head"
                _progress(self.progress, "preflight_head", attempt=attempt, max_attempts=3,
                          stage=stage, timeout_seconds=_timeout(self.client))
                self.client._warm_catalog_connection(connection)
            except (OSError, http.client.HTTPException, ProviderError) as error:
                self.client._discard_connection()
                elapsed = max(0.0, self.clock() - started)
                frozen.durable(self.destination / "connection_preflight.jsonl", {
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "logical_attempt_id": self.client.current_attempt, "safe_preflight_attempt": attempt,
                    "phase": "failed_before_model_post", "error_type": type(error).__name__,
                    "model_post_sent": False, "policy": "Only CONNECT and unauthenticated public HEAD may be repeated",
                }, append=True)
                _progress(self.progress, "preflight_failed", attempt=attempt, max_attempts=3,
                          stage=stage, elapsed_seconds=elapsed, error_type=type(error).__name__,
                          model_post_sent=False)
                if attempt == 3 or isinstance(error, ssl.SSLCertVerificationError):
                    _progress(self.progress, "preflight_exhausted", attempt=attempt, max_attempts=3,
                              stage=stage, elapsed_seconds=elapsed, error_type=type(error).__name__,
                              model_post_sent=False,
                              reason="tls_identity_failure" if isinstance(error, ssl.SSLCertVerificationError) else "attempts_exhausted")
                    raise ProviderError("Connection preflight failed before any model POST (" + type(error).__name__ + ")",
                                        request_sent=False) from None
                delay = float(2 ** (attempt - 1))
                _progress(self.progress, "preflight_retry_wait", attempt=attempt, max_attempts=3,
                          next_attempt=attempt + 1, wait_seconds=delay, model_post_sent=False)
                self.sleep(delay)
            else:
                frozen.durable(self.destination / "connection_preflight.jsonl", {
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "logical_attempt_id": self.client.current_attempt, "safe_preflight_attempt": attempt,
                    "phase": "fresh_connection_ready", "model_post_sent": False,
                }, append=True)
                _progress(self.progress, "preflight_ready", attempt=attempt, max_attempts=3,
                          stage="ready", elapsed_seconds=max(0.0, self.clock() - started), model_post_sent=False)
                break
        # This call is deliberately outside the preflight exception/retry loop.
        # Once POST bytes might leave, a disconnect is an ambiguous paid outcome.
        try:
            _progress(self.progress, "model_post_start", preflight_attempt=attempt,
                      model_post_sent=False, dispatch_starting=True)
            return self.transport(method, path, payload)
        finally:
            self.client._discard_connection()
