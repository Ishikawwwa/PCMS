"""Pacing uses a fake clock and an invented transport, never the network."""
import copy
import http.client
import json
import ssl

import pytest

from .pacing import FreshConnectionTransport, PacedTransport
from constraint_memory.polza import ProviderError


def test_spaces_starts_without_changing_requests(tmp_path):
    now = [0.0]
    calls = []
    def sleep(seconds):
        now[0] += seconds
    def transport(method, path, payload):
        calls.append((now[0], method, path, copy.deepcopy(payload)))
        return {"ok": True}
    paced = PacedTransport(transport, tmp_path, clock=lambda: now[0], sleep=sleep)
    payload = {"seed": 5, "model": "invented", "messages": [{"role": "user", "content": "fixture"}]}
    saved = copy.deepcopy(payload)
    paced("POST", "/chat/completions", payload)
    now[0] += 2
    paced("POST", "/chat/completions", payload)
    now[0] += 20
    paced("POST", "/chat/completions", payload)
    assert [call[0] for call in calls] == [10, 20, 40]
    assert all(call[3] == saved for call in calls)
    assert payload == saved
    rows = [json.loads(line) for line in (tmp_path / "request_pacing.jsonl").read_text().splitlines()]
    assert [row["wait_seconds"] for row in rows] == [10, 8, 0]


@pytest.mark.parametrize("failure", [RuntimeError("TOO_MANY_REQUESTS"), ConnectionError("ambiguous")])
def test_never_retries_even_rate_errors(tmp_path, failure):
    calls = []
    def transport(*args):
        calls.append(args)
        raise failure
    paced = PacedTransport(transport, tmp_path, clock=lambda: 0, sleep=lambda _: None)
    with pytest.raises(type(failure), match=str(failure)):
        paced("POST", "/chat/completions", {"seed": 5})
    assert len(calls) == 1


class FakeClient:
    def __init__(self, failures=(), *, fail_on="connect"):
        self._closed = False
        self.base_url = "https://polza.ai/api/v1"
        self.api_key = "invented-test-credential"
        self.current_attempt = "invented-attempt"
        self._connection = "stale-connection"
        self._catalog_warmed = True
        self.connect_timeout = 10
        self.failures, self.fail_on = list(failures), fail_on
        self.connects, self.heads, self.discards = 0, 0, 0

    def _discard_connection(self):
        self.discards += 1
        self._connection, self._catalog_warmed = None, False

    def _connect(self):
        assert self._connection is None
        self.connects += 1
        if self.fail_on == "connect" and self.failures:
            raise self.failures.pop(0)
        self._connection = object()
        return self._connection

    def _warm_catalog_connection(self, connection):
        assert connection is self._connection
        assert self._catalog_warmed is False
        self.heads += 1
        if self.fail_on == "head" and self.failures:
            raise self.failures.pop(0)
        self._catalog_warmed = True


def test_every_post_gets_a_fresh_connection_after_any_idle_gap(tmp_path):
    client = FakeClient()
    seen, requests = [], []
    def post(method, path, payload):
        assert client._catalog_warmed
        seen.append(client._connection)
        requests.append((method, path, copy.deepcopy(payload)))
        return {"accepted": True}
    fresh = FreshConnectionTransport(post, client, tmp_path, sleep=lambda _: pytest.fail("Unexpected retry"))
    payload = {"model": "fixture", "seed": 5, "messages": [{"role": "user", "content": "data"}]}
    saved = copy.deepcopy(payload)
    fresh("POST", "/chat/completions", payload)
    # A later request cannot reuse an old connection even if a clock jumps or
    # the process was suspended between calls. No idle-time heuristic is used.
    client._connection, client._catalog_warmed = "stale-after-suspend", True
    fresh("POST", "/chat/completions", payload)
    assert client.connects == client.heads == len(requests) == 2
    assert seen[0] is not seen[1]
    assert all(request[2] == saved for request in requests) and payload == saved
    assert client._connection is None


@pytest.mark.parametrize("fail_on", ["connect", "head"])
def test_only_prepost_connect_or_public_head_failures_can_retry(tmp_path, fail_on):
    client = FakeClient([http.client.RemoteDisconnected("fixture")], fail_on=fail_on)
    posts, waits = [], []
    fresh = FreshConnectionTransport(lambda *args: posts.append(args), client, tmp_path, sleep=waits.append)
    fresh("POST", "/chat/completions", {"seed": 5})
    assert client.connects == 2 and len(posts) == 1 and waits == [1]
    rows = [json.loads(line) for line in (tmp_path / "connection_preflight.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["failed_before_model_post", "fresh_connection_ready"]
    assert all(row["model_post_sent"] is False for row in rows)


def test_ambiguous_post_disconnect_is_never_repeated(tmp_path):
    client, posts = FakeClient(), []
    failure = ProviderError("Invented ambiguous POST", request_sent=True)
    def post(*args):
        posts.append(args)
        raise failure
    fresh = FreshConnectionTransport(post, client, tmp_path, sleep=lambda _: pytest.fail("POST retry forbidden"))
    with pytest.raises(ProviderError) as caught:
        fresh("POST", "/chat/completions", {"seed": 5})
    assert caught.value is failure and caught.value.request_sent is True
    assert len(posts) == client.connects == client.heads == 1
    assert client._connection is None


def test_exhausted_preflight_proves_no_model_post_was_sent(tmp_path):
    client = FakeClient([ConnectionError("fixture")] * 3)
    waits = []
    fresh = FreshConnectionTransport(lambda *_: pytest.fail("POST forbidden"), client, tmp_path, sleep=waits.append)
    with pytest.raises(ProviderError) as caught:
        fresh("POST", "/chat/completions", {"seed": 5})
    assert caught.value.request_sent is False
    assert client.connects == 3 and client.heads == 0 and waits == [1, 2]


def test_tls_identity_failure_is_not_retried(tmp_path):
    client = FakeClient([ssl.SSLCertVerificationError(1, "fixture")])
    fresh = FreshConnectionTransport(lambda *_: pytest.fail("POST forbidden"), client, tmp_path,
                                     sleep=lambda _: pytest.fail("TLS failure retry forbidden"))
    with pytest.raises(ProviderError) as caught:
        fresh("POST", "/chat/completions", {})
    assert caught.value.request_sent is False and client.connects == 1


def test_progress_reports_pacing_start_end_and_actual_wait_without_changing_dispatch(tmp_path):
    now, events, calls = [0.0], [], []
    def sleep(seconds):
        now[0] += seconds
    def report(event, **fields):
        events.append((event, fields))
    paced = PacedTransport(lambda *args: calls.append((now[0], args)), tmp_path,
                           clock=lambda: now[0], sleep=sleep, progress=report)
    payload = {"seed": 20261013, "messages": [{"role": "user", "content": "offline fixture"}]}
    saved = copy.deepcopy(payload)
    paced("POST", "/chat/completions", payload)
    assert [name for name, _ in events] == ["pacing_wait", "pacing_done"]
    assert events[0][1]["wait_seconds"] == events[1][1]["wait_seconds"] == 10
    assert events[1][1]["elapsed_seconds"] == 10
    now[0] += 20
    paced("POST", "/chat/completions", payload)
    assert events[-1][0] == "pacing_done"
    assert events[-1][1]["wait_seconds"] == events[-1][1]["elapsed_seconds"] == 0
    assert [start for start, _ in calls] == [10, 30]
    assert payload == saved


def test_progress_identifies_connect_head_ready_and_single_post_start_before_dispatch(tmp_path):
    now, events, posts = [0.0], [], []
    client = FakeClient()
    native_connect, native_head = client._connect, client._warm_catalog_connection
    def connect():
        now[0] += 3
        return native_connect()
    def head(connection):
        now[0] += 2
        return native_head(connection)
    client._connect, client._warm_catalog_connection = connect, head
    def report(event, **fields):
        events.append((event, fields))
    def post(*args):
        assert events[-1][0] == "model_post_start"
        assert events[-1][1]["model_post_sent"] is False
        assert events[-1][1]["dispatch_starting"] is True
        posts.append(args)
        return "invented receipt"
    fresh = FreshConnectionTransport(post, client, tmp_path, progress=report, clock=lambda: now[0])
    assert fresh("POST", "/chat/completions", {"seed": 5}) == "invented receipt"
    assert [name for name, _ in events] == ["preflight_connect", "preflight_head", "preflight_ready", "model_post_start"]
    assert events[0][1]["attempt"] == events[1][1]["attempt"] == 1
    assert events[0][1]["max_attempts"] == events[1][1]["max_attempts"] == 3
    assert events[0][1]["timeout_seconds"] == events[1][1]["timeout_seconds"] == 10
    assert events[2][1]["elapsed_seconds"] == 5
    assert events[3][1]["preflight_attempt"] == 1
    assert client.connects == client.heads == len(posts) == 1


@pytest.mark.parametrize("fail_on,failed_stage", [("connect", "connect"), ("head", "head")])
def test_progress_explains_safe_retry_stage_and_delay_but_never_repeats_post(tmp_path, fail_on, failed_stage):
    client = FakeClient([http.client.RemoteDisconnected("private exception details")], fail_on=fail_on)
    events, waits, posts = [], [], []
    def report(event, **fields):
        events.append((event, fields))
    fresh = FreshConnectionTransport(lambda *args: posts.append(args), client, tmp_path,
                                     sleep=waits.append, clock=lambda: 0.0, progress=report)
    fresh("POST", "/chat/completions", {})
    failed = next(fields for name, fields in events if name == "preflight_failed")
    retry = next(fields for name, fields in events if name == "preflight_retry_wait")
    assert failed["stage"] == failed_stage and failed["attempt"] == 1
    assert failed["error_type"] == "RemoteDisconnected"
    assert failed["model_post_sent"] is False
    assert retry["wait_seconds"] == 1 and retry["next_attempt"] == 2
    assert events[-1][0] == "model_post_start" and events[-1][1]["preflight_attempt"] == 2
    assert client.connects == 2 and len(posts) == 1 and waits == [1]
    assert "private exception details" not in json.dumps(events)


def test_progress_exhaustion_proves_no_post_and_never_exposes_error_credentials_or_urls(tmp_path):
    sensitive = "pza_FAKE_TEST_ONLY https://private-proxy.invalid/path user:password"
    client = FakeClient([TimeoutError(sensitive)] * 3)
    events, waits, posts = [], [], []
    def report(event, **fields):
        events.append((event, fields))
    fresh = FreshConnectionTransport(lambda *args: posts.append(args), client, tmp_path,
                                     sleep=waits.append, clock=lambda: 0.0, progress=report)
    with pytest.raises(ProviderError) as failure:
        fresh("POST", "/chat/completions", {})
    assert failure.value.request_sent is False
    assert [fields["attempt"] for event, fields in events if event == "preflight_connect"] == [1, 2, 3]
    assert [fields["wait_seconds"] for event, fields in events if event == "preflight_retry_wait"] == [1, 2]
    assert events[-1][0] == "preflight_exhausted"
    assert events[-1][1]["attempt"] == 3 and events[-1][1]["reason"] == "attempts_exhausted"
    assert events[-1][1]["model_post_sent"] is False
    assert not posts and waits == [1, 2]
    serialized = json.dumps(events) + str(failure.value)
    assert sensitive not in serialized
    assert "private-proxy" not in serialized and "password" not in serialized
    assert "pza_FAKE_TEST_ONLY" not in serialized
    assert not any(event == "model_post_start" for event, _ in events)


def test_progress_tls_failure_reports_immediate_exhaustion_without_safe_or_paid_retry(tmp_path):
    client = FakeClient([ssl.SSLCertVerificationError(1, "private TLS details")])
    events = []
    fresh = FreshConnectionTransport(lambda *_: pytest.fail("No POST is permitted"), client, tmp_path,
                                     sleep=lambda _: pytest.fail("TLS identity failure cannot retry"),
                                     progress=lambda event, **fields: events.append((event, fields)))
    with pytest.raises(ProviderError) as failure:
        fresh("POST", "/chat/completions", {})
    assert failure.value.request_sent is False and client.connects == 1
    assert [event for event, _ in events] == ["preflight_connect", "preflight_failed", "preflight_exhausted"]
    assert events[-1][1]["reason"] == "tls_identity_failure"


def test_broken_progress_observer_cannot_retry_or_mask_an_ambiguous_post(tmp_path):
    client, posts = FakeClient(), []
    original = ProviderError("Invented ambiguous POST", request_sent=True)
    def report(*args, **kwargs):
        raise ValueError("Invented unavailable console")
    def post(*args):
        posts.append(args)
        raise original
    fresh = FreshConnectionTransport(post, client, tmp_path, progress=report,
                                     sleep=lambda _: pytest.fail("No retry is allowed"))
    with pytest.raises(ProviderError) as failure:
        fresh("POST", "/chat/completions", {"seed": 5})
    assert failure.value is original and failure.value.request_sent is True
    assert client.connects == client.heads == len(posts) == 1


def test_broken_progress_observer_does_not_change_safe_preflight_retries_or_pacing(tmp_path):
    client = FakeClient([ConnectionError("fixture")])
    now, waits, posts = [0.0], [], []
    def report(*args, **kwargs):
        raise ValueError("Invented unavailable console")
    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds
    fresh = FreshConnectionTransport(lambda *args: posts.append(args), client, tmp_path,
                                     sleep=sleep, clock=lambda: now[0], progress=report)
    paced = PacedTransport(fresh, tmp_path, sleep=sleep, clock=lambda: now[0], progress=report)
    paced("POST", "/chat/completions", {})
    assert waits == [10, 1]
    assert client.connects == 2 and len(posts) == 1


def test_invalid_timeout_metadata_is_omitted_from_progress_without_exposing_a_url(tmp_path):
    client = FakeClient()
    client.connect_timeout = "https://private-proxy.invalid/credential"
    events = []
    fresh = FreshConnectionTransport(lambda *_: None, client, tmp_path,
                                     progress=lambda event, **fields: events.append((event, fields)))
    fresh("POST", "/chat/completions", {})
    assert all(fields["timeout_seconds"] is None for event, fields in events if event in ("preflight_connect", "preflight_head"))
    assert "private-proxy" not in json.dumps(events)
