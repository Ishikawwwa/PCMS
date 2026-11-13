"""Network-free lifetime, TLS, origin, and retry tests for the Polza transport."""
from __future__ import annotations

import json
import ssl

import pytest

import constraint_memory.polza as module
from constraint_memory.polza import PolzaClient, ProviderError


class Response:
    def __init__(self, status=200, payload=None, failure=None):
        self.status = status
        self.payload = payload if payload is not None else {"ok": True}
        self.failure = failure

    def read(self):
        if self.failure:
            raise self.failure
        return json.dumps(self.payload).encode()


class Connection:
    def __init__(self, responses=None, connect_failure=None):
        self.responses = list(responses or [])
        self.connect_failure = connect_failure
        self.sock = None
        self.requests = []
        self.closed = False
        self.socket_timeouts = []

    def connect(self):
        if self.connect_failure:
            raise self.connect_failure
        self.sock = self

    def settimeout(self, timeout):
        self.socket_timeouts.append(timeout)

    def set_tunnel(self, host, port, headers=None):
        self.tunnel = (host, port, headers or {})

    def request(self, method, target, body=None, headers=None):
        self.requests.append({"method": method, "target": target, "body": body, "headers": headers or {},
                              "socket_timeout":self.socket_timeouts[-1] if self.socket_timeouts else None})

    def getresponse(self):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self):
        self.closed = True
        self.sock = None


@pytest.fixture
def stub_network(monkeypatch):
    connections = []
    factory_calls = []
    monkeypatch.setattr(module.urllib.request, "getproxies", lambda: {})
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)

    def factory(host, port, **kwargs):
        factory_calls.append((host, port, kwargs))
        return connections[len(factory_calls) - 1]

    monkeypatch.setattr(module.http.client, "HTTPSConnection", factory)
    return connections, factory_calls


def test_verified_tls_catalog_warmup_and_completion_share_connection(stub_network):
    connections, calls = stub_network
    connection = Connection([Response(), Response(payload={"data": []}), Response(payload={"choices": []})])
    connections.append(connection)
    client = PolzaClient(api_key="offline-fixture-key", retries=0)
    assert client.models() == []
    client._request("POST", "/chat/completions", {"model": "qwen/qwen3.6-27b"})
    assert len(calls) == 1
    host, port, parameters = calls[0]
    assert host == "polza.ai" and port == 443
    assert parameters["context"].check_hostname
    assert parameters["context"].verify_mode == ssl.CERT_REQUIRED
    assert [(r["method"], r["target"]) for r in connection.requests] == [
        ("HEAD", "/"), ("GET", "/api/v1/models?type=chat"), ("POST", "/api/v1/chat/completions")]
    assert "Authorization" not in connection.requests[0]["headers"]
    assert "Authorization" not in connection.requests[1]["headers"]
    assert connection.requests[2]["headers"]["Authorization"] == "Bearer offline-fixture-key"
    assert not client.records
    assert client.transport_diagnostics[0]["billable_model_call"] is False
    client.close()
    assert connection.closed
    with pytest.raises(ProviderError, match="closed"):
        client.models()


def test_safe_get_timeout_warms_each_new_verified_connection(stub_network):
    connections, calls = stub_network
    first = Connection([Response(), TimeoutError("offline timeout")])
    second = Connection([Response(),Response(payload={"data": []})])
    connections.extend([first, second])
    client = PolzaClient(retries=1)
    assert client.models() == []
    assert first.closed and len(calls) == 2
    assert [r["method"] for r in first.requests] == ["HEAD", "GET"]
    assert [r["method"] for r in second.requests] == ["HEAD","GET"]


@pytest.mark.parametrize("failure", [TimeoutError("offline timeout"), Response(status=503)])
def test_paid_post_never_retries_after_sending(stub_network, failure):
    connections, calls = stub_network
    connection = Connection([Response(),failure])
    connections.append(connection)
    client = PolzaClient(api_key="offline", retries=3)
    with pytest.raises(ProviderError):
        client._request("POST", "/chat/completions", {"model": "qwen/qwen3.6-27b"})
    assert len(calls) == 1 and [r["method"] for r in connection.requests]==["HEAD","POST"]
    assert "Authorization" not in connection.requests[0]["headers"]


def test_pre_request_connect_failure_may_retry_paid_post(stub_network):
    connections, calls = stub_network
    first = Connection(connect_failure=TimeoutError("connect failed before sending"))
    second = Connection([Response(),Response()])
    connections.extend([first, second])
    client = PolzaClient(api_key="offline", retries=1)
    assert client._request("POST", "/chat/completions", {}) == {"ok": True}
    assert first.closed and not first.requests and len(calls) == 2
    assert [r["method"] for r in second.requests]==["HEAD","POST"]


def test_http_proxy_tunnels_to_verified_destination_without_exposing_credentials(monkeypatch, stub_network):
    connections, calls = stub_network
    monkeypatch.setattr(module.urllib.request, "getproxies", lambda: {"https": "http://user:private-proxy-secret@proxy.example:8080"})
    monkeypatch.setattr(module.urllib.request, "proxy_bypass", lambda host: False)
    connection = Connection([Response(), Response(payload={"data": []})])
    connections.append(connection)
    client = PolzaClient(api_key="offline")
    assert client.models() == []
    assert calls[0][:2] == ("proxy.example", 8080)
    assert calls[0][2]["context"].check_hostname
    assert calls[0][2]["context"].verify_mode == ssl.CERT_REQUIRED
    assert connection.tunnel[:2] == ("polza.ai", 443)
    assert "Proxy-Authorization" in connection.tunnel[2]
    assert all("Authorization" not in row["headers"] and "Proxy-Authorization" not in row["headers"] for row in connection.requests)
    assert "private-proxy-secret" not in json.dumps(client.transport_diagnostics)


@pytest.mark.parametrize("proxy", ["https://user:private-proxy-secret@proxy.example", "socks5://proxy.example:1080", "http://proxy.example/path"])
def test_unsupported_proxy_fails_before_paid_send_without_credentials(monkeypatch, stub_network, proxy):
    connections, calls = stub_network
    monkeypatch.setattr(module.urllib.request, "getproxies", lambda: {"https": proxy})
    monkeypatch.setattr(module.urllib.request, "proxy_bypass", lambda host: False)
    client = PolzaClient(api_key="offline")
    with pytest.raises(ProviderError) as caught:
        client.complete([])
    assert "proxy" in str(caught.value).lower()
    assert "private-proxy-secret" not in str(caught.value)
    assert client.records[0]["request_sent"] is False
    assert client.records[0]["total_tokens"] == 0 and client.records[0]["cost_rub"] == 0
    assert not calls


def test_no_proxy_bypass_uses_direct_verified_connection(monkeypatch, stub_network):
    connections, calls = stub_network
    monkeypatch.setattr(module.urllib.request, "getproxies", lambda: {"https": "http://proxy.example:8080"})
    monkeypatch.setattr(module.urllib.request, "proxy_bypass", lambda host: host == "polza.ai")
    connections.append(Connection([Response(), Response(payload={"data": []})]))
    assert PolzaClient().models() == []
    assert calls[0][:2] == ("polza.ai", 443)


@pytest.mark.parametrize("url", ["http://polza.ai/api/v1", "https://relay.example/api/v1", "https://polza.ai:8443/api/v1",
                                  "https://user:secret@polza.ai/api/v1", "https://polza.ai/api/v1?key=secret"])
def test_live_transport_rejects_unauthorized_origins(url):
    with pytest.raises(ProviderError, match="requires HTTPS to polza.ai"):
        PolzaClient(api_key="offline", base_url=url)


def test_context_manager_closes_connection(stub_network):
    connections, calls = stub_network
    connection = Connection([Response(), Response(payload={"data": []})])
    connections.append(connection)
    with PolzaClient(retries=0) as client:
        client.models()
    assert connection.closed


def test_connection_timeout_is_short_without_shortening_inference(stub_network):
    connections, calls = stub_network
    connection = Connection([Response(),Response()])
    connections.append(connection)
    client = PolzaClient(api_key="offline",timeout=180,connect_timeout=5,retries=0)
    client._request("POST","/chat/completions",{})
    assert calls[0][2]["timeout"] == 5
    assert connection.timeout == 180
    assert [(r["method"],r["socket_timeout"]) for r in connection.requests]==[("HEAD",5),("POST",180)]


def test_live_transport_assembles_stream_and_accounts_only_final_usage(stub_network,tmp_path):
    import io
    connections,calls=stub_network
    class SSE(Response):
        def __init__(self):
            super().__init__()
            self.body=io.BytesIO((
                'data: {"id":"gen-stream","model":"qwen/qwen3.6-27b","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15,"cost_rub":0.02}}\n\n'
                'data: [DONE]\n\n').encode())
        def readline(self,size): return self.body.readline(size)
    connection=Connection([Response(),SSE()]); connections.append(connection)
    client=PolzaClient(api_key="offline",stream=True,usage_path=tmp_path/"usage.jsonl",retries=0)
    raw=client.complete([],operation="extraction")
    assert raw["choices"][0]["message"]["content"]=="ok"
    assert client.records[0]["total_tokens"]==15 and client.records[0]["cost_rub"]==0.02
    progress=[json.loads(line) for line in (tmp_path/"transport_events.jsonl").read_text().splitlines()]
    assert progress[0]["operation"]=="extraction" and progress[0]["request_id"]=="gen-stream"
    assert "content" not in progress[0]
    assert json.loads(connection.requests[-1]["body"])["stream"] is True


def test_interrupted_paid_stream_preserves_request_id_and_unknown_billing(stub_network,tmp_path):
    import io
    connections,calls=stub_network
    class BrokenStream(Response):
        def __init__(self):
            super().__init__()
            self.body=io.BytesIO(b'data: {"id":"gen-partial","choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n')
        def readline(self,size):
            value=self.body.readline(size)
            if not value: raise TimeoutError("offline lost stream")
            return value
    connections.append(Connection([Response(),BrokenStream()]))
    client=PolzaClient(api_key="offline",stream=True,usage_path=tmp_path/"usage.jsonl",retries=3)
    with pytest.raises(ProviderError,match="no automatic paid retry"):
        client.complete([])
    assert len(calls)==1 and client.records[0]["request_id"]=="gen-partial"
    assert client.records[0]["total_tokens"] is None and client.records[0]["cost_rub"] is None


def test_exact_catalog_snapshot_is_frozen_per_client_without_repeat_fetch():
    calls=[]
    def transport(method,path,payload):
        calls.append((method,path))
        return {"data":[{"id":"qwen/qwen3.6-27b","name":"Qwen"}]}
    client=PolzaClient(transport=transport)
    catalog=client.verify_model(); catalog["name"]="caller mutation"
    assert client.verify_model()["name"]=="Qwen" and len(calls)==1


def test_saved_catalog_records_provenance_without_claiming_live_availability(tmp_path):
    path=tmp_path/"catalog.json"
    path.write_text(json.dumps({"data":[{"id":"qwen/qwen3.6-27b"}]}))
    client=PolzaClient(transport=lambda *args:pytest.fail("snapshot does not make a network request"))
    assert client.verify_model(snapshot_path=path)["id"]==client.model_name
    assert client.catalog_provenance["mode"]=="cached_catalog_snapshot"
    assert client.catalog_provenance["availability_requires_live_exact_model_inference"] is True
    assert len(client.catalog_provenance["sha256"])==64
    with pytest.raises(ProviderError,match="already frozen"): client.verify_model(snapshot_path=path)
