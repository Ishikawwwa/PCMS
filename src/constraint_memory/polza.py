"""Polza Chat Completions transport with exact provider usage accounting."""
from __future__ import annotations

import json
import base64
import copy
import datetime as dt
import hashlib
import http.client
import math
import os
import ssl
import time
import urllib.request
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .secrets import redact
from .streaming import StreamError, read_completion_stream


class ProviderError(RuntimeError):
    def __init__(self,message,request_sent=None,request_id=None,usage=None):
        super().__init__(message)
        self.request_sent=request_sent
        self.request_id=request_id
        self.usage=usage


class BudgetExceeded(RuntimeError):
    pass


class PolzaClient:
    def __init__(self, model="qwen/qwen3.6-27b", base_url="https://polza.ai/api/v1",
                 api_key=None, temperature=0.0, max_tokens=4096, seed=20261013,
                 usage_path=None, max_total_tokens=500000, max_cost_rub=100.0,
                 timeout=90, retries=2, provider=None, reasoning=None, transport=None, connect_timeout=10,stream=False,
                 extraction_reasoning=None,extraction_max_tokens=None):
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self._origin = urllib.parse.urlsplit(self.base_url)
        try:
            valid_origin = (self._origin.scheme == "https" and self._origin.hostname == "polza.ai"
                            and self._origin.port in (None, 443) and not self._origin.username
                            and not self._origin.password and not self._origin.query and not self._origin.fragment)
        except ValueError:
            valid_origin = False
        if not valid_origin and transport is None:
            raise ProviderError("Live Polza transport requires HTTPS to polza.ai on port 443 without URL credentials")
        self.api_key = api_key or os.environ.get("POLZA_API_KEY")
        self.temperature, self.max_tokens, self.seed = temperature, max_tokens, seed
        self.timeout, self.retries = timeout, retries
        self.connect_timeout = min(connect_timeout, timeout)
        self.provider, self.reasoning = provider, reasoning
        self.stream=stream
        self.extraction_reasoning=extraction_reasoning
        self.extraction_max_tokens=extraction_max_tokens or max_tokens
        self.usage_path = Path(usage_path) if usage_path else None
        self.max_total_tokens, self.max_cost_rub = max_total_tokens, max_cost_rub
        self.transport = transport
        self.records: list[dict] = []
        self.group_id = None
        self.traveler = None
        self.operation = "agent"
        self._connection = None
        self._catalog_warmed = False
        self._closed = False
        self.transport_diagnostics: list[dict] = []
        self._verified_model=None
        self.catalog_provenance={"mode":"not_verified"}

    def _discard_connection(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._catalog_warmed = False

    def close(self):
        """Close the persistent transport; model-accounting records remain available."""
        self._discard_connection()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise ProviderError("Polza client is closed",request_sent=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _connect(self):
        proxies = urllib.request.getproxies()
        configured_proxy = proxies.get("https") or proxies.get("all")
        if self._connection is None:
            host, port = self._origin.hostname, 443
            tunnel_headers = {}
            via_proxy = configured_proxy and not urllib.request.proxy_bypass(host)
            if via_proxy:
                # CONNECT carries no Polza credentials. HTTPSConnection verifies
                # the destination's certificate using its tunnel host. Never
                # include proxy URLs or authentication in exceptions or logs.
                try:
                    proxy = urllib.parse.urlsplit(configured_proxy)
                    if (proxy.scheme != "http" or not proxy.hostname or proxy.path not in {"", "/"}
                            or proxy.query or proxy.fragment or not 1 <= (proxy.port or 80) <= 65535):
                        raise ValueError("Unsupported proxy")
                    host, port = proxy.hostname, proxy.port or 80
                    if proxy.username is not None or proxy.password is not None:
                        credentials = urllib.parse.unquote(proxy.username or "") + ":" + urllib.parse.unquote(proxy.password or "")
                        tunnel_headers["Proxy-Authorization"] = "Basic " + base64.b64encode(credentials.encode()).decode("ascii")
                except (TypeError, ValueError):
                    raise ProviderError("Configured proxy requires a supported HTTP CONNECT endpoint",request_sent=False) from None
            self._connection = http.client.HTTPSConnection(host, port,
                timeout=self.connect_timeout, context=ssl.create_default_context())
            if via_proxy:
                self._connection.set_tunnel(self._origin.hostname,443,headers=tunnel_headers)
                self.transport_diagnostics.append({"operation":"configured_proxy_connect","destination":"polza.ai:443",
                    "proxy_authentication_configured":bool(tunnel_headers),"billable_model_call":False})
        # Connect explicitly before sending paid request bytes. A failure here is
        # unambiguously pre-request, and can be retried within the configured cap.
        if self._connection.sock is None:
            self._catalog_warmed = False
            self._connection.timeout = self.connect_timeout
            self._connection.connect()
            self._connection.timeout = self.timeout
            self._connection.sock.settimeout(self.timeout)
        return self._connection

    def _warm_catalog_connection(self, connection):
        if self._catalog_warmed:
            return
        started = time.perf_counter()
        connection.sock.settimeout(self.connect_timeout)
        connection.request("HEAD", "/", headers={"User-Agent": "constraint-memory-lab/0.1", "Connection": "keep-alive"})
        response = connection.getresponse()
        response.read()
        if not 200 <= response.status < 300:
            raise ProviderError(f"Polza public transport warmup returned HTTP {response.status}")
        self._catalog_warmed = True
        self.transport_diagnostics.append({"operation": "public_catalog_connection_warmup", "method": "HEAD",
            "path": "/", "status": response.status, "authenticated": False,
            "latency_seconds": time.perf_counter() - started, "billable_model_call": False})

    def _transport_event(self,values):
        if self.usage_path:
            target=self.usage_path.with_name("transport_events.jsonl")
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(redact({"group_id":self.group_id,"traveler":self.traveler,"operation":self.operation,**values}),ensure_ascii=False)+"\n")

    def set_scope(self, group_id, traveler=None, operation="agent"):
        self.group_id, self.traveler, self.operation = group_id, traveler, operation

    def _request(self, method, path, payload=None, authenticated=True):
        if self._closed:
            raise ProviderError("Polza client is closed")
        if authenticated and not self.api_key:
            raise ProviderError("POLZA_API_KEY is missing. Set it in the environment or ignored .env file.",request_sent=False)
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "constraint-memory-lab/0.1", "Connection": "keep-alive"}
        if authenticated:
            headers["Authorization"] = "Bearer " + self.api_key
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        if self.transport:
            return self.transport(method, path, payload)
        if method not in {"GET", "POST", "HEAD"} or not path.startswith("/") or path.startswith("//") or "\r" in path or "\n" in path:
            raise ProviderError("Unsupported Polza request method or relative API path")
        target = self._origin.path.rstrip("/") + path
        safe_read = method in {"GET", "HEAD"}
        attempts = min(max(int(self.retries), 0), 3) + 1
        for attempt in range(attempts):
            request_started = False
            try:
                connection = self._connect()
                if not self._catalog_warmed:
                    self._warm_catalog_connection(connection)
                read_timeout=self.connect_timeout if safe_read else self.timeout
                connection.timeout=read_timeout
                if connection.sock is not None:
                    connection.sock.settimeout(read_timeout)
                request_started = True
                connection.request(method, target, body=body, headers=headers)
                response = connection.getresponse()
                if not 200 <= response.status < 300:
                    response.read()
                    # A POST is never repeated after request bytes may have been
                    # sent, even on 429/5xx. Remote error bodies may echo secrets.
                    if safe_read and response.status in {429, 502, 503, 504} and attempt < attempts - 1:
                        self._discard_connection()
                        time.sleep(min(2 ** attempt, 4))
                        continue
                    raise ProviderError(f"Polza returned HTTP {response.status} for {urllib.parse.urlsplit(target).path}")
                if payload and payload.get("stream"):
                    started=time.perf_counter()
                    last_progress=-10.0
                    def progress(values):
                        nonlocal last_progress
                        elapsed=time.perf_counter()-started
                        if elapsed-last_progress>=10:
                            self._transport_event({"status":"streaming","elapsed_seconds":elapsed,**values})
                            last_progress=elapsed
                    try:
                        return read_completion_stream(response,on_progress=progress)
                    except StreamError as error:
                        self._discard_connection()
                        raise ProviderError(f"Polza completion stream failed ({error.code}): {error}; no automatic paid retry",
                            request_sent=True,request_id=error.request_id,usage=error.usage) from None
                raw = response.read()
                try:
                    return json.loads(raw)
                except (ValueError, UnicodeError):
                    raise ProviderError("Polza returned an invalid JSON response") from None
            except ssl.SSLCertVerificationError:
                self._discard_connection()
                raise ProviderError("Polza TLS certificate verification failed") from None
            except (OSError, http.client.HTTPException) as exc:
                self._discard_connection()
                if (safe_read or not request_started) and attempt < attempts - 1:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                qualifier = "no automatic ambiguous POST retry" if not safe_read and request_started else "safe connection retries exhausted"
                raise ProviderError(f"Polza network request failed ({type(exc).__name__}); {qualifier}",request_sent=request_started) from None

    def models(self):
        result = self._request("GET", "/models?type=chat", authenticated=False)
        return result.get("data", result.get("models", []))

    def verify_model(self,snapshot_path=None):
        if self._verified_model is not None:
            if snapshot_path is not None:
                raise ProviderError("The model catalog is already frozen for this client",request_sent=False)
            return copy.deepcopy(self._verified_model)
        if snapshot_path is not None:
            path=Path(snapshot_path)
            raw_bytes=path.read_bytes()
            snapshot=json.loads(raw_bytes)
            models=snapshot.get("data",snapshot.get("models",[])) if isinstance(snapshot,dict) else snapshot
            provenance={"mode":"cached_catalog_snapshot","path":str(path),"sha256":hashlib.sha256(raw_bytes).hexdigest(),
                "snapshot_mtime_utc":dt.datetime.fromtimestamp(path.stat().st_mtime,dt.timezone.utc).isoformat(),
                "availability_requires_live_exact_model_inference":True}
        else:
            models=self.models()
            provenance={"mode":"live_catalog_fetch","origin":self.base_url+"/models?type=chat",
                "fetched_utc":dt.datetime.now(dt.timezone.utc).isoformat()}
        if not isinstance(models,list):
            raise ProviderError("Catalog snapshot lacks a model list",request_sent=False)
        matches = [m for m in models if isinstance(m,dict) and m.get("id") == self.model_name]
        if not matches:
            raise ProviderError(f"Exact configured model {self.model_name} is unavailable. No substitution is allowed.")
        self._verified_model=copy.deepcopy(matches[0])
        self.catalog_provenance=provenance
        return copy.deepcopy(self._verified_model)

    def _check_budget(self, messages, output_cap, tools=None):
        records = [r for r in self.records if r.get("group_id") == self.group_id]
        if any(r.get("total_tokens") is None for r in records):
            raise BudgetExceeded("A previous model request has unknown token usage; the group cannot safely continue")
        used = sum(r["total_tokens"] for r in records)
        # Conservative pre-call envelope for English travel prompts; actual provider tokens logged.
        reserve = (sum(len(json.dumps(m, ensure_ascii=False).encode("utf-8")) for m in messages)
                   + (len(json.dumps(tools,ensure_ascii=False).encode("utf-8")) if tools else 0)
                   + output_cap + 128)
        if used + reserve > self.max_total_tokens:
            raise BudgetExceeded("Group total-token budget cannot accommodate the next request")
        costs = [r["cost_rub"] for r in records]
        if any(c is None for c in costs) and self.max_cost_rub is not None:
            raise BudgetExceeded("Provider omitted cost; cannot enforce the requested monetary cap")
        if self.max_cost_rub is not None and sum(costs) >= self.max_cost_rub:
            raise BudgetExceeded("Group RUB spending cap reached (one in-flight request may exceed it)")

    def complete(self,messages,tools=None,operation=None,max_tokens=None,json_output=False,response_schema=None):
        started=time.perf_counter()
        count_before=len(self.records)
        previous_operation=self.operation
        self.operation=operation or self.operation
        try:
            return self._complete(messages,tools,operation,max_tokens,json_output,response_schema)
        except ProviderError as error:
            if len(self.records)==count_before:
                sent=error.request_sent
                op=operation or self.operation
                usage=error.usage or {}
                row={"group_id":self.group_id,"traveler":self.traveler,"operation":op,
                    "category":"agent" if op=="agent" else "maintenance","model":self.model_name,
                    "request_status":"failed","request_sent":sent,"request_id":error.request_id,"error":redact(str(error)),
                    "input_tokens":usage.get("prompt_tokens",0 if sent is False else None),"output_tokens":usage.get("completion_tokens",0 if sent is False else None),
                    "total_tokens":usage.get("total_tokens",0 if sent is False else None),"cost_rub":usage.get("cost_rub",0 if sent is False else None),
                    "currency":"RUB","latency_seconds":time.perf_counter()-started,
                    "billing_status":"provider_usage_received_for_failed_request" if usage else "not_sent" if sent is False else "unknown_after_failed_request"}
                self._record_usage(row)
            raise
        finally:
            self.operation=previous_operation

    def _record_usage(self,record):
        self.records.append(record)
        if self.usage_path:
            self.usage_path.parent.mkdir(parents=True,exist_ok=True)
            with self.usage_path.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(redact(record),ensure_ascii=False)+"\n")

    def _complete(self, messages, tools=None, operation=None, max_tokens=None, json_output=False,response_schema=None):
        op = operation or self.operation
        configured_cap=self.extraction_max_tokens if op=="extraction" else self.max_tokens
        cap = min(max_tokens or configured_cap, configured_cap)
        budget_tools={"tools":tools,"response_schema":response_schema} if response_schema is not None else tools
        self._check_budget(messages, cap, budget_tools)
        payload: dict[str, Any] = {"model": self.model_name, "messages": messages,
                                   "temperature": self.temperature, "max_tokens": cap, "seed": self.seed}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        if response_schema is not None:
            payload["response_format"]={"type":"json_schema","json_schema":{"name":"constraint_extraction","schema":response_schema,"strict":True}}
        if self.provider:
            payload["provider"] = self.provider
        requested_reasoning=self.extraction_reasoning if op=="extraction" and self.extraction_reasoning is not None else self.reasoning
        if requested_reasoning is not None:
            payload["reasoning"] = requested_reasoning
        if self.stream:
            payload["stream"]=True
        start = time.perf_counter()
        raw = self._request("POST", "/chat/completions", payload)
        elapsed = time.perf_counter() - start
        if not isinstance(raw, dict) or raw.get("error") or not raw.get("choices"):
            raise ProviderError("Polza returned no completion choices")
        usage = raw.get("usage")
        if not isinstance(usage, dict) or any(k not in usage for k in ("prompt_tokens", "completion_tokens")):
            raise ProviderError("Provider omitted token usage; run accounting would be incomplete")
        for field in ("prompt_tokens","completion_tokens","total_tokens"):
            if field not in usage:
                continue
            value=usage[field]
            if not isinstance(value,int) or isinstance(value,bool) or value<0:
                raise ProviderError("Provider returned invalid integer token usage")
        total=usage.get("total_tokens",usage["prompt_tokens"]+usage["completion_tokens"])
        if total<usage["prompt_tokens"]+usage["completion_tokens"]:
            raise ProviderError("Provider total_tokens understates prompt and completion usage")
        cost=usage.get("cost_rub",usage.get("cost"))
        if cost is not None and (not isinstance(cost,(int,float)) or isinstance(cost,bool) or not math.isfinite(cost) or cost<0):
            raise ProviderError("Provider returned invalid RUB billing")
        returned_model=raw.get("model")
        identity_status=("MISSING_OR_INVALID" if not isinstance(returned_model,str) or not returned_model.strip() else
                         "EXACT_MATCH" if returned_model==self.model_name else "MISMATCH")
        record = {"group_id": self.group_id, "traveler": self.traveler, "operation": op,
                  "category": "agent" if op == "agent" else "maintenance",
                  "request_status":"completed","routing_requested":self.provider,
                  "expected_model":self.model_name,"model_identity_validation":identity_status,
                  "response_accepted":identity_status=="EXACT_MATCH",
                  "reasoning_requested":requested_reasoning,"output_cap":cap,
                  "response_format_type":payload.get("response_format",{}).get("type"),
                  "request_id": raw.get("id"), "model": raw.get("model"), "provider": raw.get("provider"),
                  "input_tokens": int(usage["prompt_tokens"]), "output_tokens": int(usage["completion_tokens"]),
                  "total_tokens": total,
                  "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                  "cost_rub": cost, "currency": "RUB",
                  "latency_seconds": elapsed, "finish_reason": raw["choices"][0].get("finish_reason")}
        self._record_usage(record)
        # A received completion remains billable even when its returned model
        # identity cannot support this experiment. Persist its actual identity
        # and provider accounting before rejecting content; do not retry it or
        # append a second failure record in complete().
        if identity_status!="EXACT_MATCH":
            message=("Provider omitted a valid returned model identity; response rejected" if identity_status=="MISSING_OR_INVALID" else
                     "Provider returned a different model than configured; response rejected")
            raise ProviderError(message,request_sent=True,request_id=raw.get("id"),usage=copy.deepcopy(usage))
        if op=="extraction" and self.usage_path:
            target=self.usage_path.with_name("writer_responses.jsonl")
            with target.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(redact({"group_id":self.group_id,"traveler":self.traveler,
                    "request_id":raw.get("id"),"finish_reason":raw["choices"][0].get("finish_reason"),
                    "content":raw["choices"][0]["message"].get("content"),"diagnostic_only":True}),ensure_ascii=False)+"\n")
        return raw

    def chat_with_tools(self, messages, tools, temperature=None, max_tokens=None):
        if temperature is not None and temperature!=self.temperature:
            raise ProviderError("Call temperature differs from frozen experiment configuration")
        raw = self.complete(messages, tools,max_tokens=max_tokens)
        message = raw["choices"][0]["message"]
        calls = []
        for item in message.get("tool_calls") or []:
            try:
                arguments = json.loads(item["function"]["arguments"])
            except (ValueError, KeyError, TypeError):
                raise ProviderError("Model emitted malformed tool-call arguments") from None
            calls.append(SimpleNamespace(id=item["id"], name=item["function"]["name"], arguments=arguments))
        return SimpleNamespace(content=message.get("content"), tool_calls=calls or None, raw_response=raw)

    def format_assistant_tool_calls(self, tool_calls):
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in tool_calls]}

    def format_tool_result(self, tool_call_id, result, name=None):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}

    def get_usage_stats(self):
        costs = [r["cost_rub"] for r in self.records]
        return {"total_input_tokens": sum(r["input_tokens"] for r in self.records) if all(r["input_tokens"] is not None for r in self.records) else None,
                "total_output_tokens": sum(r["output_tokens"] for r in self.records) if all(r["output_tokens"] is not None for r in self.records) else None,
                "known_input_tokens":sum(r["input_tokens"] for r in self.records if r["input_tokens"] is not None),
                "known_output_tokens":sum(r["output_tokens"] for r in self.records if r["output_tokens"] is not None),
                "known_cost_rub":sum(c for c in costs if c is not None),
                "total_cost_rub": sum(costs) if all(c is not None for c in costs) else None,
                "currency": "RUB", "calls": len(self.records)}
