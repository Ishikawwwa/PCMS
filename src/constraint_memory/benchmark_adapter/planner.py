"""Run the official planner in a process receiving only one causal request.

The worker owns tools and the unchanged upstream ReAct loop. Model calls travel
over JSON stdio to a parent-owned client; credentials and hidden dataset paths
are not sent to the worker. This is a data-visibility boundary, not a hostile-code
OS sandbox. The official tools expose no filesystem or arbitrary-code action.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .data import assert_visible_request


class PlannerWorker:
    def __init__(self, upstream_root: Path, db_path: Path, client, *, model: str = "qwen/qwen3.6-27b",
                 temperature: float = 0.0, max_steps: int = 30, timeout_seconds: float = 300,
                 denied_paths=(),visible_travelers=None,causal_round=None,task_contract_text=None,
                 tool_city_state_annotation_normalization=False,tool_price_unit=None):
        self.upstream_root = Path(upstream_root).resolve()
        self.db_path = Path(db_path).resolve()
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.denied_paths = [str(Path(p).resolve()) for p in denied_paths]
        self.visible_travelers=dict(visible_travelers or {})
        self.causal_round=causal_round
        if task_contract_text is not None and (not isinstance(task_contract_text,str) or not task_contract_text.strip()):
            raise ValueError("Declared task contract must be nonempty text")
        self.task_contract_text=task_contract_text
        if not isinstance(tool_city_state_annotation_normalization,bool):
            raise ValueError("Tool city adapter must be an explicit shared boolean")
        self.tool_city_state_annotation_normalization=tool_city_state_annotation_normalization
        if tool_price_unit not in {None,"USD"}:
            raise ValueError("Unknown declared travel price unit")
        self.tool_price_unit=tool_price_unit
        self.model_requests_sent=0

    def run_turn(self, request: dict, memory_context: str, base_person: dict) -> dict:
        self.model_requests_sent=0
        assert_visible_request(request)
        if set(base_person) != {"name", "query", "daily_plans"}:
            raise ValueError("Base traveler must contain only public name, query, and fixed daily_plans")
        if not isinstance(memory_context, str):
            raise ValueError("Memory context must be serialized text")
        root = self.upstream_root.parent.parent
        # Keep only interpreter/OS necessities; all credentials and dataset/cache
        # environment settings stay in the parent.
        permitted_environment = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT", "USERPROFILE", "LOCALAPPDATA"}
        worker_env = {k: v for k, v in os.environ.items() if k.upper() in permitted_environment}
        worker_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        worker_env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen([sys.executable, "-m", "constraint_memory.benchmark_adapter.planner_worker"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   encoding="utf-8", env=worker_env, cwd=str(root), bufsize=1)
        output_queue: queue.Queue = queue.Queue()
        debug_output: list[str] = []

        def read_stdout():
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        def read_stderr():
            for line in process.stderr:
                if sum(map(len, debug_output)) < 100_000:
                    debug_output.append(line)

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        payload = {"upstream_root": str(self.upstream_root), "db_path": str(self.db_path), "model": self.model,
                   "temperature": self.temperature, "max_steps": self.max_steps, "request": request,
                   "base_person": base_person, "memory_context": memory_context,
                   "task_contract_text":self.task_contract_text,
                   "tool_city_state_annotation_normalization":self.tool_city_state_annotation_normalization,
                   "tool_price_unit":self.tool_price_unit,
                   "visible_travelers":self.visible_travelers or {request["name"]:request["traveler_index"]},
                   "causal_round":self.causal_round or request["traveler_index"],
                   "denied_paths": [str(root / "data/hidden"), str(root / "data/visible"), str(root / "data/audit_annotations"), str(root / ".env"), *self.denied_paths]}
        start = time.monotonic()
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            while True:
                remaining = self.timeout_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("Planner worker wall-clock limit exceeded")
                try:
                    line = output_queue.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError("Planner worker did not return before wall-clock limit") from exc
                if line is None:
                    raise RuntimeError("Planner worker exited without a result: " + "".join(debug_output)[-4000:])
                message = json.loads(line)
                if message["kind"] == "model_request":
                    if self.model_requests_sent>=self.max_steps:
                        raise RuntimeError("Official planner exceeded its remaining step allowance")
                    self.model_requests_sent+=1
                    response = self.client.chat_with_tools(message["messages"], message["tools"])
                    calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls or []]
                    raw = response.raw_response if isinstance(response.raw_response, dict) else None
                    result = {"content": response.content, "tool_calls": calls, "raw_response": raw}
                    process.stdin.write(json.dumps(result, ensure_ascii=False) + "\n")
                    process.stdin.flush()
                elif message["kind"] == "result":
                    result = message["result"]
                    result["parent_model_requests"]=self.model_requests_sent
                    if result.get("total_steps")!=self.model_requests_sent:
                        raise RuntimeError("Official planner step count differs from parent dispatch count")
                    result["latency_seconds"] = time.monotonic() - start
                    result["worker_debug"] = "".join(debug_output)
                    return result
                elif message["kind"] == "error":
                    raise RuntimeError(message["error"])
                else:
                    raise RuntimeError("Unexpected worker message")
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            for stream in [process.stdin, process.stdout, process.stderr]:
                if stream:
                    stream.close()
