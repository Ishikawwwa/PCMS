"""History controls and shared causal/evidence machinery (stdlib only)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from typing import Any


FORBIDDEN_FIELDS = {"gold", "gold_plan", "gold_constraints", "gold_answer", "reference_plan", "target_plan", "hidden_answer", "future_events", "oracle"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def token_estimate(text: str) -> int:
    """Reproducible byte-based approximation, never advertised as model tokens."""
    return math.ceil(len(text.encode("utf-8")) / 4)


def lexical(text: Any) -> list[str]:
    return re.findall(r"[\w]+", text.lower() if isinstance(text, str) else canonical(text).lower())


def reject_privileged(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"Privileged field prohibited in normal memory input: {key}")
            reject_privileged(item)
    elif isinstance(value, list):
        for item in value:
            reject_privileged(item)


class MemoryBackend:
    variant = "BASE"

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self._token_counter = self.config.pop("token_counter", None)
        self._token_counter_name = self.config.pop("token_counter_name", "custom_token_counter" if self._token_counter else "ceil(utf8_bytes/4)")
        self.budget = int(self.config.get("token_budget", self.config.get("memory_token_budget", 2048)))
        if self.budget < 1:
            raise ValueError("token_budget must be positive")
        self.reset("")

    def set_token_counter(self, counter, name: str = "exact_model_tokenizer") -> None:
        """Attach a runtime tokenizer without putting callables in replay config."""
        if not callable(counter):
            raise TypeError("token counter must be callable")
        self._token_counter = counter
        self._token_counter_name = name

    def _token_count(self, text: str) -> int:
        count = int(self._token_counter(text)) if self._token_counter else token_estimate(text)
        if count < 0:
            raise ValueError("Token counter returned a negative value")
        return count

    def _bounded_notice(self, text: str) -> str:
        while text and self._token_count(text) > self.budget:
            text = text[:-1]
        return text

    def reset(self, group_id: str) -> None:
        self.group_id = str(group_id)
        self.events: dict[str, dict] = {}
        self.event_order: list[str] = []
        self.current_step = -1
        self._request: dict = {}
        self._metrics = Counter()
        self._last_context: dict = {}
        self._diagnostics: list[dict] = []

    def _check_group(self, value: dict) -> None:
        reject_privileged(value)
        supplied = value.get("group_id", value.get("group"))
        if supplied is not None and str(supplied) != self.group_id:
            raise ValueError(f"Group mismatch: memory={self.group_id!r}, input={supplied!r}")

    def observe(self, event: dict) -> None:
        self._check_group(event)
        for key in ("event_id", "step", "type", "content"):
            if key not in event:
                raise ValueError(f"Evidence missing required field {key}")
        eid = str(event["event_id"])
        step = int(event["step"])
        if step < 0:
            raise ValueError("Evidence step must be nonnegative")
        saved = {"event_id": eid, "step": step, "type": str(event["type"]),
                 "content": copy.deepcopy(event["content"]), "group_id": self.group_id}
        saved["content_hash"] = hashlib.sha256(canonical(saved["content"]).encode("utf-8")).hexdigest()
        saved["immutable_pointer"] = f"event:{eid}"
        if eid in self.events:
            if canonical(saved) != canonical(self.events[eid]):
                raise ValueError(f"Immutable evidence conflict: {eid}")
            return
        self.events[eid] = saved
        self.event_order.append(eid)
        self.current_step = max(self.current_step, step)
        self._metrics["observed_events"] += 1

    def _prepare(self, request: dict) -> int:
        self._check_group(request)
        self._request = copy.deepcopy(request)
        self._diagnostics = []
        self._metrics["retrievals"] += 1
        return int(request.get("step", max(self.current_step, int(request.get("round_idx", 0)))))

    def _visible_events(self, step: int) -> list[dict]:
        return [self.events[eid] for eid in self.event_order if self.events[eid]["step"] <= step]

    def _response(self, text: str, *, status: str = "OK", **fields: Any) -> dict:
        used = self._token_count(text)
        self._metrics["retrieved_token_estimate"] += used
        self._metrics["peak_context_token_estimate"] = max(self._metrics["peak_context_token_estimate"], used)
        response = {"variant": self.variant, "context": text, "text": text, "status": status,
                    "diagnostics": copy.deepcopy(self._diagnostics), "token_estimate": used,
                    "token_estimator": self._token_counter_name, "token_count": used, "token_budget": self.budget,
                    "closure_ids": [], "missing_refs": [], "batches": [], **fields}
        self._last_context = copy.deepcopy(response)
        return response

    def begin_subtask(self, request: dict) -> dict:
        raise NotImplementedError

    def _record_action(self, result: dict) -> str:
        self._check_group(result)
        action = result.get("action", result.get("submission", result.get("answer", "")))
        ordinal = self._metrics["commits"] + 1
        eid = f"action:{self.group_id}:{result.get('traveler', self._request.get('traveler', 'unknown'))}:{ordinal}"
        self.observe({"event_id": eid, "step": int(result.get("step", self._request.get("step", max(0, self.current_step)))),
                      "type": "agent_action", "content": action})
        self._metrics["commits"] += 1
        return eid

    def commit_subtask(self, result: dict) -> dict:
        eid = self._record_action(result)
        return {"status": "OK", "action_event_id": eid, "diagnostics": []}

    def snapshot(self) -> dict:
        return {"variant": self.variant, "group_id": self.group_id,
                "evidence": copy.deepcopy(list(self.events.values()))}

    def retrieve_evidence(self, ref: str, step: int | None = None) -> dict:
        """Retrieve archived evidence by its immutable ID, with a causal bound."""
        eid = ref[6:] if ref.startswith("event:") else ref
        if eid not in self.events:
            raise KeyError(ref)
        event = self.events[eid]
        if step is not None and event["step"] > step:
            raise ValueError(f"Evidence {ref} is not causally visible at step {step}")
        return copy.deepcopy(event)

    def stats(self) -> dict:
        return {"variant": self.variant, "group_id": self.group_id, **dict(self._metrics),
                "storage_bytes": len(canonical(self.snapshot()).encode("utf-8")),
                "retrieval_visible_token_estimate": self._last_context.get("token_estimate", 0),
                "token_estimator": self._token_counter_name}


class NoMemory(MemoryBackend):
    variant = "B0"

    def observe(self, event: dict) -> None:
        self._check_group(event)
        self._metrics["observed_events"] += 1

    def begin_subtask(self, request: dict) -> dict:
        self._prepare(request)
        return self._response("")

    def commit_subtask(self, result: dict) -> dict:
        self._check_group(result)
        self._metrics["commits"] += 1
        return {"status": "OK", "action_event_id": None, "diagnostics": []}


class FullHistoryMemory(MemoryBackend):
    variant = "B1"

    def begin_subtask(self, request: dict) -> dict:
        step = self._prepare(request)
        events = self._visible_events(step)
        text = "PAST ALLOWED TRACE\n" + "\n".join(canonical(event) for event in events)
        status = "BUDGET_OVERFLOW" if self._token_count(text) > self.budget else "OK"
        if status != "OK":
            self._diagnostics.append({"code": status, "message": "Full-history control is untruncated; planner must report an explicit context failure if it cannot accept this trace."})
            self._metrics["budget_overflows"] += 1
        return self._response(text, status=status, evidence_ids=[e["event_id"] for e in events])


class BM25Memory(MemoryBackend):
    variant = "B2"

    def begin_subtask(self, request: dict) -> dict:
        step = self._prepare(request)
        documents = self._visible_events(step)
        query = lexical(request.get("query", request.get("request", "")))
        terms = [Counter(lexical(event["content"])) for event in documents]
        lengths = [sum(term.values()) for term in terms]
        average = sum(lengths) / max(1, len(lengths))
        df = Counter(word for term in terms for word in term)
        ranked = []
        for index, (event, term, length) in enumerate(zip(documents, terms, lengths)):
            score = 0.0
            for word in set(query):
                frequency = term[word]
                if not frequency:
                    continue
                idf = math.log(1 + (len(documents) - df[word] + .5) / (df[word] + .5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * length / max(1, average)))
            if score:
                ranked.append((score, index, event))
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        selected = []
        lines = ["RETRIEVED PAST ALLOWED TRACE (BM25)"]
        omitted = []
        for score, _, event in ranked[:int(self.config.get("top_k", 8))]:
            line = canonical(event)
            if self._token_count("\n".join(lines + [line])) <= self.budget:
                lines.append(line)
                selected.append(event["event_id"])
            else:
                omitted.append(event["event_id"])
        return self._response(self._bounded_notice("\n".join(lines)), evidence_ids=selected, omitted_evidence_ids=omitted,
                              scores={event["event_id"]: round(score, 8) for score, _, event in ranked})
