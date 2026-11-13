"""E3I: exact materialized tables plus an explicit unchanged-base contract.

Contract obligations stay outside the generic semantic validator. They check
only independently covered current-person unchanged slots, preserve submitted
choices, and disclose uncertainty without claiming overall task success.
"""
from __future__ import annotations

import copy
import hashlib

from .backends import canonical
from .inheritance import POLICY, SOURCE_KIND, compile_inheritance_obligations, validate_inheritance
from .subject_scope import compile_scope_coverage
from .table_frontier import TableFrontierMemory


class InheritedFrontierMemory(TableFrontierMemory):
    def __init__(self, config=None):
        options = dict(config or {})
        if options.get("public_base_outside_memory", True) is not True:
            raise ValueError("E3I requires the fixed public base in the common planner prompt")
        super().__init__(options)
        self.variant = "E3I"
        self.features["declared_inheritance"] = True

    def reset(self, group_id):
        super().reset(group_id)
        self._inheritance_contract = None
        self._base_traveler = None
        self._base_values = []
        self._base_fingerprint = None
        self._configured_at = None
        self._inheritance_compiled = {}
        self._inheritance_archive = []
        self._inheritance_checks = {}
        self._inheritance_check_history = []
        self._current_inheritance = None
        self._current_inheritance_issues = []
        self._current_inheritance_repair = None

    def configure_inheritance(self, contract, base_traveler):
        """Capture the already committed public base once, with observed policy."""
        if not isinstance(contract, dict):
            raise ValueError("E3I requires an explicit observed task contract")
        self._check_group(contract)
        if contract.get("group_id") != self.group_id or contract.get("source_kind") != SOURCE_KIND or contract.get("policy") != POLICY or contract.get("enabled") is not True:
            raise ValueError("E3I requires the enabled current-group TASK_CONTRACT policy")
        version, created = contract.get("version"), contract.get("created_at")
        refs, text = contract.get("source_refs"), contract.get("text")
        if (not contract.get("contract_id") or type(version) is not int or version < 1
                or type(created) is not int or not 0 <= created <= self.current_step
                or not isinstance(text, str) or not text.strip()
                or not isinstance(refs, list) or not refs
                or not isinstance(base_traveler, str) or not base_traveler):
            raise ValueError("Incomplete or future inheritance configuration")
        sources = []
        for ref in refs:
            event, error = self._evidence(ref, self.current_step)
            if error or event is None or event["group_id"] != self.group_id or event["type"] != "task_contract" or ref != f"event:{event['event_id']}":
                raise ValueError("Task contract must cite an already observed whole task_contract event")
            sources.append(event["content"])
        if not all(isinstance(content, str) for content in sources) or text not in sources:
            raise ValueError("Declared task contract text differs from observed evidence")
        if self._inheritance_contract is not None:
            if canonical(contract) != canonical(self._inheritance_contract) or base_traveler != self._base_traveler:
                raise ValueError("Configured public inheritance contract and base are immutable")
            return {"status": "CONFIGURED", "base_fingerprint": self._base_fingerprint,
                    "base_slot_count": len(self._base_values)}
        values = [record for record in self.committed_decisions(self.current_step)
                  if record["traveler"] == base_traveler and record.get("attribute", "value") == "value"]
        if not values:
            raise ValueError("Public base must be committed before inheritance configuration")
        for record in values:
            if record.get("status") != "COMMITTED" or not record.get("source_refs"):
                raise ValueError("Public base lacks valid committed provenance")
            for ref in record["source_refs"]:
                event, error = self._evidence(ref, self.current_step)
                if error or event is None or event["group_id"] != self.group_id:
                    raise ValueError("Public base evidence is not causally visible")
            if not self._trace(copy.deepcopy(record), self.current_step, decision=True):
                raise ValueError("Public base choices differ from their actual action evidence")
        self._inheritance_contract = copy.deepcopy(contract)
        self._base_traveler = base_traveler
        self._base_values = copy.deepcopy(sorted(values, key=lambda row: row["decision_id"]))
        self._base_fingerprint = hashlib.sha256(canonical(self._base_values).encode("utf-8")).hexdigest()
        self._configured_at = self.current_step
        return {"status": "CONFIGURED", "base_fingerprint": self._base_fingerprint,
                "base_slot_count": len(self._base_values)}

    def _request_evidence_issues(self, request, step):
        issues = []
        refs = request.get("source_refs")
        if not isinstance(refs, list) or not refs:
            issues.append({"code": "INHERITANCE_REQUEST_SOURCE_MISSING"})
        else:
            for ref in refs:
                event, error = self._evidence(ref, step)
                if (error or event is None or event["group_id"] != self.group_id or event["type"] != "request"
                        or ref != f"event:{event['event_id']}" or event["content"] != request.get("query")):
                    issues.append({"code": "INHERITANCE_REQUEST_SOURCE_UNVERIFIED", "reference": ref})
        prior = request.get("prior_travelers")
        if not isinstance(prior, list):
            issues.append({"code": "INHERITANCE_PRIOR_VISIBILITY_UNDECLARED"})
        else:
            visible = {task.get("scope", {}).get("traveler") for task in self.tasks.values()
                       if task.get("created_at", step + 1) <= step}
            visible.update(record["traveler"] for record in self.committed_decisions(step))
            for declared in prior:
                name = declared if isinstance(declared, str) else declared.get("traveler", declared.get("name")) if isinstance(declared, dict) else None
                if name == request.get("traveler") or name not in visible:
                    issues.append({"code": "INHERITANCE_PRIOR_NOT_VISIBLE", "traveler": name})
        return issues

    @staticmethod
    def _uncertain(compiled, issues):
        result = copy.deepcopy(compiled)
        result.update(status="UNCERTAIN", hard_checks_enabled=False)
        result.setdefault("diagnostics", []).extend(copy.deepcopy(issues))
        for record in result.get("constraints", []):
            record.update(status="UNRESOLVED", hard_check=False)
        for record in result.get("obligations", []):
            record["status"] = "UNRESOLVED"
        return result

    def _compile_current(self, request, step):
        issues = self._request_evidence_issues(request, step)
        if self._inheritance_contract is None or self._configured_at > step:
            issues.append({"code": "INHERITANCE_CONTRACT_NOT_VISIBLE"})
        try:
            certificate = compile_scope_coverage(request, request.get("prior_travelers"), request.get("source_refs"))
            compiled = compile_inheritance_obligations(self._inheritance_contract, self._base_values, request,
                request.get("extraction"), scope_coverage=certificate)
        except (TypeError, ValueError):
            certificate = None
            issues.append({"code": "INHERITANCE_CURRENT_SCOPE_INVALID"})
            compiled = {"schema_version": 1, "status": "UNCERTAIN", "group_id": self.group_id,
                "traveler": request["traveler"], "source_kind": SOURCE_KIND, "hard_checks_enabled": False,
                "obligations": [], "constraints": [], "exceptions": [], "diagnostics": []}
        if issues:
            compiled = self._uncertain(compiled, issues)
        compiled["causal_request_verified"] = not issues
        self._current_inheritance_issues = issues
        self._inheritance_compiled[request["traveler"]] = copy.deepcopy(compiled)
        self._inheritance_archive.append({"traveler": request["traveler"], "step": step,
            "certificate": copy.deepcopy(certificate), "compiled": copy.deepcopy(compiled)})
        return compiled

    def _current_values(self, traveler, step):
        return [record for record in self.committed_decisions(step)
                if record["traveler"] == traveler and record.get("attribute", "value") == "value"]

    def begin_subtask(self, request):
        self._check_group(request)
        step = request.get("step")
        if type(step) is not int or step < 0 or not isinstance(request.get("traveler"), str) or not request["traveler"]:
            raise ValueError("E3I requires an explicit current traveler and causal step")
        self._current_inheritance = None
        self._current_inheritance_issues = []
        self._current_inheritance_repair = None
        traveler = request["traveler"]
        # The public base's initial begin precedes configure_inheritance.
        if self._inheritance_contract is not None and traveler != self._base_traveler:
            self._current_inheritance = self._compile_current(request, step)
            previous = self._inheritance_checks.get(traveler)
            if previous and previous["step"] <= step:
                current_check = validate_inheritance(self._current_inheritance, self._current_values(traveler, step))
                if current_check["status"] == "VIOLATED":
                    self._current_inheritance_repair = current_check
        context = super().begin_subtask(request)
        if self._current_inheritance is not None:
            compiled = self._current_inheritance
            context.update(inheritance_status=compiled["status"],
                           inheritance_hard_checks_enabled=compiled["hard_checks_enabled"],
                           inheritance_contract_ref=compiled.get("contract_ref"))
            if not compiled["hard_checks_enabled"] and context["status"] == "OK":
                context["status"] = "UNRESOLVED"
            context["diagnostics"].extend(copy.deepcopy(self._current_inheritance_issues))
        self._last_context = copy.deepcopy(context)
        return context

    def _sections(self, records, definitions, outcomes, compact=False):
        sections = super()._sections(records, definitions, outcomes, compact=compact)
        compiled = self._current_inheritance
        if compiled is not None:
            exceptions = {}
            for record in compiled.get("exceptions", []):
                exceptions.setdefault(str(record["day"]), []).append(record["slot"])
            row = {"default": "keep unrequested slots of immutable public base in common prompt",
                   "base": self._base_traveler, "base_sha256": self._base_fingerprint,
                   "contract_ref": compiled.get("contract_ref"),
                   "source_refs": list(dict.fromkeys([*self._inheritance_contract["source_refs"],
                       *((compiled.get("scope_coverage") or {}).get("source_refs", []))])),
                   "status": compiled["status"], "exceptions": {day: sorted(set(slots)) for day, slots in sorted(exceptions.items(), key=lambda item: int(item[0]))}}
            if not compiled["hard_checks_enabled"]:
                row["uncertainty"] = sorted({issue["code"] for issue in compiled.get("diagnostics", [])})
            sections["DECLARED INHERITANCE"] = [row]
        if self._current_inheritance_repair:
            actual = {(record["day"], record["slot"]): record for record in
                      self._current_values(compiled["traveler"], self._request["step"])}
            rows = []
            for violation in self._current_inheritance_repair["violations"]:
                subject = violation["subject"]
                current = actual.get((subject["day"], subject["slot"]))
                row = {"traveler": compiled["traveler"], "day": subject["day"], "slot": subject["slot"],
                    "expected": violation["expected"], "actual": violation["actual"], "code": violation["code"],
                    "base_decision_ref": violation["base_decision_ref"],
                    "source_refs": list(dict.fromkeys([*violation["source_refs"], *(current.get("source_refs", []) if current else [])]))}
                if current:
                    row["actual_ref"] = f"{current['decision_id']}@v{current['version']}"
                rows.append(row)
            sections["INHERITANCE REPAIR"] = rows
        return sections

    def _batches(self, sections):
        # The parent batches only SECTION_NAMES; custom obligation and repair
        # rows must participate in the same envelope and remain explicit.
        names = list(sections)
        batches = []
        current = {name: [] for name in names}
        for name in names:
            for row in sections[name]:
                trial = copy.deepcopy(current)
                trial[name].append(row)
                if self._token_count(self._serialize(trial)) > self.budget and any(current.values()):
                    text = self._serialize(current)
                    batches.append({"batch_index": len(batches), "context": text, "token_estimate": self._token_count(text), "status": "OK"})
                    current = {key: [] for key in names}
                    trial = copy.deepcopy(current)
                    trial[name].append(row)
                if self._token_count(self._serialize(trial)) > self.budget:
                    text = self._serialize(trial)
                    batches.append({"batch_index": len(batches), "context": text, "token_estimate": self._token_count(text), "status": "RECORD_EXCEEDS_BUDGET"})
                    current = {key: [] for key in names}
                else:
                    current = trial
        if any(current.values()):
            text = self._serialize(current)
            batches.append({"batch_index": len(batches), "context": text, "token_estimate": self._token_count(text), "status": "OK"})
        return batches

    def commit_subtask(self, result):
        actual = copy.deepcopy(result)
        current = self._request.get("traveler", actual.get("traveler"))
        extraction = actual.get("extraction")
        incoming = actual.get("decisions", extraction.get("decisions", []) if isinstance(extraction, dict) else [])
        current_present = any(isinstance(record, dict) and all(key in record for key in ("day", "slot", "value")) and record.get("traveler") == current and record.get("attribute", "value") == "value" for record in incoming or [])
        # The coordinator distinguishes an absent heading (no replacement)
        # from an accepted explicit empty full section (withdraw old slots).
        # Preserve that exact protocol instead of inferring from row count.
        # Failure to satisfy requirements does not undo an actual submission.
        # There is no branch rollback or environment rollback in this backend.
        if actual.get("branch_status") == "FAILED":
            actual.pop("branch_status")
        committed = super().commit_subtask(actual)
        check_step = self.events[committed["action_event_id"]]["step"]
        if self._inheritance_contract is None or current == self._base_traveler:
            return committed
        compiled = self._inheritance_compiled.get(current)
        if compiled is None or actual.get("traveler", current) != current or (compiled.get("scope_coverage") or {}).get("step", check_step + 1) > check_step:
            check = {"status": "UNVERIFIED", "source_kind": SOURCE_KIND, "task_success": None,
                "checked_slot_count": 0, "violations": [], "diagnostics": [{"code": "CURRENT_INHERITANCE_SUBJECT_UNVERIFIED"}],
                "contract_ref": None}
        else:
            check = validate_inheritance(compiled, self._current_values(current, check_step))
        check["current_submission_present"] = current_present
        check["checked_actual_refs"] = [f"{record['decision_id']}@v{record['version']}" for record in self._current_values(current, check_step)]
        archived = {"traveler": current, "step": check_step, "action_event_id": committed["action_event_id"], "check": copy.deepcopy(check)}
        self._inheritance_checks[current] = copy.deepcopy(archived)
        self._inheritance_check_history.append(archived)
        committed["inheritance_check"] = copy.deepcopy(check)
        return committed

    def snapshot(self):
        result = super().snapshot()
        result["inheritance"] = {"contract": copy.deepcopy(self._inheritance_contract),
            "base_traveler": self._base_traveler, "configured_at": self._configured_at,
            "base_fingerprint": self._base_fingerprint, "base_decisions": copy.deepcopy(self._base_values),
            "compiled": copy.deepcopy(self._inheritance_compiled), "archive": copy.deepcopy(self._inheritance_archive),
            "checks": copy.deepcopy(self._inheritance_checks), "check_history": copy.deepcopy(self._inheritance_check_history)}
        return result


__all__ = ["InheritedFrontierMemory"]
