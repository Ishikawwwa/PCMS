"""The E0--E6 conditions, with independent and inspectable mechanism switches.

This implements memory reconstruction and deterministic invalidation. E5 does
not pretend that invalidation itself repairs an itinerary: the caller must spend
its action budget replanning the reported affected boundary.
"""
from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any

from .backends import MemoryBackend, canonical
from .grounding import decimal_value, numeric_literal_present, property_row_proof


KINDS = {"LOCAL", "INHERITED", "JOIN", "RELATION", "ENVIRONMENT", "FACTUAL"}
CONSTRAINT_STATUSES = {"ACTIVE", "SATISFIED", "VIOLATED", "SUPERSEDED", "UNRESOLVED", "INVALIDATED"}
DECISION_STATUSES = {"PROPOSED", "COMMITTED", "INVALIDATED"}
OPERATORS = {"EQUALS", "NOT_EQUALS", "LT", "LTE", "GT", "GTE", "IN", "NOT_IN", "ABS_DIFF_LTE", "INTERSECTS", "SUPERSET_OF", "DISJOINT"}
ACTIVE = {"ACTIVE", "SATISFIED", "VIOLATED", "UNRESOLVED"}
SECTION_NAMES = ("KNOWN CONTEXT", "PAST OUTCOMES", "ACTIVE CONSTRAINTS", "DEFINITIONS", "OPEN QUESTIONS")


def decision_ref(traveler: str, day: Any, slot: str, attribute: str | None = None) -> str:
    base = f"decision:{traveler}:day{day}:{slot}"
    return base if attribute in (None, "", "value") else f"{base}:{attribute}"


def _split_ref(ref: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)@v(\d+)$", ref)
    rid, version = (match.group(1), int(match.group(2))) if match else (ref, None)
    # The explicit value attribute names the submitted choice itself. Normalize
    # only that decision-reference shape; other properties retain their IDs.
    choice_alias = re.fullmatch(r"(decision:[^:]+:day[^:]+:[^:]+):value", rid)
    if choice_alias:
        rid = choice_alias.group(1)
    return rid, version


def _as_scope(record: dict, group: str, traveler: str | None = None) -> dict:
    scope = copy.deepcopy(record.get("scope", record.get("parent_scope", {})))
    if not isinstance(scope, dict):
        scope = {"traveler": str(scope)}
    scope.setdefault("group", group)
    if traveler and "traveler" not in scope:
        scope["traveler"] = traveler
    return scope


class LedgerMemory(MemoryBackend):
    def __init__(self, variant: str, config: dict | None = None):
        self.variant = variant
        level = int(variant[1:])
        options = dict(config or {})
        self.features = {
            "typed_ledger": level >= 1, "dependency_closure": level >= 2,
            "protected_compression": level >= 3, "deterministic_validation": level >= 4,
            "targeted_recovery": level >= 5, "glossary_hops": level >= 6,
        }
        # Explicit overrides identify micro-ablations in snapshots and run logs.
        for feature in self.features:
            if feature in options:
                self.features[feature] = bool(options[feature])
        super().__init__(options)

    def reset(self, group_id: str) -> None:
        super().reset(group_id)
        self.constraints: dict[str, list[dict]] = {}
        self.decisions: dict[str, list[dict]] = {}
        self.definitions: dict[str, list[dict]] = {}
        self.tasks: dict[str, dict] = {}
        self.outcomes: list[dict] = []
        self.visible_index: set[str] = set()
        self.dependencies: list[dict] = []
        self.recovery_log: list[dict] = []
        self.invalidations: dict[str, dict] = {}
        self._record_counter = Counter()

    def _error(self, code: str, record_id: str | None = None, **details: Any) -> dict:
        issue = {"code": code, **({"record_id": record_id} if record_id else {}), **details}
        # Decimal comparison diagnostics retain exact spellings while remaining
        # serializable in immutable logs, snapshots, and deterministic replay.
        import json
        issue = json.loads(canonical(issue))
        self._diagnostics.append(issue)
        self._metrics[f"diagnostic_{code.lower()}"] += 1
        return issue

    def _evidence(self, ref: str, step: int) -> tuple[dict | None, str | None]:
        if not isinstance(ref, str) or not ref.startswith("event:"):
            return None, "INVALID_EVIDENCE_REF"
        body = ref[6:]
        # Event IDs may themselves contain colons. Match the longest stored ID.
        matches = [eid for eid in self.events if body == eid or body.startswith(eid + ":span_")]
        if not matches:
            return None, "MISSING_EVIDENCE"
        event = self.events[max(matches, key=len)]
        if event["step"] > step:
            return None, "FUTURE_EVIDENCE"
        suffix = body[len(event["event_id"]):]
        if suffix:
            match = re.fullmatch(r":span_(\d+)_(\d+)", suffix)
            content = event["content"] if isinstance(event["content"], str) else canonical(event["content"])
            if not match or not 0 <= int(match[1]) < int(match[2]) <= len(content):
                return None, "INVALID_EVIDENCE_SPAN"
            event = {**event, "content": content[int(match[1]):int(match[2])]}
        return event, None

    def _validate_refs(self, record: dict, step: int, key: str = "source_refs") -> bool:
        refs = record.get(key)
        rid = record.get("constraint_id", record.get("decision_id", record.get("term_id")))
        if not isinstance(refs, list) or not refs:
            self._error("MISSING_SOURCE_REFS", rid)
            return False
        valid = True
        for ref in refs:
            _, error = self._evidence(ref, step)
            if error:
                self._error(error, rid, reference=ref)
                valid = False
        return valid

    def _trace(self, record: dict, step: int, *, decision: bool = False) -> bool:
        sources = [self._evidence(ref, step)[0] for ref in record.get("source_refs", [])]
        texts = [(e["content"] if isinstance(e["content"], str) else canonical(e["content"])) for e in sources if e]
        text = "\n".join(texts).casefold()
        action_text = "\n".join((e["content"] if isinstance(e["content"], str) else canonical(e["content"])) for e in sources if e and e["type"] == "agent_action").casefold()
        rid = record.get("decision_id" if decision else "constraint_id")
        quote = record.get("evidence_quote")
        if quote and str(quote).casefold() not in text:
            self._error("EVIDENCE_QUOTE_MISMATCH", rid)
            return False
        target = {"value": record.get("value")} if decision else record.get("target", {})
        if "value" in target:
            value = target["value"]
            atoms = value if isinstance(value, list) else [value]
            # This is a falsification check, not a semantic proof of extraction.
            for atom in atoms:
                if atom is None:
                    continue
                needle = str(atom).casefold() if isinstance(atom, (str, int, float)) else canonical(atom).casefold()
                occurs = needle in text
                if isinstance(atom, (int, float)) and not isinstance(atom, bool):
                    occurs = numeric_literal_present(atom, text)
                if not occurs:
                    self._error("EVIDENCE_MISMATCH", rid, value=atom)
                    return False
                if decision and record.get("attribute", "value") == "value" and (not action_text or needle not in action_text):
                    self._error("ACTUAL_ACTION_MISMATCH", rid, value=atom)
                    return False
        elif "ref" in target:
            parts = str(target["ref"]).split(":")
            if len(parts) > 1 and parts[1].casefold() not in text:
                self._error("EVIDENCE_MISMATCH", rid, reference=target["ref"])
                return False
        if decision and record.get("attribute", "value") != "value":
            def explicit_property(content: Any) -> bool:
                if isinstance(content, dict):
                    return any((str(key) == record["attribute"] or str(key).endswith("_" + record["attribute"])) and value == record["value"] or explicit_property(value) for key, value in content.items())
                if isinstance(content, list):
                    return any(explicit_property(value) for value in content)
                return False

            directly_submitted = any(e and e["type"] == "agent_action" and explicit_property(e["content"]) for e in sources)
            base_ref = decision_ref(record["traveler"], record["day"], record["slot"])
            base = self._resolve(base_ref, step, diagnostics=False)
            if not directly_submitted and base is None:
                self._error("PROPERTY_WITHOUT_SUBMITTED_ENTITY", rid, reference=base_ref)
                return False
            if not directly_submitted:
                contents = [e["content"] for e in sources if e and e["type"] == "tool_result"]
                proven, code = property_row_proof(base["value"], record["slot"], record["attribute"], record["value"], contents)
                if not proven:
                    self._error(code, rid, submitted_entity=base["value"], extracted_value=record["value"])
                    return False
            if base:
                record["entity_ref"] = f"{base_ref}@v{base['version']}"
                record.setdefault("depends_on", []).append(record["entity_ref"])
        record["evidence_validation"] = "REFERENCE_AND_LITERAL_CHECKED"
        record["semantic_verification"] = "UNVERIFIED"
        return True

    def _next_id(self, kind: str, traveler: str) -> str:
        self._record_counter[kind] += 1
        return f"{kind}:{self.group_id}:{traveler}:{self._record_counter[kind]}"

    def _store(self, bucket: dict[str, list[dict]], rid: str, record: dict, step: int) -> bool:
        historical = bucket.setdefault(rid, [])
        incoming = copy.deepcopy(record)
        incoming["created_at"] = step
        next_version = max((r["version"] for r in historical), default=0) + 1
        supplied = incoming.get("version", incoming.get("revision"))
        if supplied is None and historical:
            comparable = {k: v for k, v in incoming.items() if k not in {"created_at", "version", "revision"}}
            previous = {k: v for k, v in historical[-1].items() if k not in {"created_at", "version", "revision"}}
            if canonical(comparable) == canonical(previous):
                return False
        if supplied is not None and int(supplied) < next_version:
            # Re-indexing the identical extraction is idempotent.
            comparable = {k: v for k, v in incoming.items() if k not in {"created_at", "version", "revision"}}
            old = next((r for r in historical if r["version"] == int(supplied)), None)
            if old and canonical(comparable) == canonical({k: v for k, v in old.items() if k not in {"created_at", "version", "revision"}}):
                return False
            self._error("DUPLICATE_VERSION", rid, version=supplied)
            return False
        incoming["version"] = int(supplied) if supplied is not None else next_version
        if "constraint_id" in incoming:
            incoming["revision"] = incoming["version"]
        historical.append(incoming)
        self.visible_index.add(rid)
        for ref in self._deps(incoming):
            edge = {"source_id": f"{rid}@v{incoming['version']}", "target_id": ref,
                    "relation_type": "DEPENDS_ON", "step": step}
            self.dependencies.append(edge)
        return True

    def _deps(self, record: dict) -> list[str]:
        refs = list(record.get("depends_on", []))
        if isinstance(record.get("target"), dict) and record["target"].get("ref"):
            refs.append(record["target"]["ref"])
        return list(dict.fromkeys(str(ref) for ref in refs))

    def _retrieval_dependencies(self, record: dict) -> list[str]:
        """Planner-read policy; archive traversal always uses the full _deps."""
        return self._deps(record)

    def _ingest(self, extraction: dict, step: int, traveler: str, action_ref: str | None = None, allow_decisions: bool = True) -> dict:
        if not isinstance(extraction, dict):
            self._error("INVALID_EXTRACTION", message="Expected an object")
            return {"constraints": [], "decisions": [], "definitions": []}
        self._check_group(extraction)
        stored = {"constraints": [], "decisions": [], "definitions": []}
        if extraction.get("parse_success") is False:
            self._error("PARSER_FAILURE")
        for unresolved in extraction.get("unresolved", []):
            self._error("PARSER_UNRESOLVED", detail=copy.deepcopy(unresolved))
        strict = self.features["deterministic_validation"]
        for kind, bucket, id_key, ref_key in (("constraints", self.constraints, "constraint_id", "source_refs"),
                                              ("definitions", self.definitions, "term_id", "evidence_refs"),
                                              ("decisions", self.decisions, "decision_id", "source_refs")):
            if kind == "decisions" and not allow_decisions:
                if extraction.get(kind):
                    self._error("PREPLAN_DECISION_PROHIBITED")
                continue
            rows = extraction.get(kind, [])
            if not isinstance(rows, list):
                self._error("INVALID_EXTRACTION", message=f"{kind} must be a list")
                continue
            for raw in rows:
                if not isinstance(raw, dict):
                    self._error("INVALID_RECORD", record_type=kind)
                    continue
                record = copy.deepcopy(raw)
                self._check_group(record)
                if action_ref:
                    record[ref_key] = [action_ref if ref == "event:action" else ref for ref in record.get(ref_key, [])]
                if kind == "decisions":
                    if not all(key in record for key in ("traveler", "day", "slot", "value")):
                        self._error("INVALID_DECISION", message="traveler/day/slot/value required")
                        continue
                    record.setdefault("attribute", "value")
                    record.setdefault(id_key, decision_ref(record["traveler"], record["day"], record["slot"], record["attribute"]))
                    if record[id_key] != decision_ref(record["traveler"], record["day"], record["slot"], record["attribute"]):
                        self._error("INVALID_DECISION_ID", record[id_key], message="ID must encode traveler/day/slot/attribute")
                        continue
                    record.setdefault("status", "COMMITTED" if action_ref else "PROPOSED")
                    record.setdefault(ref_key, [action_ref] if action_ref else [])
                    if action_ref and action_ref not in record[ref_key]:
                        record[ref_key].append(action_ref)
                    if record["status"] not in DECISION_STATUSES:
                        self._error("INVALID_STATUS", record[id_key])
                        continue
                    record.setdefault("depends_on", [])
                    # Submitted choices depend on binding requirements for this
                    # exact slot, including attributes of the same entity.
                    for cid in self.constraints:
                        constraint = self._resolve(cid, step, diagnostics=False)
                        if constraint and self._same_slot(constraint.get("subject", {}), record) and constraint.get("status") in ACTIVE:
                            record["depends_on"].append(f"{cid}@v{constraint['version']}")
                    if record["attribute"] != "value":
                        base_ref = decision_ref(record["traveler"], record["day"], record["slot"])
                        base = self._resolve(base_ref, step, diagnostics=False)
                        if base:
                            record["entity_ref"] = f"{base_ref}@v{base['version']}"
                            record["depends_on"].append(record["entity_ref"])
                    record["depends_on"] = list(dict.fromkeys(record["depends_on"]))
                elif kind == "constraints":
                    record.setdefault(id_key, self._next_id("constraint", traveler))
                    record.setdefault("kind", "LOCAL")
                    record["scope"] = _as_scope(record, self.group_id, traveler)
                    if str(record["scope"].get("group")) != self.group_id:
                        self._error("GROUP_SCOPE_MISMATCH", record[id_key])
                        continue
                    if not all(key in record for key in ("subject", "operator", "target")) or not isinstance(record.get("target"), dict):
                        self._error("INVALID_CONSTRAINT", record[id_key])
                        continue
                    if record["kind"] not in KINDS or record["operator"] not in OPERATORS:
                        self._error("INVALID_CONSTRAINT_ENUM", record[id_key])
                        continue
                    record.setdefault("status", "ACTIVE")
                    if record["status"] not in CONSTRAINT_STATUSES:
                        self._error("INVALID_STATUS", record[id_key])
                        continue
                    if record["status"] == "SATISFIED":
                        record["status"] = "UNRESOLVED"
                        self._error("UNVERIFIED_SATISFACTION_CLAIM", record[id_key])
                    if not isinstance(record["subject"], dict):
                        self._error("INVALID_SUBJECT", record[id_key])
                        continue
                    record.setdefault("slot", record["subject"].get("slot"))
                else:
                    record.setdefault("term", record.get("term_id", ""))
                    record.setdefault(id_key, f"definition:{record['term']}")
                    record.setdefault("entity_aliases", [])
                    record.setdefault("related_terms", [])
                    record["parent_scope"] = _as_scope(record, self.group_id)
                    if str(record["parent_scope"].get("group")) != self.group_id:
                        self._error("GROUP_SCOPE_MISMATCH", record[id_key])
                        continue
                    if not record.get("term") or not isinstance(record.get("text"), str):
                        self._error("INVALID_DEFINITION", record[id_key])
                        continue
                # Evidence references are an invariant for all conditions;
                # content/consistency validation is the E4 treatment.
                if not self._validate_refs(record, step, ref_key):
                    continue
                if strict and kind != "definitions" and not self._trace(record, step, decision=kind == "decisions"):
                    continue
                if strict and kind == "constraints" and not self._validate_constraint(record, step):
                    continue
                supersedes = record.get("supersedes")
                if supersedes and kind == "constraints":
                    prior = self._resolve(supersedes, step, diagnostics=False)
                    if prior:
                        changed = {**prior, "status": "SUPERSEDED", "superseded_by": record[id_key]}
                        changed.pop("version", None)
                        changed.pop("revision", None)
                        self._store(bucket, _split_ref(supersedes)[0], changed, step)
                    elif strict:
                        self._error("MISSING_SUPERSEDED_RECORD", record[id_key], reference=supersedes)
                        continue
                if self._store(bucket, record[id_key], record, step):
                    stored[kind].append(record[id_key])
        return stored

    def _resolve(self, ref: str, step: int, diagnostics: bool = True, allow_invalid: bool = False) -> dict | None:
        rid, version = _split_ref(ref)
        bucket = self.constraints if rid in self.constraints else self.decisions if rid in self.decisions else self.definitions if rid in self.definitions else None
        if bucket is None:
            if diagnostics:
                self._error("MISSING_REFERENCE", reference=ref)
            return None
        visible = [r for r in bucket[rid] if r["created_at"] <= step]
        if not visible:
            if diagnostics:
                self._error("FUTURE_REFERENCE", reference=ref)
            return None
        latest = max(visible, key=lambda r: r["version"])
        if rid in self.decisions and not allow_invalid:
            valid = [r for r in visible if r.get("status") == "COMMITTED" and not
                     (self.invalidations.get(f"{rid}@v{r['version']}") and self.invalidations[f"{rid}@v{r['version']}"]["step"] <= step)]
            # A failed proposed branch cannot shadow an earlier accepted actual
            # choice. Explicit invalidation marks that earlier version too, so
            # recovery never resurrects an invalidated branch.
            if valid:
                latest = max(valid, key=lambda r: r["version"])
        chosen = next((r for r in visible if r["version"] == version), None) if version else latest
        if chosen is None:
            if diagnostics:
                self._error("MISSING_VERSION", reference=ref)
            return None
        if version and chosen["version"] != latest["version"] and self.features["deterministic_validation"]:
            if diagnostics:
                self._error("STALE_DEPENDENCY", reference=ref, latest_version=latest["version"])
            return None
        status = chosen.get("status")
        invalidation = self.invalidations.get(f"{rid}@v{chosen['version']}")
        invalid_as_of = invalidation and invalidation["step"] <= step
        if not allow_invalid and (status in {"INVALIDATED", "SUPERSEDED", "PROPOSED"} or invalid_as_of):
            if diagnostics:
                self._error("INVALID_REFERENCE", reference=ref, status="INVALIDATED" if invalid_as_of else status)
            return None
        if rid in self.decisions and chosen.get("attribute", "value") != "value" and chosen.get("entity_ref") and (self.features["dependency_closure"] or self.features["deterministic_validation"]):
            entity_id, entity_version = _split_ref(chosen["entity_ref"])
            current_entity = self._resolve(entity_id, step, diagnostics=False)
            old_entities = self.decisions.get(entity_id, [])
            old_entity = next((r for r in old_entities if r["version"] == entity_version and r["created_at"] <= step), None)
            if current_entity is None or old_entity is None or current_entity["value"] != old_entity["value"]:
                if diagnostics:
                    self._error("STALE_PROPERTY_EVIDENCE", rid, entity_reference=chosen["entity_ref"], current_entity_ref=f"{entity_id}@v{current_entity['version']}" if current_entity else None)
                return None
        return copy.deepcopy(chosen)

    def _resolve_binding(self, ref: str, step: int) -> dict | None:
        """Resolve live binding edges while retaining archived version pointers."""
        rid, requested_version = _split_ref(ref)
        supersession_seen = set()
        while rid in self.constraints:
            history = [r for r in self.constraints[rid] if r["created_at"] <= step]
            latest = max(history, key=lambda r: r["version"]) if history else None
            if not latest or latest.get("status") != "SUPERSEDED" or not latest.get("superseded_by"):
                break
            if rid in supersession_seen:
                self._error("SUPERSESSION_CYCLE", reference=ref)
                return None
            supersession_seen.add(rid)
            rid = _split_ref(latest["superseded_by"])[0]
        resolved = self._resolve(rid, step)
        if resolved:
            resolved_id = resolved.get("constraint_id", resolved.get("decision_id", resolved.get("term_id")))
            actual_ref = f"{resolved_id}@v{resolved['version']}"
            if rid != _split_ref(ref)[0] or requested_version is not None and requested_version != resolved["version"]:
                self._error("DEPENDENCY_REBOUND", reference=ref, resolved_reference=actual_ref,
                            reason="LATEST_VALID_CAUSALLY_VISIBLE_BINDING")
                if self.features["deterministic_validation"]:
                    self._error("STALE_DEPENDENCY", reference=ref, latest_version=resolved["version"], resolution="REBOUND")
        return resolved

    def committed_decisions(self, step: int | None = None) -> list[dict]:
        """Latest actual valid choices/facts only, for causal observed-row joins."""
        boundary = self.current_step if step is None else step
        return [record for did in sorted(self.decisions)
                if (record := self._resolve(did, boundary, diagnostics=False)) is not None]

    @staticmethod
    def _same_slot(subject: dict, decision: dict, attribute: bool = False) -> bool:
        keys = ("traveler", "day", "slot", "attribute") if attribute else ("traveler", "day", "slot")
        return all(str(subject.get(key, "value" if key == "attribute" else "")) == str(decision.get(key, "value" if key == "attribute" else "")) for key in keys)

    def _validate_constraint(self, record: dict, step: int) -> bool:
        rid = record["constraint_id"]
        target = record["target"]
        if "ref" in target:
            referenced = self._resolve(target["ref"], step, diagnostics=False)
            if referenced is None:
                self._error("UNRESOLVED_REFERENCE", rid, reference=target["ref"])
                # Missing references remain explicit unresolved requirements.
                record["status"] = "UNRESOLVED"
            else:
                left_unit = record.get("unit", target.get("unit"))
                if left_unit and referenced.get("unit") and left_unit != referenced["unit"]:
                    self._error("UNITS_MISMATCH", rid, actual=left_unit, expected=referenced["unit"])
                    record["status"] = "UNRESOLVED"
                subject = record.get("subject", {})
                # Cross-day joins can be declared explicitly; only a declared
                # same-day binding is checked for mismatch.
                if record.get("same_day") and str(subject.get("day")) != str(referenced.get("day")):
                    self._error("DAY_SLOT_MISMATCH", rid)
                    record["status"] = "UNRESOLVED"
        for cid in self.constraints:
            other = self._resolve(cid, step, diagnostics=False)
            if not other or cid == rid or cid == record.get("supersedes") or other.get("status") not in ACTIVE:
                continue
            if self._same_slot(record["subject"], other.get("subject", {}), attribute=True):
                aunit = record.get("unit", record["target"].get("unit"))
                bunit = other.get("unit", other["target"].get("unit"))
                if aunit and bunit and aunit != bunit:
                    self._error("UNITS_MISMATCH", rid, other_constraint=cid, units=[aunit, bunit])
                    record["status"] = "UNRESOLVED"
                elif self._contradict(record, other):
                    self._error("CONTRADICTORY_CONSTRAINTS", rid, other_constraint=cid)
                    record["status"] = "UNRESOLVED"
        return True

    @staticmethod
    def _contradict(first: dict, second: dict) -> bool:
        a, b = first["target"].get("value"), second["target"].get("value")
        if a is None or b is None:
            return False
        op_a, op_b = first["operator"], second["operator"]
        if op_a == op_b == "EQUALS":
            return a != b
        if op_a == "EQUALS":
            return LedgerMemory._compare(a, op_b, b) is False
        if op_b == "EQUALS":
            return LedgerMemory._compare(b, op_a, a) is False
        lower = [(v, op == "GT") for op, v in ((op_a, a), (op_b, b)) if op in {"GT", "GTE"}]
        upper = [(v, op == "LT") for op, v in ((op_a, a), (op_b, b)) if op in {"LT", "LTE"}]
        if lower and upper and isinstance(lower[0][0], (int, float)) and isinstance(upper[0][0], (int, float)):
            return lower[0][0] > upper[0][0] or (lower[0][0] == upper[0][0] and (lower[0][1] or upper[0][1]))
        return False

    @staticmethod
    def _compare(value: Any, operator: str, target: Any) -> bool | None:
        from .predicate_kernel import compare
        try:
            if operator in {"INTERSECTS", "SUPERSET_OF", "DISJOINT"}:
                def terms(item):
                    if isinstance(item, str):
                        return {part.strip().casefold() for part in re.split(r"[,;]", item) if part.strip()}
                    if isinstance(item, list) and all(isinstance(part, str) for part in item):
                        return {part.strip().casefold() for part in item if part.strip()}
                    return None
                left, right = terms(value), terms(target)
                if left is None or right is None:
                    return None
                return compare(left,operator,right)
            if operator == "ABS_DIFF_LTE":
                if not isinstance(target, dict): return None
                left, right, tolerance = decimal_value(value), decimal_value(target.get("value")), decimal_value(target.get("tolerance"))
                if left is None or right is None or tolerance is None or tolerance < 0: return None
                return compare(left,operator,{"value":right,"tolerance":tolerance})
            left_number, right_number = decimal_value(value), decimal_value(target)
            if left_number is not None and right_number is not None and operator in {"EQUALS", "NOT_EQUALS", "LT", "LTE", "GT", "GTE"}:
                value, target = left_number, right_number
            return compare(value,operator,target)
        except (TypeError, ValueError):
            return None
        return None

    def _named_travelers(self, query: str) -> set[str]:
        names = {r["traveler"] for versions in self.decisions.values() for r in versions}
        return {name for name in names if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", query, re.I)}

    def _seed_ids(self, request: dict, step: int) -> list[str]:
        traveler = str(request.get("traveler", ""))
        query = str(request.get("query", request.get("request", "")))
        seeds = list(request.get("dependency_refs", []))
        slots = set(request.get("slots", []))
        if request.get("slot"):
            slots.add(request["slot"])
        for cid in sorted(self.constraints):
            if cid not in self.visible_index:
                continue
            record = self._resolve(cid, step, diagnostics=False)
            if not record or record.get("status") not in ACTIVE:
                continue
            scope_name = record.get("scope", {}).get("traveler")
            subject = record.get("subject", {})
            applicable = scope_name in (None, "", "*", traveler) or subject.get("traveler") == traveler
            if applicable and (not slots or not subject.get("slot") or subject.get("slot") in slots):
                seeds.append(cid)
        # A literal stable reference always wins over lexical inference.
        seeds += re.findall(r"(?:decision|constraint|definition):[\w:.-]+(?:@v\d+)?", query)
        covered_names = set()
        for ref in seeds:
            if str(ref).startswith("decision:"):
                covered_names.add(str(ref).split(":")[1])
            record = self._resolve(ref, step, diagnostics=False)
            if record:
                for dependency in self._deps(record):
                    if dependency.startswith("decision:"):
                        covered_names.add(dependency.split(":")[1])
        for name in self._named_travelers(query):
            if name in request.get("public_anchor_travelers", []):
                # These exact plans already appear in the shared fixed prompt.
                # Explicit typed dependencies still retrieve their needed slots.
                continue
            if name in covered_names and self.features["dependency_closure"]:
                continue
            candidates = []
            for did in sorted(self.decisions):
                record = self._resolve(did, step, diagnostics=False)
                if not record or record["traveler"] != name or did not in self.visible_index:
                    continue
                if slots and record["slot"] not in slots:
                    continue
                candidates.append(did)
            seeds.extend(candidates)
        return list(dict.fromkeys(seeds))

    def _closure(self, seeds: list[str], step: int, *, retrieval: bool = False) -> tuple[list[dict], list[str]]:
        records: list[dict] = []
        seen: set[str] = set()
        visiting: set[str] = set()
        missing: list[str] = []

        def visit(ref: str, explicit_seed: bool = False) -> str | None:
            if ref in visiting:
                self._error("DEPENDENCY_CYCLE", reference=ref)
                missing.append(ref)
                return
            record = self._resolve(ref, step) if explicit_seed and _split_ref(ref)[1] is not None else self._resolve_binding(ref, step)
            if record is None:
                missing.append(ref)
                # An unknown property is not an unknown submitted entity. Keep
                # the actual entity available so the planner can query it, while
                # leaving the property's value explicitly unresolved.
                rid, version = _split_ref(ref)
                parts = rid.split(":")
                if len(parts) == 5 and parts[0] == "decision" and not (explicit_seed and version is not None):
                    entity_ref = ":".join(parts[:4])
                    visiting.add(ref)
                    anchor = visit(entity_ref)
                    visiting.discard(ref)
                    if anchor:
                        self._error("PROPERTY_EVIDENCE_REQUIRED", reference=ref, submitted_entity_ref=anchor)
                return
            rid = record.get("constraint_id", record.get("decision_id", record.get("term_id")))
            versioned = f"{rid}@v{record['version']}"
            if versioned in visiting:
                self._error("DEPENDENCY_CYCLE", reference=versioned)
                missing.append(versioned)
                return
            if versioned in seen:
                return versioned
            visiting.add(ref)
            visiting.add(versioned)
            resolved_dependencies = {}
            dependencies = self._retrieval_dependencies(record) if retrieval else self._deps(record)
            for dependency in dependencies:
                resolved_dependency = visit(dependency)
                if resolved_dependency:
                    resolved_dependencies[dependency] = resolved_dependency
            visiting.discard(ref)
            visiting.discard(versioned)
            seen.add(versioned)
            record["record_ref"] = versioned
            if resolved_dependencies:
                record["resolved_dependencies"] = resolved_dependencies
                target_ref = record.get("target", {}).get("ref")
                if target_ref in resolved_dependencies:
                    record["resolved_target_ref"] = resolved_dependencies[target_ref]
            records.append(record)
            return versioned

        for seed in seeds:
            visit(seed, explicit_seed=True)
        return records, list(dict.fromkeys(missing))

    def _glossary(self, query: str, step: int, traveler: str) -> tuple[list[dict], list[str]]:
        definitions = []
        for tid in sorted(self.definitions):
            if tid not in self.visible_index:
                continue
            record = self._resolve(tid, step, diagnostics=False)
            if record and record.get("parent_scope", {}).get("traveler") in (None, "", "*", traveler):
                definitions.append(record)
        lookup: dict[str, list[dict]] = {}
        for record in definitions:
            for term in [record["term"], record["term_id"], *record["entity_aliases"]]:
                lookup.setdefault(str(term).casefold(), []).append(record)
        roots = [term for term in lookup if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", query, re.I)]
        pending = [(term, 0, ()) for term in roots]
        result = []
        seen = set()
        unresolved = []
        hops = int(self.config.get("glossary_max_hops", self.config.get("max_glossary_hops", 4))) if self.features["glossary_hops"] else 1
        cap = int(self.config.get("glossary_step_limit", 16))
        steps = 0
        while pending and steps < cap:
            term, depth, chain = pending.pop(0)
            if term in chain:
                self._error("GLOSSARY_CYCLE", term=term, chain=list(chain))
                unresolved.append(term)
                continue
            candidates = lookup.get(term, [])
            if len(candidates) > 1:
                self._error("AMBIGUOUS_ALIAS", term=term, candidates=[r["term_id"] for r in candidates])
                unresolved.append(term)
                continue
            if not candidates:
                self._error("UNDEFINED_TERM", term=term)
                unresolved.append(term)
                continue
            record = candidates[0]
            if record["term_id"] in seen:
                continue
            seen.add(record["term_id"])
            result.append(record)
            steps += 1
            # E0--E5 retrieve literal definitions in one pass. E6 alone follows
            # explicit links and names found in definition text.
            related = list(record.get("related_terms", []))
            if self.features["glossary_hops"]:
                related += [key for key in lookup if key not in {record["term"].casefold(), record["term_id"].casefold()} and
                            re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", record["text"], re.I)]
            if depth + 1 < hops:
                parent_chain = (*chain, term)
                pending.extend((child, depth + 1, parent_chain) for child in dict.fromkeys(str(item).casefold() for item in related))
            elif related and self.features["glossary_hops"]:
                self._error("GLOSSARY_HOP_LIMIT", term=record["term"], unresolved=related)
                unresolved.extend(related)
        if pending:
            self._error("GLOSSARY_STEP_LIMIT", unresolved=[term for term, _, _ in pending])
            unresolved.extend(term for term, _, _ in pending)
        self._metrics["glossary_steps"] += steps
        return result, list(dict.fromkeys(unresolved))

    def _sections(self, records: list[dict], definitions: list[dict], outcomes: list[dict], compact: bool = False) -> dict[str, list[Any]]:
        narrative_keys = {"evidence_quote", "narrative", "explanation", "reasoning", "rationale", "description"}
        constraints = [{k: v for k, v in r.items() if k not in narrative_keys} for r in records if "constraint_id" in r]
        decisions = [{k: v for k, v in r.items() if k not in narrative_keys} for r in records if "decision_id" in r]
        if not self.features["typed_ledger"]:
            constraints = [{"requirement": {k: r[k] for k in ("kind", "subject", "operator", "target", "source_refs", "status") if k in r}} for r in constraints]
            decisions = [{k: r[k] for k in ("traveler", "day", "slot", "attribute", "value", "unit", "source_refs") if k in r} for r in decisions]
        elif compact:
            # record_ref already carries identity and version. Decision IDs
            # encode traveler/day/slot/property, while constraints retain their
            # explicit subject. Do not spend the common budget on duplicate
            # audit metadata; all binding and evidence fields stay exact.
            protected = {"record_ref", "kind", "subject", "operator", "target", "source_refs", "status", "value", "unit", "depends_on", "entity_ref", "resolved_dependencies", "resolved_target_ref"}
            constraints = [{k: v for k, v in r.items() if k in protected} for r in constraints]
            decisions = [{k: v for k, v in r.items() if k in protected} for r in decisions]
        return {"KNOWN CONTEXT": decisions,
                "PAST OUTCOMES": [{"outcome_id": o["outcome_id"], "traveler": o["traveler"], "actual_action": o["actual_action"], "status": o["status"], "committed_decision_refs": o["committed_decision_refs"], "observation_refs": o["observation_refs"]} for o in outcomes],
                "ACTIVE CONSTRAINTS": constraints,
                "DEFINITIONS": [{k: r[k] for k in ("term_id", "term", "text", "entity_aliases", "related_terms", "evidence_refs", "version", "parent_scope") if k in r} for r in definitions],
                "OPEN QUESTIONS": copy.deepcopy(self._diagnostics)}

    @staticmethod
    def _serialize(sections: dict[str, list[Any]]) -> str:
        return "\n\n".join(name + "\n" + ("\n".join(canonical(row) for row in sections[name]) or "(none)") for name in SECTION_NAMES)

    def _batches(self, sections: dict[str, list[Any]]) -> list[dict]:
        batches = []
        current = {name: [] for name in SECTION_NAMES}
        for name in SECTION_NAMES:
            for row in sections[name]:
                trial = copy.deepcopy(current)
                trial[name].append(row)
                if self._token_count(self._serialize(trial)) > self.budget and any(current.values()):
                    text = self._serialize(current)
                    batches.append({"batch_index": len(batches), "context": text, "token_estimate": self._token_count(text), "status": "OK"})
                    current = {key: [] for key in SECTION_NAMES}
                    trial = copy.deepcopy(current)
                    trial[name].append(row)
                if self._token_count(self._serialize(trial)) > self.budget:
                    text = self._serialize(trial)
                    batches.append({"batch_index": len(batches), "context": text, "token_estimate": self._token_count(text), "status": "RECORD_EXCEEDS_BUDGET"})
                    current = {key: [] for key in SECTION_NAMES}
                else:
                    current = trial
        if any(current.values()):
            text = self._serialize(current)
            batches.append({"batch_index": len(batches), "context": text, "token_estimate": self._token_count(text), "status": "OK"})
        return batches

    def begin_subtask(self, request: dict) -> dict:
        step = self._prepare(request)
        traveler = str(request.get("traveler", ""))
        tid = f"task:{self.group_id}:{traveler}"
        self.tasks.setdefault(tid, {"task_id": tid, "parent_id": f"group:{self.group_id}", "status": "ACTIVE", "created_at": step, "scope": {"group": self.group_id, "traveler": traveler}})
        if request.get("extraction"):
            self._ingest(request["extraction"], step, traveler, allow_decisions=False)
        seeds = self._seed_ids(request, step)
        if self.features["dependency_closure"]:
            records, missing = self._closure(seeds, step, retrieval=True)
        else:
            records = [record for ref in seeds if (record := self._resolve(ref, step))]
            missing = [issue["reference"] for issue in self._diagnostics if issue["code"] in {"MISSING_REFERENCE", "INVALID_REFERENCE", "FUTURE_REFERENCE"}]
            for record in records:
                rid = record.get("constraint_id", record.get("decision_id", record.get("term_id")))
                record["record_ref"] = f"{rid}@v{record['version']}"
        query = str(request.get("query", request.get("request", "")))
        definitions, unresolved_terms = self._glossary(query + " " + canonical(records), step, traveler)
        named = self._named_travelers(query)
        needed_decisions = {r["decision_id"] for r in records if "decision_id" in r}
        outcomes = [o for o in self.outcomes if o["created_at"] <= step and o["status"] not in {"FAILED", "INVALIDATED"} and
                    o["traveler"] not in request.get("public_anchor_travelers", []) and
                    (o["traveler"] in named or any(_split_ref(ref)[0] in needed_decisions for ref in o["committed_decision_refs"])) and
                    o["outcome_id"] in self.visible_index and
                    all(self._resolve(ref, step, diagnostics=False) is not None for ref in o["committed_decision_refs"])]
        recovery = self.pending_recoveries(step)
        for item in recovery:
            self._error("RECOVERY_REQUIRED", recovery_id=item["recovery_id"], trigger=item["trigger"],
                        affected_decisions=item["affected_decisions"], earliest_affected_step=item["earliest_affected_step"],
                        trigger_diagnostics=item.get("trigger_diagnostics", []),
                        remaining_revision_action_budget=max(0, item["action_budget"] - item["revision_action_cost"]) if item["action_budget"] is not None else None)
        required = [r["record_ref"] for r in records]
        sections = self._sections(records, definitions, outcomes)
        full = self._serialize(sections)
        status = "UNRESOLVED" if missing or unresolved_terms or any(issue["code"] in {"PARSER_UNRESOLVED", "PARSER_FAILURE"} for issue in self._diagnostics) else "OK"
        validation_errors = {"EVIDENCE_MISMATCH", "EVIDENCE_QUOTE_MISMATCH", "ACTUAL_ACTION_MISMATCH", "CONTRADICTORY_CONSTRAINTS", "UNITS_MISMATCH", "DAY_SLOT_MISMATCH", "INVALID_CONSTRAINT", "INVALID_STATUS", "INVALID_CONSTRAINT_ENUM", "GROUP_SCOPE_MISMATCH", "DUPLICATE_VERSION", "FUTURE_EVIDENCE", "INVALID_EVIDENCE_SPAN", "MISSING_EVIDENCE", "MISSING_SOURCE_REFS"}
        if self.features["deterministic_validation"] and any(issue["code"] in validation_errors for issue in self._diagnostics):
            status = "VALIDATION_FAILED"
        batches = []
        omitted = []
        if any(any(key in record for key in ("evidence_quote", "narrative", "explanation", "reasoning", "rationale", "description")) for record in records):
            self._metrics["compression_events"] += 1
            self._metrics["narrative_compactions"] += 1
        if self._token_count(full) > self.budget:
            self._metrics["compression_events"] += 1
            sections = self._sections(records, definitions, [], compact=True)
            full = self._serialize(sections)
            if self._token_count(full) > self.budget and self.features["protected_compression"]:
                status = "BUDGET_OVERFLOW"
                self._error("BUDGET_OVERFLOW", required_token_estimate=self._token_count(full), token_budget=self.budget, required_ids=required)
                sections["OPEN QUESTIONS"] = copy.deepcopy(self._diagnostics)
                batches = self._batches(sections)
                # A compact diagnostic is the actual planner context. Every
                # required record is separately available in labeled batches.
                full = "BUDGET_OVERFLOW\n" + canonical({"required_record_count": len(required), "required_token_estimate": self._token_count(self._serialize(sections)), "batches": len(batches)})
                if self._token_count(full) > self.budget:
                    full = self._bounded_notice("BUDGET_OVERFLOW")
                self._metrics["budget_overflows"] += 1
            elif self._token_count(full) > self.budget:
                # The original policy may lose retrieval-visible records. The
                # audit archive and explicit recall diagnostics stay intact.
                while records and self._token_count(full) > self.budget:
                    removed = records.pop(0)
                    omitted.append(removed["record_ref"])
                    if self.config.get("destructive_removal", self.config.get("compression_mode", "merge") == "drop"):
                        self.visible_index.discard(_split_ref(removed["record_ref"])[0])
                    sections = self._sections(records, definitions, [], compact=True)
                    full = self._serialize(sections)
                while definitions and self._token_count(full) > self.budget:
                    removed = definitions.pop(0)
                    omitted.append(f"{removed['term_id']}@v{removed['version']}")
                    sections = self._sections(records, definitions, [], compact=True)
                    full = self._serialize(sections)
                status = "PARTIAL_CONTEXT"
                self._error("LOSSY_COMPRESSION", omitted_ids=omitted, required_ids=required)
                self._metrics["compression_lost_records"] += len(omitted)
                # Very small budgets may not even fit headings. Report this,
                # never silently overrun the configured context envelope.
                if self._token_count(full) > self.budget:
                    full = self._bounded_notice("PARTIAL_CONTEXT")
        returned = [r["record_ref"] for r in records] if status != "BUDGET_OVERFLOW" else []
        if recovery and status == "OK":
            status = "RECOVERY_REQUIRED"
        return self._response(full, status=status, closure_ids=returned, required_closure_ids=required,
                              missing_refs=missing, glossary_ids=[r["term_id"] for r in definitions],
                              unresolved_terms=unresolved_terms, batches=batches, omitted_ids=omitted,
                              records=copy.deepcopy(records) if status != "BUDGET_OVERFLOW" else [],
                              coverage={"required": len(required), "returned": len(returned),
                                        "recall": len(returned) / len(required) if required else 1.0},
                              recovery=copy.deepcopy(recovery), features=dict(self.features))

    def _check_submission(self, decisions: list[str], step: int, scope_ids: set[str] | None = None,
                          submitted_travelers: set[str] | None = None) -> list[str]:
        violations = []
        for cid in self.constraints:
            constraint = self._resolve(cid, step, diagnostics=False)
            if not constraint or constraint.get("status") not in ACTIVE:
                continue
            subject = constraint["subject"]
            did = decision_ref(subject.get("traveler", ""), subject.get("day", ""), subject.get("slot", ""), subject.get("attribute"))
            if scope_ids is not None and did not in scope_ids:
                continue
            decision = self._resolve(did, step, diagnostics=False)
            if not decision:
                relevant = did in decisions or did in (scope_ids or set()) or subject.get("traveler") in (submitted_travelers or {self._request.get("traveler")})
                if relevant:
                    if subject.get("attribute", "value") in (None, "value"):
                        self._error("MISSING_SUBMITTED_SLOT", cid, reference=did)
                        violations.append(did)
                    else:
                        self._error("UNOBSERVED_SUBMITTED_PROPERTY", cid, reference=did)
                continue
            target = constraint["target"]
            expected = target.get("value")
            if "ref" in target:
                prior = self._resolve_binding(target["ref"], step)
                if not prior:
                    self._error("UNRESOLVED_REFERENCE", cid, reference=target["ref"])
                    continue
                expected = prior["value"]
                if target.get("scale", 1) != 1 or target.get("offset", 0) != 0:
                    number, scale, offset = decimal_value(expected), decimal_value(target.get("scale", 1)), decimal_value(target.get("offset", 0))
                    if number is None or scale is None or offset is None:
                        self._error("UNVERIFIABLE_RELATION", cid)
                        continue
                    expected = number * scale + offset
                unit = constraint.get("unit", target.get("unit"))
                if unit and prior.get("unit") and unit != prior["unit"]:
                    self._error("UNITS_MISMATCH", cid)
                    continue
            unit = constraint.get("unit", target.get("unit"))
            if unit and decision.get("unit") and unit != decision["unit"]:
                self._error("UNITS_MISMATCH", cid)
                continue
            compared_target = {"value": expected, "tolerance": target.get("tolerance")} if constraint["operator"] == "ABS_DIFF_LTE" else expected
            result = self._compare(decision["value"], constraint["operator"], compared_target)
            if result is False:
                self._error("SUBMISSION_VIOLATION", cid, decision_id=did, actual=decision["value"], expected=expected)
                violations.append(did)
            elif result is None:
                self._error("UNVERIFIABLE_CONSTRAINT", cid)
            # Do not rewrite archived constraint status. Verification results
            # are separate observations, with exact record versions.
            if result is not None:
                self._metrics["verified_constraints"] += 1
        return list(dict.fromkeys(violations))

    def _invalidate_record(self, did: str, step: int, trigger: str, action_ref: str | None = None, record: dict | None = None) -> str | None:
        record = copy.deepcopy(record) if record else self._resolve(did, step, diagnostics=False)
        if not record:
            return None
        ref = f"{did}@v{record['version']}"
        # A tombstone retires every earlier accepted version of this choice.
        # A merely failed proposed branch uses no tombstone and can still fall
        # back to the earlier accepted version.
        for historical in self.decisions[did]:
            if historical["created_at"] <= step and historical.get("status") == "COMMITTED":
                self.invalidations[f"{did}@v{historical['version']}"] = {"step": step, "trigger": trigger,
                                                                         "source_ref": action_ref}
        invalid = {**record, "status": "INVALIDATED", "invalidated_by": trigger}
        if action_ref:
            invalid["invalidation_evidence_refs"] = [action_ref]
        invalid.pop("version", None)
        self._store(self.decisions, did, invalid, step)
        return ref

    def _replace_travelers(self, travelers: list[str], decisions: list[dict], step: int, action_ref: str) -> list[str]:
        """Retire omitted slots and facts attached to changed actual entities."""
        incoming_values = {decision_ref(r["traveler"], r["day"], r["slot"]): r["value"] for r in decisions
                           if isinstance(r, dict) and all(key in r for key in ("traveler", "day", "slot", "value")) and r.get("attribute", "value") == "value"}
        before = {did: self._resolve(did, step, diagnostics=False) for did in self.decisions}
        retired = []
        for did, record in before.items():
            if record is None or record["traveler"] not in travelers:
                continue
            base_id = decision_ref(record["traveler"], record["day"], record["slot"])
            base = before.get(base_id)
            omitted = base_id not in incoming_values
            changed_entity = record.get("attribute", "value") != "value" and (base is None or incoming_values.get(base_id) != base["value"])
            if omitted or changed_entity:
                ref = self._invalidate_record(did, step, "FULL_PLAN_REPLACEMENT" if omitted else "SELECTED_ENTITY_REPLACED", action_ref, record)
                if ref:
                    retired.append(ref)
        return retired

    def _invalidate(self, seeds: list[str], step: int, trigger: str, action_budget: int | None = None) -> dict:
        trigger_diagnostics = copy.deepcopy(self._diagnostics)
        affected = set(seeds)
        changed = True
        while changed:
            changed = False
            for did, versions in self.decisions.items():
                record = self._resolve(did, step, diagnostics=False)
                if record and did not in affected:
                    refs = self._deps(record)
                    # Decision -> constraint -> prior decision is a common
                    # graph shape. Traverse it rather than only direct edges.
                    closure, _ = self._closure(refs, step)
                    if any(r.get("decision_id") in affected for r in closure):
                        affected.add(did)
                        changed = True
        historical = []
        earliest = None
        before = {did: self._resolve(did, step, diagnostics=False) for did in affected}
        for did in sorted(affected):
            record = before[did]
            if not record:
                continue
            ref = f"{did}@v{record['version']}"
            historical.append(ref)
            earliest = min(earliest if earliest is not None else record["created_at"], record["created_at"])
            self._invalidate_record(did, step, trigger, record=record)
        recovery = {"recovery_id": f"recovery:{self.group_id}:{len(self.recovery_log)+1}",
                    "step": step, "trigger": trigger, "seed_decisions": list(seeds),
                    "affected_decisions": sorted(affected), "invalidated_version_refs": historical,
                    "earliest_affected_step": earliest, "status": "REPLAN_REQUIRED",
                    "action_budget": action_budget, "tool_call_cost": 0,
                    "required_after_versions": { _split_ref(ref)[0]: _split_ref(ref)[1] for ref in historical },
                    "replanning_performed": False, "success": None,
                    "revision_action_cost": 0, "acknowledgments": [], "acknowledged_decision_refs": []}
        # Omitted required slots can have no earlier decision version. Their
        # causal constraint still identifies a real, visible repair subject.
        recovery["affected_subjects"] = {}
        for did in sorted(affected):
            record = before.get(did)
            if record:
                recovery["affected_subjects"][did] = {"traveler": record["traveler"], "day": record["day"], "slot": record["slot"]}
            else:
                for cid in self.constraints:
                    constraint = self._resolve(cid, step, diagnostics=False)
                    subject = constraint.get("subject", {}) if constraint else {}
                    if subject and decision_ref(subject.get("traveler", ""), subject.get("day", ""), subject.get("slot", ""), subject.get("attribute")) == did:
                        recovery["affected_subjects"][did] = copy.deepcopy(subject)
                        break
        recovery["trigger_diagnostics"] = trigger_diagnostics
        self.recovery_log.append(recovery)
        self._metrics["recovery_triggers"] += 1
        self._metrics["invalidated_decisions"] += len(historical)
        self._error("RECOVERY_REQUIRED", affected_decisions=sorted(affected), earliest_affected_step=earliest)
        return recovery

    def pending_recoveries(self, step: int | None = None) -> list[dict]:
        """Return independent copies; callers cannot silently acknowledge work."""
        return copy.deepcopy([r for r in self.recovery_log if r["status"] == "REPLAN_REQUIRED" and (step is None or r["step"] <= step)])

    def required_repair_travelers(self, recovery_id: str | None = None) -> list[str]:
        rows = self.pending_recoveries()
        if recovery_id is not None:
            rows = [r for r in rows if r["recovery_id"] == recovery_id]
        priorities = {}
        for recovery in rows:
            acknowledged=set(recovery.get("acknowledged_decision_refs",[]))
            for ref in recovery["invalidated_version_refs"]:
                did, version = _split_ref(ref)
                current=self._resolve(did,self.current_step,diagnostics=False)
                if current and f"{did}@v{current['version']}" in acknowledged:
                    continue
                record = next((r for r in self.decisions.get(did, []) if r["version"] == version), None)
                if record:
                    priorities[record["traveler"]] = min(priorities.get(record["traveler"], record["created_at"]), record["created_at"])
            for did, subject in recovery.get("affected_subjects", {}).items():
                current = self._resolve(did, self.current_step, diagnostics=False)
                if current and f"{did}@v{current['version']}" in acknowledged:
                    continue
                name = subject.get("traveler")
                if name:
                    priorities[name] = min(priorities.get(name, recovery["step"]), recovery["step"])
        return sorted(priorities, key=lambda name: (priorities[name], name))

    def acknowledge_recovery(self, recovery_id: str, acknowledgment: dict) -> dict:
        """Acknowledge actual repaired commits; this method never writes choices.

        Partial acknowledgments accumulate exact action/decision evidence. Only
        a complete current checked boundary clears pending recovery. Task
        success remains unknown unless an external completion signal is given.
        """
        self._check_group(acknowledgment)
        self._diagnostics = []
        recovery = next((r for r in self.recovery_log if r["recovery_id"] == recovery_id), None)
        if recovery is None:
            self._error("UNKNOWN_RECOVERY", reference=recovery_id)
            return {"status": "VALIDATION_FAILED", "diagnostics": copy.deepcopy(self._diagnostics)}
        if not self.features["targeted_recovery"]:
            self._error("REVISION_DISABLED")
            return {"status": "VALIDATION_FAILED", "diagnostics": copy.deepcopy(self._diagnostics)}
        step = int(acknowledgment.get("step", self.current_step))
        if step < recovery["step"]:
            self._error("CAUSAL_RECOVERY_ACK_MISMATCH", reference=recovery_id)
            return {"status": "VALIDATION_FAILED", "diagnostics": copy.deepcopy(self._diagnostics)}
        action_ids = acknowledgment.get("repaired_action_event_ids", [])
        refs = acknowledgment.get("repaired_decision_refs", [])
        cost = acknowledgment.get("revision_action_cost", 0)
        if not isinstance(action_ids, list) or not all(isinstance(eid, str) for eid in action_ids) or not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs) or not isinstance(cost, int) or isinstance(cost, bool) or cost < 0:
            self._error("INVALID_RECOVERY_ACK")
            return {"status": "VALIDATION_FAILED", "diagnostics": copy.deepcopy(self._diagnostics)}
        ack_id = acknowledgment.get("acknowledgment_id", canonical({"actions": sorted(action_ids), "decisions": sorted(refs), "cost": cost}))
        existing = next((a for a in recovery["acknowledgments"] if a["acknowledgment_id"] == ack_id), None)
        if existing is None:
            accepted = []
            valid_action_refs = set()
            for eid in action_ids:
                event, error = self._evidence(eid if eid.startswith("event:") else f"event:{eid}", step)
                if error or event is None or event["type"] != "agent_action" or event["step"] < recovery["step"]:
                    self._error(error or "INVALID_REPAIR_ACTION", reference=eid)
                else:
                    valid_action_refs.add(f"event:{event['event_id']}")
            for ref in refs:
                did, version = _split_ref(ref)
                if version is None or did not in recovery["affected_decisions"]:
                    self._error("OUT_OF_SCOPE_REPAIR_REFERENCE", reference=ref)
                    continue
                record = self._resolve(ref, step)
                if record is None:
                    continue
                if record["version"] <= recovery["required_after_versions"].get(did, 0) or record["created_at"] < recovery["step"] or not valid_action_refs.intersection(record["source_refs"]):
                    self._error("REPAIR_NOT_ACTUALLY_RECOMMITTED", reference=ref)
                    continue
                if not self._validate_refs(record, step) or not self._trace(record, step, decision=True):
                    continue
                accepted.append(ref)
            recovery["acknowledgments"].append({"acknowledgment_id": ack_id, "step": step,
                                                "action_event_ids": list(action_ids), "accepted_decision_refs": accepted,
                                                "revision_action_cost": cost, "diagnostics": copy.deepcopy(self._diagnostics)})
            recovery["revision_action_cost"] += cost
            recovery["acknowledged_decision_refs"] = list(dict.fromkeys([*recovery["acknowledged_decision_refs"], *accepted]))
            self._metrics["recovery_acknowledgments"] += 1
        acknowledged = set(recovery["acknowledged_decision_refs"])
        missing = []
        current_refs = []
        for did in recovery["affected_decisions"]:
            record = self._resolve(did, step, diagnostics=False)
            current_ref = f"{did}@v{record['version']}" if record else None
            if current_ref not in acknowledged:
                missing.append(did)
            else:
                current_refs.append(current_ref)
        if not missing:
            closure, unresolved = self._closure(recovery["affected_decisions"], step)
            self._check_submission(recovery["affected_decisions"], step, scope_ids=set(recovery["affected_decisions"]))
            if unresolved or any(issue["code"] in {"SUBMISSION_VIOLATION", "MISSING_SUBMITTED_SLOT", "UNRESOLVED_REFERENCE", "UNVERIFIABLE_CONSTRAINT", "UNVERIFIABLE_RELATION", "UNITS_MISMATCH"} for issue in self._diagnostics):
                self._error("REPAIRED_BOUNDARY_UNVERIFIED", recovery_id=recovery_id)
                missing = list(recovery["affected_decisions"])
        budget = recovery["action_budget"]
        over_budget = budget is not None and recovery["revision_action_cost"] > int(budget)
        if over_budget:
            self._error("REVISION_BUDGET_EXCEEDED", action_budget=budget, actual_action_cost=recovery["revision_action_cost"])
        if not missing and not over_budget:
            recovery["status"] = "REPLANNED_UNVERIFIED"
            recovery["replanning_performed"] = True
            recovery["repaired_decision_refs"] = current_refs
            recovery["acknowledged_at"] = step
            recovery["success"] = True if acknowledgment.get("completion_verified") is True else None
        elif missing:
            self._error("RECOVERY_INCOMPLETE", missing_decisions=missing)
        return {"status": "REVISION_BUDGET_EXCEEDED" if over_budget else "RECOVERY_REQUIRED" if missing else "REPLANNED_UNVERIFIED",
                "recovery_id": recovery_id, "missing_decision_ids": missing, "recovery": copy.deepcopy(recovery),
                "diagnostics": copy.deepcopy(self._diagnostics)}

    def commit_subtask(self, result: dict) -> dict:
        self._diagnostics = []
        eid = self._record_action(result)
        step = self.events[eid]["step"]
        traveler = str(result.get("traveler", self._request.get("traveler", "")))
        extraction = copy.deepcopy(result.get("extraction", {}))
        # Deterministically parsed actual itinerary records can be passed apart
        # from model extraction, and take precedence over extracted decisions.
        if result.get("decisions") is not None:
            extraction["decisions"] = copy.deepcopy(result["decisions"])
        if result.get("branch_status") == "FAILED":
            for record in extraction.get("decisions", []):
                record["status"] = "INVALIDATED"
        replaced = self._replace_travelers(list(result.get("replace_travelers", [])), extraction.get("decisions", []), step, f"event:{eid}")
        stored = self._ingest(extraction, step, traveler, f"event:{eid}")
        action = copy.deepcopy(result.get("action", result.get("submission", result.get("answer", ""))))
        committed = []
        for did in stored["decisions"]:
            record = self._resolve(did, step, diagnostics=False)
            if record:
                committed.append(f"{did}@v{record['version']}")
        oid = self._next_id("outcome", traveler)
        status = "FAILED" if result.get("status") in {"FAILED", "failure", "failed"} else "COMPLETED" if result.get("completion_verified") is True else "UNCERTAIN"
        outcome = {"outcome_id": oid, "task_id": f"task:{self.group_id}:{traveler}", "traveler": traveler,
                   "committed_decision_refs": committed, "observation_refs": [f"event:{eid}"],
                   "status": status, "actual_action": action, "created_at": step}
        self.outcomes.append(outcome)
        self.visible_index.add(oid)
        if outcome["task_id"] in self.tasks:
            self.tasks[outcome["task_id"]]["status"] = status
        submitted_travelers = {traveler, *result.get("replace_travelers", []), *(r.get("traveler") for r in extraction.get("decisions", []) if isinstance(r, dict))}
        violations = self._check_submission(stored["decisions"], step, submitted_travelers=submitted_travelers) if self.features["deterministic_validation"] else []
        declared = self._request.get("declared_requirements", [])
        if self.features["deterministic_validation"]:
            for requirement in declared:
                rid = requirement if isinstance(requirement, str) else requirement.get("constraint_id")
                if rid and not self._resolve(rid, step, diagnostics=False):
                    self._error("UNRECORDED_REQUIREMENT", reference=rid)
        seeds = list(result.get("invalidate_decision_refs", []))
        if violations:
            seeds.extend(violations)
        recoveries = []
        if seeds and self.features["targeted_recovery"]:
            recoveries.append(self._invalidate(list(dict.fromkeys(_split_ref(ref)[0] for ref in seeds)), step,
                                               result.get("recovery_trigger", "validator_failure"), result.get("action_budget", result.get("remaining_action_budget"))))
        elif seeds:
            self._error("REVISION_DISABLED", affected_decisions=seeds)
        errors = [issue for issue in self._diagnostics if issue["code"] not in {"UNVERIFIED_SATISFACTION_CLAIM"}]
        return {"status": "RECOVERY_REQUIRED" if recoveries else "VALIDATION_FAILED" if errors else "OK",
                "action_event_id": eid, "stored_ids": stored, "outcome_id": oid,
                "diagnostics": copy.deepcopy(self._diagnostics), "recovery": copy.deepcopy(recoveries), "replaced_version_refs": replaced}

    def snapshot(self) -> dict:
        result = super().snapshot()
        result.update({"features": dict(self.features), "tasks": copy.deepcopy(list(self.tasks.values())),
                       "constraints": copy.deepcopy([r for versions in self.constraints.values() for r in versions]),
                       "decisions": copy.deepcopy([r for versions in self.decisions.values() for r in versions]),
                       "definitions": copy.deepcopy([r for versions in self.definitions.values() for r in versions]),
                       "outcomes": copy.deepcopy(self.outcomes), "dependencies": copy.deepcopy(self.dependencies),
                       "invalidations": copy.deepcopy(self.invalidations), "recovery_log": copy.deepcopy(self.recovery_log),
                       "retrieval_index": sorted(self.visible_index)})
        return result

    def stats(self) -> dict:
        result = super().stats()
        result.update({"features": dict(self.features), "constraint_versions": sum(map(len, self.constraints.values())),
                       "decision_versions": sum(map(len, self.decisions.values())),
                       "definition_versions": sum(map(len, self.definitions.values())),
                       "archived_evidence_events": len(self.events), "dependency_edges": len(self.dependencies),
                       "retrieval_index_records": len(self.visible_index)})
        return result
