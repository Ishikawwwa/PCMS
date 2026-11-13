"""Explicit declared inheritance contracts, separate from inferred requests.

The public base is known input, never an expected target itinerary. This module
does nothing unless a caller deliberately declares a TASK_CONTRACT. A request
scope's uncertainty disables hard inheritance checking; it cannot silently
turn unparsed legitimate changes into requirements to keep an old selection.
An independently verified REQUEST_SCOPE certificate can establish changed-slot
coverage without asserting that a model understood the request's semantics.

Compiled records are data only. The validator neither changes submissions nor
asserts overall task success. A protocol that enables this factor must disclose
the additional contract and give the same contract to every treatment/control.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

from .backends import reject_privileged
from .subject_scope import compile_scope_coverage


POLICY = "INHERIT_UNMODIFIED_SLOTS"
SOURCE_KIND = "TASK_CONTRACT"
SLOTS = {"current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"}
ACTIVE_CONSTRAINT_STATUSES = {"ACTIVE", "SATISFIED", "VIOLATED", "UNRESOLVED"}


def declare_inheritance_contract(
    contract_id: str,
    group_id: str,
    text: str,
    source_refs: list[str],
    *,
    enabled: bool = False,
    version: int = 1,
    created_at: int = 0,
) -> dict:
    """Declare an opt-in policy; no caller gets inheritance by default."""
    if not isinstance(enabled, bool):
        raise ValueError("Contract enabled must be an explicit boolean")
    if not contract_id or not text.strip() or not _refs(source_refs):
        raise ValueError("Declared contract requires ID, text, and source refs")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("Contract version must be a positive integer")
    if not isinstance(created_at, int) or isinstance(created_at, bool) or created_at < 0:
        raise ValueError("Contract created_at must be a nonnegative causal step")
    result = {"contract_id": str(contract_id), "group_id": str(group_id), "policy": POLICY,
              "source_kind": SOURCE_KIND, "text": text, "source_refs": list(source_refs),
              "enabled": enabled, "version": version, "created_at": created_at}
    reject_privileged(result)
    return result


def _refs(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(ref, str) and bool(ref) for ref in value)


def _day(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, str) and re.fullmatch(r"[1-9]\d*", value):
        return int(value)
    return None


def _version(value: dict) -> int | None:
    raw = value.get("version", value.get("revision"))
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1 else None


def _scope_group(value: dict) -> str | None:
    supplied = value.get("group_id", value.get("group"))
    scope = value.get("scope")
    if supplied is None and isinstance(scope, dict):
        supplied = scope.get("group")
    return str(supplied) if supplied is not None else None


def _check_group(value: dict, group: str) -> None:
    supplied = _scope_group(value)
    if supplied is not None and supplied != group:
        raise ValueError("Inheritance input group differs from declared group")


def _base_ref(record: dict, day: int) -> str:
    expected = f"decision:{record['traveler']}:day{day}:{record['slot']}"
    if record.get("decision_id", expected) != expected:
        raise ValueError("Base decision ID disagrees with traveler/day/slot")
    return expected


def _empty(status: str, group: str, traveler: str, diagnostics=None) -> dict:
    return {"schema_version": 1, "status": status, "group_id": group, "traveler": traveler,
            "source_kind": SOURCE_KIND, "hard_checks_enabled": False,
            "obligations": [], "constraints": [], "exceptions": [],
            "diagnostics": diagnostics or []}


def compile_inheritance_obligations(
    contract: dict | None,
    base_decisions: list[dict],
    request: dict,
    extraction: dict | None = None,
    *,
    scope_coverage: dict | None = None,
) -> dict:
    """Compile grouped obligations and exact graph-ready leaf records.

    Each explicitly constrained current day/slot is an exception, even when the
    constraint changes only a price, rating, cuisine, or room attribute. Scoped
    unresolved items may also exempt a slot. Unscoped/malformed uncertainty
    disables all hard checking while retaining declared candidate records as
    UNRESOLVED hints. Full request wording is never treated as contract evidence.

    When scope_coverage is supplied, the current request must separately declare
    its raw query, already visible prior_travelers, and current source_refs. The
    entire certificate is recomputed from those inputs and compared exactly;
    a supplied flag, altered source, hash, span, or other field cannot enable
    checks. This independently verifies changed-slot scope only. Typed model
    extraction becomes advisory and does not assert semantic interpretation or
    choose exceptions. The absent-certificate path preserves legacy behavior.
    """
    if not isinstance(request, dict):
        raise ValueError("Current request must be an object")
    reject_privileged(request)
    group = str(request.get("group_id", ""))
    traveler = request.get("traveler")
    if not group or not isinstance(traveler, str) or not traveler:
        raise ValueError("Current request needs group_id and traveler")
    if contract is None:
        return _empty("DISABLED", group, traveler)
    if not isinstance(contract, dict):
        raise ValueError("Task contract must be an object or None")
    reject_privileged(contract)
    _check_group(contract, group)
    if contract.get("enabled") is not True:
        return _empty("DISABLED", group, traveler)
    if contract.get("source_kind") != SOURCE_KIND or contract.get("policy") != POLICY:
        raise ValueError("Hard inheritance requires an explicitly declared TASK_CONTRACT policy")
    contract_version = _version(contract)
    if not contract.get("contract_id") or not isinstance(contract.get("text"), str) or not contract["text"].strip() or not _refs(contract.get("source_refs")) or contract_version is None:
        raise ValueError("Declared contract lacks identity, text, sources, or version")
    step = request.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Current request needs a nonnegative causal step")
    diagnostics = []
    uncertain = False
    if contract.get("created_at", 0) > step:
        return _empty("UNCERTAIN", group, traveler, [{"code": "FUTURE_TASK_CONTRACT"}])
    if not isinstance(base_decisions, list):
        raise ValueError("Known base decisions must be a list")
    histories = {}
    base_names = set()
    future_only = set()
    for raw in base_decisions:
        if not isinstance(raw, dict):
            diagnostics.append({"code": "MALFORMED_BASE_DECISION"})
            uncertain = True
            continue
        reject_privileged(raw)
        _check_group(raw, group)
        if raw.get("attribute", "value") not in (None, "value"):
            continue
        day, version = _day(raw.get("day")), _version(raw)
        if day is None or version is None or raw.get("slot") not in SLOTS or not isinstance(raw.get("traveler"), str) or not raw["traveler"] or "value" not in raw or not _refs(raw.get("source_refs")):
            diagnostics.append({"code": "MALFORMED_BASE_DECISION"})
            uncertain = True
            continue
        rid = _base_ref(raw, day)
        created = raw.get("created_at")
        if not isinstance(created, int) or isinstance(created, bool) or created < 0:
            diagnostics.append({"code": "UNVERIFIABLE_BASE_VISIBILITY", "reference": rid})
            uncertain = True
            continue
        if created > step:
            future_only.add(rid)
            continue
        histories.setdefault(rid, []).append(copy.deepcopy(raw))
        base_names.add(raw["traveler"])
    if len(base_names) != 1:
        diagnostics.append({"code": "AMBIGUOUS_OR_MISSING_BASE_TRAVELER"})
        uncertain = True
    selected = []
    for rid, history in histories.items():
        highest = max(_version(row) for row in history)
        versions = [row for row in history if _version(row) == highest]
        if len(versions) > 1 and any(row != versions[0] for row in versions[1:]):
            diagnostics.append({"code": "AMBIGUOUS_BASE_VERSION", "reference": rid, "version": highest})
            uncertain = True
            continue
        latest = versions[0]
        if latest.get("status") != "COMMITTED":
            diagnostics.append({"code": "BASE_SLOT_NOT_KNOWN_VALID", "reference": rid})
            uncertain = True
            continue
        selected.append(latest)
    if future_only - set(histories):
        diagnostics.append({"code": "FUTURE_BASE_SLOT", "references": sorted(future_only - set(histories))})
        uncertain = True
    if not selected:
        uncertain = True
        diagnostics.append({"code": "NO_KNOWN_BASE_SLOTS"})

    parser = extraction if extraction is not None else request.get("extraction")
    exceptions = {}
    verified_scope = None
    scope_verified = False
    advisory_diagnostics = []

    def exception_from(item: Any, source: str) -> bool:
        if not isinstance(item, dict):
            return False
        _check_group(item, group)
        subject = item.get("subject")
        if not isinstance(subject, dict):
            return False
        person, day, slot = subject.get("traveler"), _day(subject.get("day")), subject.get("slot")
        if day is None or slot not in SLOTS or person not in (traveler, "*"):
            return False
        refs = item.get("source_refs", [])
        if not _refs(refs):
            return False
        key = (day, slot)
        entry = exceptions.setdefault(key, {"day": day, "slot": slot, "reasons": [], "source_refs": [], "constraint_refs": []})
        entry["reasons"].append(source)
        entry["source_refs"] = list(dict.fromkeys([*entry["source_refs"], *refs]))
        if item.get("constraint_id"):
            revision = _version(item)
            entry["constraint_refs"].append(item["constraint_id"] + (f"@v{revision}" if revision else ""))
        return True

    if scope_coverage is not None:
        # These declarations belong to the caller's current causal request;
        # never copy them from the certificate that is being verified.
        prior = request.get("prior_travelers")
        refs = request.get("source_refs")
        if not isinstance(prior, list) or not _refs(refs):
            diagnostics.append({"code": "MISSING_INDEPENDENT_SCOPE_DECLARATIONS"})
            uncertain = True
        else:
            try:
                verified_scope = compile_scope_coverage(request, prior, refs)
            except (TypeError, ValueError):
                diagnostics.append({"code": "INDEPENDENT_SCOPE_RECOMPUTATION_FAILED"})
                uncertain = True
        if verified_scope is not None:
            try:
                if not isinstance(scope_coverage, dict):
                    raise ValueError("Scope certificate must be an object")
                reject_privileged(scope_coverage)
                # JSON comparison also distinguishes booleans from integers,
                # and rejects non-JSON values rather than stringifying them.
                serialized = json.dumps(scope_coverage, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                expected = json.dumps(verified_scope, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                scope_verified = serialized == expected
            except (TypeError, ValueError):
                scope_verified = False
            if not scope_verified:
                diagnostics.append({"code": "INDEPENDENT_SCOPE_CERTIFICATE_MISMATCH"})
                uncertain = True
            if verified_scope["coverage_complete"] is not True:
                diagnostics.append({"code": "INDEPENDENT_SCOPE_COVERAGE_UNCERTAIN", "detail": copy.deepcopy(verified_scope["diagnostics"])})
                uncertain = True
            for row in verified_scope["scopes"]:
                if row.get("independently_scoped") is not True:
                    continue
                if not exception_from(row, "INDEPENDENT_CURRENT_SCOPE"):
                    diagnostics.append({"code": "INVALID_RECOMPUTED_CURRENT_SCOPE"})
                    uncertain = True
                    continue
                subject = row["subject"]
                entry = exceptions[(_day(subject["day"]), subject["slot"])]
                entry["source_kind"] = "REQUEST_SCOPE"
                entry.setdefault("scope_evidence", []).append({
                    "evidence_quote": row["evidence_quote"],
                    "evidence_span": copy.deepcopy(row["evidence_span"]),
                    "scope_cues": copy.deepcopy(row["scope_cues"]),
                    "source_refs": copy.deepcopy(row["source_refs"]),
                })
        # Typed extraction may still expose uncertain relation/policy semantics
        # to the caller. It cannot change independently certified exceptions or
        # disable an otherwise complete unchanged-slot contract check.
        if not isinstance(parser, dict):
            advisory_diagnostics.append({"code": "MISSING_ADVISORY_REQUEST_INTERPRETATION"})
        else:
            reject_privileged(parser)
            supplied_group = _scope_group(parser)
            if supplied_group is not None and supplied_group != group:
                advisory_diagnostics.append({"code": "ADVISORY_REQUEST_GROUP_MISMATCH"})
            else:
                if parser.get("parse_success") is not True:
                    advisory_diagnostics.append({"code": "ADVISORY_REQUEST_INTERPRETATION_UNCONFIRMED"})
                if parser.get("unresolved"):
                    advisory_diagnostics.append({"code": "ADVISORY_REQUEST_SEMANTICS_UNRESOLVED", "detail": copy.deepcopy(parser["unresolved"])})
    else:
        if not isinstance(parser, dict):
            diagnostics.append({"code": "MISSING_REQUEST_INTERPRETATION"})
            uncertain = True
            parser = {}
        reject_privileged(parser)
        _check_group(parser, group)
        if parser.get("parse_success") is False:
            diagnostics.append({"code": "REQUEST_PARSE_FAILED"})
            uncertain = True
        rows = parser.get("constraints", [])
        unresolved = parser.get("unresolved", [])
        if not isinstance(rows, list) or not isinstance(unresolved, list):
            diagnostics.append({"code": "MALFORMED_REQUEST_INTERPRETATION"})
            uncertain = True
            rows, unresolved = [], []
        for row in rows:
            if isinstance(row, dict) and row.get("status", "ACTIVE") not in ACTIVE_CONSTRAINT_STATUSES:
                continue
            if not exception_from(row, "CURRENT_TYPED_CONSTRAINT"):
                diagnostics.append({"code": "UNSCOPED_OR_UNSUPPORTED_CURRENT_CONSTRAINT"})
                uncertain = True
        for item in unresolved:
            if not exception_from(item, "SCOPED_UNRESOLVED_REQUEST"):
                diagnostics.append({"code": "UNSCOPED_REQUEST_UNCERTAINTY", "detail": copy.deepcopy(item)})
                uncertain = True
        if parser.get("parse_success") is not True:
            diagnostics.append({"code": "REQUEST_INTERPRETATION_NOT_CONFIRMED"})
            uncertain = True
        if parser.get("interpretation_complete") is not True:
            diagnostics.append({"code": "REQUEST_COVERAGE_NOT_VERIFIED"})
            uncertain = True

    contract_ref = f"task-contract:{contract['contract_id']}@v{contract_version}"
    result = _empty("UNCERTAIN" if uncertain else "ACTIVE", group, traveler, diagnostics)
    result.update(hard_checks_enabled=not uncertain, contract_id=contract["contract_id"], contract_ref=contract_ref,
                  contract_version=contract_version, contract_text=contract["text"],
                  contract_source_refs=copy.deepcopy(contract["source_refs"]),
                  base_traveler=next(iter(base_names)) if len(base_names) == 1 else None,
                  exceptions=[exceptions[key] for key in sorted(exceptions)])
    if scope_coverage is not None:
        result.update(scope_coverage=copy.deepcopy(verified_scope), scope_coverage_verified=scope_verified,
                      exception_basis="INDEPENDENT_REQUEST_SCOPE", semantic_interpretation_verified=False,
                      advisory_diagnostics=advisory_diagnostics)
    by_day = {}
    for record in sorted(selected, key=lambda row: (_day(row["day"]), row["slot"])):
        day, slot = _day(record["day"]), record["slot"]
        if (day, slot) in exceptions:
            continue
        base_id = _base_ref(record, day)
        base_ref = f"{base_id}@v{record['version']}"
        cid = f"constraint:task-contract:{contract['contract_id']}:{traveler}:day{day}:{slot}"
        source_refs = list(dict.fromkeys([*contract["source_refs"], *record["source_refs"]]))
        leaf = {"constraint_id": cid, "kind": "INHERITED", "source_kind": SOURCE_KIND,
                "scope": {"group": group, "traveler": traveler, "day": day},
                "subject": {"traveler": traveler, "day": day, "slot": slot, "attribute": "value"},
                "operator": "EQUALS", "target": {"value": copy.deepcopy(record["value"])},
                "source_refs": source_refs, "depends_on": [base_ref], "contract_ref": contract_ref,
                "base_decision_ref": base_ref, "binding_mode": "DECLARED_BASE_SNAPSHOT",
                "version": contract_version, "revision": contract_version,
                "status": "UNRESOLVED" if uncertain else "ACTIVE", "hard_check": not uncertain}
        result["constraints"].append(leaf)
        group_record = by_day.setdefault(day, {"obligation_id": f"inheritance:{contract['contract_id']}:{traveler}:day{day}",
            "contract_ref": contract_ref, "source_kind": SOURCE_KIND, "traveler": traveler, "day": day,
            "version": contract_version, "status": "UNRESOLVED" if uncertain else "ACTIVE",
            "source_refs": copy.deepcopy(contract["source_refs"]), "depends_on": [], "slots": []})
        group_record["depends_on"].append(base_ref)
        group_record["slots"].append({"slot": slot, "value": copy.deepcopy(record["value"]),
            "constraint_ref": f"{cid}@v{contract_version}", "base_decision_ref": base_ref,
            "base_source_refs": copy.deepcopy(record["source_refs"])})
    result["obligations"] = [by_day[day] for day in sorted(by_day)]
    return result


def validate_inheritance(compiled: dict, actual_decisions: list[dict]) -> dict:
    """Check only declared known unchanged slots, preserving actual actions."""
    if not isinstance(compiled, dict) or not isinstance(actual_decisions, list):
        raise ValueError("Compiled contract and actual decision list required")
    reject_privileged(compiled)
    group, traveler = compiled.get("group_id"), compiled.get("traveler")
    result = {"status": "DISABLED" if compiled.get("status") == "DISABLED" else "UNVERIFIED",
              "source_kind": SOURCE_KIND, "task_success": None, "checked_slot_count": 0,
              "violations": [], "diagnostics": [], "contract_ref": compiled.get("contract_ref")}
    if not compiled.get("hard_checks_enabled"):
        result["diagnostics"] = copy.deepcopy(compiled.get("diagnostics", []))
        return result
    if compiled.get("source_kind") != SOURCE_KIND or compiled.get("status") != "ACTIVE":
        raise ValueError("Hard validation requires an active explicit task contract")
    actual = {}
    for record in actual_decisions:
        if not isinstance(record, dict):
            result["diagnostics"].append({"code": "MALFORMED_ACTUAL_DECISION"})
            continue
        reject_privileged(record)
        _check_group(record, group)
        if record.get("traveler") != traveler or record.get("attribute", "value") not in (None, "value"):
            continue
        day, slot = _day(record.get("day")), record.get("slot")
        if day is None or slot not in SLOTS or "value" not in record:
            result["diagnostics"].append({"code": "MALFORMED_ACTUAL_DECISION"})
            continue
        actual.setdefault((day, slot), []).append(copy.deepcopy(record["value"]))
    for record in compiled.get("constraints", []):
        if not record.get("hard_check"):
            continue
        subject = record["subject"]
        key = (subject["day"], subject["slot"])
        values = actual.get(key, [])
        expected = record["target"]["value"]
        result["checked_slot_count"] += 1
        if not values:
            code = "INHERITED_SLOT_MISSING"
        elif any(value != values[0] for value in values[1:]):
            code = "AMBIGUOUS_ACTUAL_SUBMISSION"
        elif not _same_slot_value(key[1], values[0], expected):
            code = "INHERITED_SLOT_CHANGED"
        else:
            continue
        result["violations"].append({"code": code, "constraint_id": record["constraint_id"],
            "subject": copy.deepcopy(subject), "expected": copy.deepcopy(expected),
            "actual": copy.deepcopy(values[0]) if len(values) == 1 else copy.deepcopy(values),
            "source_kind": SOURCE_KIND, "contract_ref": record["contract_ref"],
            "base_decision_ref": record["base_decision_ref"], "source_refs": copy.deepcopy(record["source_refs"])})
    result["status"] = "VIOLATED" if result["violations"] else "UNVERIFIED" if result["diagnostics"] else "PASSED"
    return result


def _same_slot_value(slot: str, actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    # The public planner uses semicolon-delimited attraction lists. A single
    # final list delimiter does not select a different attraction. Every name,
    # internal delimiter, order and duplicate remains literal; no score-derived
    # similarity threshold or automatic submission rewrite is used.
    return (slot == "attraction" and isinstance(actual, str) and isinstance(expected, str)
            and actual.removesuffix(";") == expected.removesuffix(";"))


__all__ = ["declare_inheritance_contract", "compile_inheritance_obligations", "validate_inheritance"]
