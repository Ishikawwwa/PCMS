"""Limited travel-language adapter for individually evidenced predicates.

The reusable comparison kernel has no travel knowledge. This adapter recognizes
only explicit numeric cues and a few literal shared-choice cues, and never
claims full natural-language interpretation. Unsupported wording and missing,
ambiguous, stale, or unscoped data remain unverified. Callers must first verify
raw request source events and property evidence; this module checks their causal
record declarations and exact entity/version bindings without reading tools.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re

from .backends import reject_privileged
from .predicate_kernel import compare,transform
from .subject_scope import compile_scope_coverage


SOURCE_KIND = "TRAVEL_REQUEST_PREDICATE_ADAPTER"
_NUM = r"(?:\d+(?:\.\d+)?)"
_DAYS = {name: index for index, name in enumerate(
    ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"), 1)}
_DAY = re.compile(r"\b(?:(?P<word>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)[ -]+day|day[ -]*(?P<number>[1-9]\d*))\b", re.I)
_SLOT = re.compile(r"\b(?:breakfast|lunch|dinner|accommodation)\b", re.I)
_NEGATION = re.compile(r"\b(?:not|never|neither|unless|without|cannot|avoid|reject|exclude|except|if|whether|maybe|perhaps|possibly|might|considering)\b|\b(?:don|doesn|didn|isn|wasn|wouldn|shouldn|can)['’]t\b|\bno\s+longer\b", re.I)
_INTENT = re.compile(r"\bI(?:\s+(?:want|need|prefer|hope)|\s+would\s+like|['’]d\s+(?:like|prefer)|['’]m\s+(?:looking|seeking)|\s+am\s+(?:looking|seeking))\b", re.I)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def _refs(value):
    return isinstance(value, list) and bool(value) and all(isinstance(ref, str) and bool(ref) for ref in value)


def _group(record):
    group = record.get("group_id")
    if group is None and isinstance(record.get("scope"), dict): group = record["scope"].get("group")
    return str(group) if group is not None else None


def _id(subject):
    rid = f"decision:{subject['traveler']}:day{subject['day']}:{subject['slot']}"
    return rid if subject.get("attribute", "value") == "value" else rid + ":" + subject["attribute"]


def _record_ref(record):
    return f"{record['decision_id']}@v{record['version']}"


def _index(records, group, step):
    if not isinstance(records, list): raise ValueError("Visible records must be a causal record list")
    histories = {}
    invalid = set()
    for record in records:
        if not isinstance(record, dict): continue
        reject_privileged(record)
        rid = record.get("decision_id")
        if not isinstance(rid, str): continue
        try:
            json.dumps(record, allow_nan=False)
        except (TypeError, ValueError):
            invalid.add(rid)
            continue
        if _group(record) != group or type(record.get("version")) is not int or record["version"] < 1 or type(record.get("created_at")) is not int or record["created_at"] < 0 or not _refs(record.get("source_refs")):
            invalid.add(rid)
            continue
        if record["created_at"] > step: continue
        try:
            if rid != _id(record) or type(record["day"]) is not int or record["day"] < 1 or "value" not in record:
                invalid.add(rid)
                continue
        except KeyError:
            invalid.add(rid)
            continue
        histories.setdefault(rid, []).append(copy.deepcopy(record))
    selected = {}
    for rid, history in histories.items():
        latest = max(row["version"] for row in history)
        candidates = [row for row in history if row["version"] == latest]
        if any(row != candidates[0] for row in candidates[1:]) or candidates[0].get("status") != "COMMITTED" or rid in invalid:
            continue
        selected[rid] = candidates[0]
    return selected


def _resolve(index, rid):
    record = index.get(rid)
    if record is None: return None, "MISSING_OR_AMBIGUOUS_CAUSAL_RECORD"
    if record.get("attribute", "value") != "value":
        entity = index.get(rid.rsplit(":", 1)[0])
        if entity is None or record.get("entity_ref") != _record_ref(entity):
            return None, "PROPERTY_ENTITY_VERSION_UNVERIFIED"
    return record, None


def _prior_names(request):
    names = []
    for item in request.get("prior_travelers", []):
        name = item if isinstance(item, str) else item.get("traveler", item.get("name")) if isinstance(item, dict) else None
        if isinstance(name, str) and name != request["traveler"] and name not in names: names.append(name)
    return names


def _anchor(clause, names):
    mentions = [(match.start(), match.end(), name) for name in names
                for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", clause)]
    if len(mentions) != 1: return None, "MISSING_OR_MULTIPLE_PRIOR_ANCHORS"
    start, end, name = mentions[0]
    suffix = clause[end:]
    if not re.match(r"['’]s\b", suffix): return None, "PRIOR_SCOPE_NOT_EXPLICIT"
    # The named possessive's reference phrase ends before further predicates.
    reference = re.split(r",|\band\b|\bbut\b|\bwith\b|(?<!\d)\.(?!\d)", suffix[2:], maxsplit=1, flags=re.I)[0]
    days = {int(match["number"]) if match["number"] else _DAYS[match["word"].lower()] for match in _DAY.finditer(reference)}
    slots = {match[0].lower() for match in _SLOT.finditer(reference)}
    if len(days) != 1 or len(slots) != 1: return None, "PRIOR_DAY_SLOT_UNVERIFIED"
    return {"traveler": name, "day": next(iter(days)), "slot": next(iter(slots)),
            "evidence_span": {"start": start, "end": end + 2 + len(reference)}}, None


def _binding(record):
    return {"record_ref": _record_ref(record), "record_id": record["decision_id"],
            "version": record["version"], "created_at": record["created_at"],
            "source_refs": copy.deepcopy(record["source_refs"]), "value": copy.deepcopy(record["value"]),
            "entity_ref": record.get("entity_ref"), "unit": record.get("unit")}


def _explicit_connector_matches(clause, cue, anchor, connector, *, required=False):
    remainder = clause[cue.end():]
    match = re.match(r"\s*" + re.escape(connector) + r"\b\s*", remainder, re.I)
    if match:
        remainder = remainder[match.end():]
    elif not required:
        return _shared_anchor_ellipsis(clause, cue, anchor)
    # The connector may already be part of the matched cue (price windows).
    return re.match(re.escape(anchor["traveler"]) + r"['’]s\b", remainder.lstrip()) is not None


def _shared_anchor_ellipsis(clause, cue, anchor):
    """Allow only a trailing conjunct of an earlier explicit comparison.

    A person appearing somewhere in a clause is not a comparand. This limited
    form requires the already named day/slot to have been introduced directly
    by a numeric comparison, a pure coordination bridge, and no later words
    that could introduce another basis or an ambient statement.
    """
    span = anchor.get("evidence_span")
    if not span or span["end"] > cue.start():
        return False
    before = clause[:span["start"]]
    explicit = re.search(
        r"(?:\b(?:priced?|costs?|costing)\s+within\s+(?:\$" + _NUM + r"|" + _NUM + r"%)\s+of"
        r"|\b(?:rated|rating)\s+within\s+" + _NUM + r"\s+of"
        r"|\b(?:priced?|costs?|costing)\s+at\s+least\s+\$" + _NUM + r"\s+(?:less|more)\s+than"
        r"|\brated\s+(?:at\s+least\s+as\s+high|higher|lower)\s+than)\s*$", before, re.I)
    if explicit is None:
        return False
    bridge = clause[span["end"]:cue.start()]
    if not re.fullmatch(r"\s*,?\s*(?:and|but|with)\s+(?:(?:is|has|a|an)\s+){0,2}", bridge, re.I):
        return False
    return re.fullmatch(r"\s*[.!?;]?\s*", clause[cue.end():]) is not None


def compile_request_checks(request, visible_records):
    """Compile supported individual clauses, without model interpretation."""
    if not isinstance(request, dict): raise ValueError("Current request must be an object")
    reject_privileged(request)
    group, traveler, step = str(request.get("group_id", "")), request.get("traveler"), request.get("step")
    if not group or not isinstance(traveler, str) or not traveler or type(step) is not int or step < 0:
        raise ValueError("Request needs group, current traveler, and causal step")
    result = {"schema_version": 1, "source_kind": SOURCE_KIND, "group_id": group, "traveler": traveler,
              "step": step, "predicates": [], "unverified": [], "full_request_semantics_verified": False,
              "task_success": None}
    try:
        coverage = compile_scope_coverage(request, request.get("prior_travelers"), request.get("source_refs"))
    except (TypeError, ValueError):
        result["unverified"].append({"code": "CURRENT_SCOPE_INPUTS_UNVERIFIED"})
        return result
    result["request_sha256"] = coverage["request_sha256"]
    result["scope_certificate"] = coverage
    index = _index(visible_records, group, step)
    names = _prior_names(request)

    def unresolved(code, scope, **detail):
        result["unverified"].append({"code": code, "subject": copy.deepcopy(scope["subject"]),
            "evidence_quote": scope["evidence_quote"], "evidence_span": copy.deepcopy(scope["evidence_span"]),
            "source_refs": copy.deepcopy(scope["source_refs"]), **detail})

    def emit(scope, attribute, operator, anchor, cue, *, scale="1", offset="0", tolerance=None):
        target_subject = {**{key: anchor[key] for key in ("traveler", "day", "slot")}, "attribute": attribute}
        record, error = _resolve(index, _id(target_subject))
        if error:
            unresolved(error, scope, target_subject=target_subject)
            return
        if attribute != "value" and _number(record["value"]) is None:
            unresolved("NONNUMERIC_REFERENCE", scope, target_subject=target_subject)
            return
        if attribute == "price" and (not isinstance(record.get("unit"), str) or not record["unit"]):
            unresolved("PRICE_UNIT_UNVERIFIED", scope, target_subject=target_subject)
            return
        subject = {**scope["subject"], "attribute": attribute}
        target = {"ref": _record_ref(record), "scale": scale, "offset": offset}
        if tolerance is not None: target["tolerance"] = tolerance
        if record.get("unit") is not None: target["unit"] = record["unit"]
        evidence = {"quote": cue[0], "start": scope["evidence_span"]["start"] + cue.start(),
                    "end": scope["evidence_span"]["start"] + cue.end()}
        pid = "request-predicate:" + hashlib.sha256((coverage["request_sha256"] + repr((subject, operator, target, evidence))).encode()).hexdigest()[:24]
        result["predicates"].append({"predicate_id": pid, "source_kind": SOURCE_KIND,
            "eligibility": "INDEPENDENT_LITERAL_CUE_AND_CAUSAL_ANCHOR", "subject": subject,
            "operator": operator, "target": target, "binding": _binding(record),
            "source_refs": copy.deepcopy(scope["source_refs"]), "evidence_quote": scope["evidence_quote"],
            "evidence_span": copy.deepcopy(scope["evidence_span"]), "predicate_evidence": evidence,
            "target_evidence": {"quote": scope["evidence_quote"][anchor["evidence_span"]["start"]:anchor["evidence_span"]["end"]],
                "start": scope["evidence_span"]["start"] + anchor["evidence_span"]["start"],
                "end": scope["evidence_span"]["start"] + anchor["evidence_span"]["end"]} if anchor.get("evidence_span") else copy.deepcopy(evidence),
            "dependencies": [_record_ref(record), *([record["entity_ref"]] if record.get("entity_ref") else [])]})

    for scope in coverage["scopes"]:
        if not scope.get("independently_scoped"):
            unresolved("CURRENT_SUBJECT_SCOPE_UNCERTAIN", scope)
            continue
        clause = scope["evidence_quote"]
        if not _INTENT.search(clause):
            unresolved("SUPPORTED_EXPLICIT_REQUEST_INTENT_MISSING", scope)
            continue
        if _NEGATION.search(clause):
            unresolved("NEGATED_OR_CONDITIONAL_CLAUSE_UNSUPPORTED", scope)
            continue
        before = len(result["predicates"])
        anchor, anchor_error = _anchor(clause, names)
        if anchor is not None:
            for cue in re.finditer(r"\b(?:priced?|costs?|costing)\s+within\s+\$(" + _NUM + r")\s+of\b", clause, re.I):
                if not _explicit_connector_matches(clause, cue, anchor, "of", required=True):
                    unresolved("PRICE_COMPARAND_NOT_THE_EXPLICIT_PRIOR_ANCHOR", scope)
                    continue
                emit(scope, "price", "ABS_DIFF_LTE", anchor, cue, tolerance=str(Decimal(cue[1])))
            for cue in re.finditer(r"\b(?:priced?|costs?|costing)\s+within\s+(" + _NUM + r")%\s+of\b", clause, re.I):
                percent = transform(cue[1],"0.01",0)
                if percent is None:
                    unresolved("ARITHMETIC_RANGE_UNVERIFIED", scope)
                    continue
                if not _explicit_connector_matches(clause, cue, anchor, "of", required=True):
                    unresolved("PRICE_COMPARAND_NOT_THE_EXPLICIT_PRIOR_ANCHOR", scope)
                    continue
                record, _ = _resolve(index, _id({**anchor, "attribute": "price"}))
                if record is not None and _number(record["value"]) is not None and _number(record["value"]) < 0:
                    unresolved("PERCENTAGE_REFERENCE_UNSUPPORTED", scope)
                    continue
                emit(scope, "price", "GTE", anchor, cue, scale=str(transform(percent,-1,1)))
                emit(scope, "price", "LTE", anchor, cue, scale=str(transform(percent,1,1)))
            for cue in re.finditer(r"\b(?:priced?|costs?|costing)\s+at\s+least\s+\$(" + _NUM + r")\s+(less|more)\b", clause, re.I):
                if not _explicit_connector_matches(clause, cue, anchor, "than"):
                    unresolved("PRICE_COMPARAND_NOT_THE_EXPLICIT_PRIOR_ANCHOR", scope)
                    continue
                amount = Decimal(cue[1])
                emit(scope, "price", "LTE" if cue[2].lower() == "less" else "GTE", anchor, cue,
                     offset=str(amount.copy_negate() if cue[2].lower() == "less" else amount))
            for cue in re.finditer(r"\b(?:rated|rating)\s+within\s+(" + _NUM + r")(?!\w|\.\d|\s*%)", clause, re.I):
                if not _explicit_connector_matches(clause, cue, anchor, "of"):
                    unresolved("RATING_COMPARAND_NOT_THE_EXPLICIT_PRIOR_ANCHOR", scope)
                    continue
                emit(scope, "rating", "ABS_DIFF_LTE", anchor, cue, tolerance=str(Decimal(cue[1])))
            for cue in re.finditer(r"\brated\s+(at\s+least\s+as\s+high|higher|lower)\b", clause, re.I):
                if not _explicit_connector_matches(clause, cue, anchor, "than"):
                    unresolved("RATING_COMPARAND_NOT_THE_EXPLICIT_PRIOR_ANCHOR", scope)
                    continue
                operator = "GTE" if cue[1].lower().startswith("at") else "GT" if cue[1].lower() == "higher" else "LT"
                emit(scope, "rating", operator, anchor, cue)
        else:
            # Only explicit shared-choice verbs with a complete known-person
            # suffix. Possessive target scope is never guessed from anaphora.
            alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
            join = re.search(r"\b(?:join|eat\s+with|stay\s+with|share\s+accommodation\s+with)\s+(?P<name>" + (alternatives or r"(?!)") + r")(?P<meal>\s+for\s+(?:breakfast|lunch|dinner))?(?:\s+again)?\.?$", clause, re.I)
            if join and sum(len(list(re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", clause))) for name in names) == 1:
                shared = {"traveler": join["name"], "day": scope["subject"]["day"], "slot": scope["subject"]["slot"]}
                emit(scope, "value", "EQUALS", shared, join)
            elif anchor_error:
                unresolved(anchor_error, scope)
        if len(result["predicates"]) == before:
            unresolved("NO_SUPPORTED_INDEPENDENT_PREDICATE", scope)
    # Preserve unexplained scope uncertainty separately; an otherwise verified
    # individual predicate never becomes a declaration of complete semantics.
    result["scope_uncertainty"] = copy.deepcopy(coverage["diagnostics"])
    return result


def validate_request_checks(compiled, actual_records, *, step=None):
    """Evaluate source-certified predicates against current causal records."""
    if not isinstance(compiled, dict): raise ValueError("Compiled predicates must be an object")
    reject_privileged(compiled)
    boundary = compiled.get("step") if step is None else step
    if type(boundary) is not int or boundary < compiled.get("step", 0):
        raise ValueError("Evaluation must use a nonnegative step after compilation")
    group = str(compiled.get("group_id", ""))
    index = _index(actual_records, group, boundary)
    result = {"status": "UNVERIFIED", "source_kind": SOURCE_KIND, "task_success": None,
        "violations": [], "satisfied": [], "unverified": [], "predicate_outcomes": [],
        "checked_predicate_count": 0, "full_request_semantics_verified": False}
    for predicate in compiled.get("predicates", []):
        outcome = {"predicate_id": predicate["predicate_id"], "subject": copy.deepcopy(predicate["subject"]),
            "operator": predicate["operator"], "source_refs": copy.deepcopy(predicate["source_refs"]),
            "evidence_quote": predicate["evidence_quote"], "evidence_span": copy.deepcopy(predicate["evidence_span"])}
        target, target_error = _resolve(index, predicate["binding"]["record_id"])
        actual, actual_error = _resolve(index, _id(predicate["subject"]))
        error = target_error or actual_error
        if not error and (_record_ref(target) != predicate["binding"]["record_ref"] or target.get("entity_ref") != predicate["binding"].get("entity_ref") or target["value"] != predicate["binding"]["value"] or target.get("unit") != predicate["binding"].get("unit")):
            error = "DEPENDENCY_CHANGED"
        unit = predicate["target"].get("unit")
        if not error and unit is not None and actual.get("unit") != unit:
            error = "UNITS_MISMATCH"
        if error:
            outcome.update(status="UNVERIFIED", code=error)
        else:
            expected = target["value"]
            if predicate["subject"]["attribute"] != "value":
                number = _number(expected)
                expected = transform(number,predicate["target"]["scale"],predicate["target"]["offset"]) if number is not None else None
            compared = {"value": expected, "tolerance": predicate["target"]["tolerance"]} if predicate["operator"] == "ABS_DIFF_LTE" else expected
            passed = compare(actual["value"], predicate["operator"], compared)
            outcome.update(status="SATISFIED" if passed is True else "VIOLATED" if passed is False else "UNVERIFIED",
                actual=copy.deepcopy(actual["value"]), actual_ref=_record_ref(actual), current_entity_ref=actual.get("entity_ref", _record_ref(actual)),
                target_ref=_record_ref(target), target_entity_ref=target.get("entity_ref", _record_ref(target)),
                actual_source_refs=copy.deepcopy(actual["source_refs"]), target_source_refs=copy.deepcopy(target["source_refs"]),
                expected_value=str(expected) if isinstance(expected, Decimal) else expected,
                **({"tolerance": predicate["target"]["tolerance"]} if "tolerance" in predicate["target"] else {}))
            if passed is None: outcome["code"] = "UNSUPPORTED_OR_NONNUMERIC_COMPARISON"
        result["predicate_outcomes"].append(outcome)
        result[{"SATISFIED": "satisfied", "VIOLATED": "violations", "UNVERIFIED": "unverified"}[outcome["status"]]].append(copy.deepcopy(outcome))
    result["checked_predicate_count"] = len(result["satisfied"]) + len(result["violations"])
    result["status"] = "VIOLATED" if result["violations"] else "UNVERIFIED" if result["unverified"] or not result["predicate_outcomes"] else "PASSED"
    return result


__all__ = ["compile_request_checks", "validate_request_checks"]
