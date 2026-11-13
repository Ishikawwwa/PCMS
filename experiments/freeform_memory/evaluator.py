"""Independent offline, three-valued scorer for committed free-form choices.

Only the Python standard library is used. This module does not import runtime,
the memory implementation, model clients, annotations from disk, or credentials.
The caller supplies scorer-only annotations *after* committing a response.
Expected labels are consistency checks and never supply a verdict.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from fractions import Fraction

SATISFIED = "SATISFIED"
VIOLATED = "VIOLATED"
UNVERIFIED = "UNVERIFIED"
STATUSES = {SATISFIED, VIOLATED, UNVERIFIED}
OPERATORS = {
    "EQ", "EQUALS", "NEQ", "NOT_EQUALS", "LT", "LTE", "GT", "GTE",
    "IN", "NOT_IN", "SUPERSET_OF", "SUBSET_OF", "INTERSECTS", "DISJOINT", "ABS_DIFF_LTE",
}
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _pointer_tokens(pointer):
    if not isinstance(pointer, str) or pointer and not pointer.startswith("/"):
        raise ValueError("An RFC 6901 pointer is required")
    result = []
    for token in pointer.split("/")[1:] if pointer else []:
        if re.search(r"~(?![01])", token):
            raise ValueError("Invalid JSON pointer escape")
        result.append(token.replace("~1", "/").replace("~0", "~"))
    return tuple(result)


def _pointer(value, pointer):
    for key in _pointer_tokens(pointer):
        if isinstance(value, list):
            if not key.isdigit() or str(int(key)) != key:
                raise ValueError("Invalid JSON array index")
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError("Pointer does not select structured evidence")
    return copy.deepcopy(value)


def _number(value):
    """Exact finite decimal arithmetic; bool never counts as a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Fraction)):
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = str(value)
    if len(text) > 1024 or not _NUMBER.fullmatch(text):
        return None
    # Bound decimal exponents before Fraction can allocate enormous integers.
    if "e" in text.lower() and abs(int(text.lower().split("e")[1])) > 1024:
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _typed_equal(left, right):
    """JSON type-aware equality; booleans, numbers and numeric strings differ."""
    numeric_types = (int, float, Fraction)
    if not isinstance(left, bool) and not isinstance(right, bool) and isinstance(left, numeric_types) and isinstance(right, numeric_types):
        a, b = _number(left), _number(right)
        return a is not None and b is not None and a == b
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return set(left) == set(right) and all(_typed_equal(left[key], right[key]) for key in left)
    if isinstance(left, float) and not math.isfinite(left):
        return False
    return left == right


def _contains(values, value):
    return any(_typed_equal(item, value) for item in values)


def _compare(left, operator, right, tolerance=None):
    if operator in {"EQ", "EQUALS", "NEQ", "NOT_EQUALS"}:
        equal = _typed_equal(left, right)
        return equal if operator in {"EQ", "EQUALS"} else not equal
    if operator in {"LT", "LTE", "GT", "GTE", "ABS_DIFF_LTE"}:
        a, b = _number(left), _number(right)
        if a is None or b is None:
            return None
        if operator == "ABS_DIFF_LTE":
            limit = _number(tolerance)
            return None if limit is None or limit < 0 else abs(a - b) <= limit
        return {"LT": a < b, "LTE": a <= b, "GT": a > b, "GTE": a >= b}[operator]
    if operator in {"IN", "NOT_IN"}:
        if not isinstance(right, list):
            return None
        present = _contains(right, left)
        return present if operator == "IN" else not present
    if operator in {"SUPERSET_OF", "SUBSET_OF", "INTERSECTS", "DISJOINT"}:
        if not isinstance(left, list) or not isinstance(right, list):
            return None
        intersects = any(_contains(right, item) for item in left)
        return {"SUPERSET_OF": all(_contains(left, item) for item in right),
                "SUBSET_OF": all(_contains(right, item) for item in left),
                "INTERSECTS": intersects, "DISJOINT": not intersects}[operator]
    return None


def _status(value):
    return SATISFIED if value is True else VIOLATED if value is False else UNVERIFIED


def _all(statuses):
    return VIOLATED if VIOLATED in statuses else UNVERIFIED if UNVERIFIED in statuses else SATISFIED


class FactStore:
    """Independent causal ledger with explicit source pointers and conflicts."""
    def __init__(self):
        self.histories = {}

    def append(self, reference, value, unit, step, source, entity_ref=None):
        if not isinstance(reference, str) or not reference:
            raise ValueError("Facts require nonempty reference identities")
        if type(step) is not int or step < 0:
            raise ValueError("Facts require a nonnegative causal step")
        if unit is not None and (not isinstance(unit, str) or not unit):
            raise ValueError("Units require explicit nonempty labels")
        if entity_ref is not None and (not isinstance(entity_ref, dict) or set(entity_ref) != {"ref", "version"}
                                       or not isinstance(entity_ref["ref"], str) or type(entity_ref["version"]) is not int
                                       or entity_ref["version"] < 1):
            raise ValueError("Entity binding requires a reference and exact positive version")
        history = self.histories.setdefault(reference, [])
        if history and step < history[-1]["step"]:
            raise ValueError("Fact writes are not chronological")
        if entity_ref is not None:
            parent = self.resolve(entity_ref["ref"])
            if (not parent["known"] or parent["record"]["version"] != entity_ref["version"]
                    or parent["record"]["step"] > step):
                raise ValueError("Entity binding lacks matching causal evidence at creation")
        history.append({"reference": reference, "version": len(history) + 1, "value": copy.deepcopy(value),
                        "unit": unit, "step": step, "source": copy.deepcopy(source), "entity_ref": copy.deepcopy(entity_ref)})

    def resolve(self, reference, binding_mode="LIVE", version=None, _seen=None):
        if binding_mode not in {"LIVE", "SNAPSHOT"} or (binding_mode == "LIVE" and version is not None):
            raise ValueError("Invalid fact binding mode/version")
        if binding_mode == "SNAPSHOT" and (type(version) is not int or version < 1):
            raise ValueError("Snapshot requires an exact positive version")
        seen = set(_seen or ())
        if reference in seen:
            return {"known": False, "reason": "ENTITY_CYCLE", "reference": reference, "bindings": []}
        seen.add(reference)
        history = self.histories.get(reference, [])
        if not history:
            return {"known": False, "reason": "MISSING_FACT", "reference": reference, "bindings": []}
        selected = history[-1] if binding_mode == "LIVE" else next((record for record in history if record["version"] == version), None)
        if selected is None:
            return {"known": False, "reason": "MISSING_SNAPSHOT_VERSION", "reference": reference, "bindings": []}
        peers = [record for record in history if record["step"] == selected["step"]]
        signature = lambda row: {key: row[key] for key in ("value", "unit", "entity_ref")}
        if any(not _typed_equal(signature(selected), signature(peer)) for peer in peers):
            return {"known": False, "reason": "CONFLICTING_SAME_STEP_FACTS", "reference": reference,
                    "bindings": copy.deepcopy(peers)}
        if selected["entity_ref"]:
            entity = selected["entity_ref"]
            resolved = self.resolve(entity["ref"], _seen=seen)
            if not resolved["known"] or resolved["record"]["version"] != entity["version"]:
                return {"known": False, "reason": "STALE_OR_UNKNOWN_ENTITY", "reference": reference,
                        "bindings": [copy.deepcopy(selected), *resolved["bindings"]]}
        return {"known": True, "record": copy.deepcopy(selected), "bindings": [copy.deepcopy(selected)]}


def _validate_operand(operand, path, subject=False):
    if not isinstance(operand, dict):
        raise ValueError(path + ": operand must be an object")
    allowed = {"ref", "binding_mode", "version"} if subject else {"ref", "value", "binding_mode", "version", "scale", "offset", "tolerance", "unit"}
    if set(operand) - allowed or ("ref" in operand) == ("value" in operand):
        raise ValueError(path + ": operand needs exactly one ref or value and declared metadata")
    if subject and "ref" not in operand:
        raise ValueError(path + ": subject must reference observed evidence")
    if "ref" in operand:
        if not isinstance(operand["ref"], str) or not operand["ref"]:
            raise ValueError(path + ": reference must be nonempty text")
        mode = operand.get("binding_mode", "LIVE")
        if mode not in {"LIVE", "SNAPSHOT"}:
            raise ValueError(path + ": invalid binding mode")
        if mode == "LIVE" and "version" in operand:
            raise ValueError(path + ": LIVE cannot pin a version")
        if mode == "SNAPSHOT" and (type(operand.get("version")) is not int or operand["version"] < 1):
            raise ValueError(path + ": SNAPSHOT requires an exact positive version")
    elif set(operand) & {"binding_mode", "version"}:
        raise ValueError(path + ": literals cannot have reference-binding metadata")
    for name in ("scale", "offset", "tolerance"):
        if name in operand and (_number(operand[name]) is None or name == "tolerance" and _number(operand[name]) < 0):
            raise ValueError(path + ": invalid numeric " + name)
    if "unit" in operand and operand["unit"] is not None and (not isinstance(operand["unit"], str) or not operand["unit"]):
        raise ValueError(path + ": invalid unit label")


def validate_expression(expression, path="$", depth=0):
    """Reject ambiguous/unsupported AST shape instead of simplifying it."""
    if depth > 64 or not isinstance(expression, dict):
        raise ValueError(path + ": expression must be a bounded object AST")
    keys = set(expression)
    if keys == {"literal"}:
        if expression["literal"] is not None and type(expression["literal"]) is not bool:
            raise ValueError(path + ": logical literal must be bool or null")
        return
    if keys in ({"all"}, {"any"}):
        operation = next(iter(keys))
        values = expression[operation]
        if not isinstance(values, list) or not values or len(values) > 128:
            raise ValueError(path + ": compound requires a nonempty bounded list")
        for index, value in enumerate(values):
            validate_expression(value, f"{path}.{operation}[{index}]", depth + 1)
        return
    if keys == {"not"}:
        validate_expression(expression["not"], path + ".not", depth + 1)
        return
    if keys == {"if", "then", "else"}:
        for name in ("if", "then", "else"):
            validate_expression(expression[name], path + "." + name, depth + 1)
        return
    if not {"subject", "operator", "target"}.issubset(keys) or keys - {"subject", "operator", "target", "unit"}:
        raise ValueError(path + ": unsupported expression shape")
    if expression["operator"] not in OPERATORS:
        raise ValueError(path + ": unsupported comparison")
    _validate_operand(expression["subject"], path + ".subject", subject=True)
    _validate_operand(expression["target"], path + ".target")
    unit = expression.get("unit")
    if unit is not None and (not isinstance(unit, str) or not unit):
        raise ValueError(path + ": invalid comparison unit")
    if unit and expression["target"].get("unit") and unit != expression["target"]["unit"]:
        raise ValueError(path + ": conflicting annotation unit declarations")
    if expression["operator"] == "ABS_DIFF_LTE" and "tolerance" not in expression["target"]:
        raise ValueError(path + ": absolute-difference comparison requires tolerance")


def _operand(operand, facts):
    if "value" in operand:
        return {"known": True, "value": copy.deepcopy(operand["value"]), "unit": operand.get("unit"),
                "bindings": [], "kind": "literal"}
    resolved = facts.resolve(operand["ref"], operand.get("binding_mode", "LIVE"), operand.get("version"))
    if not resolved["known"]:
        return {**resolved, "kind": "reference"}
    return {"known": True, "value": resolved["record"]["value"], "unit": resolved["record"]["unit"],
            "bindings": resolved["bindings"], "kind": "reference"}


def _fraction_text(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _evaluate(expression, facts, path):
    keys = set(expression)
    if keys == {"literal"}:
        return {"path": path, "kind": "literal", "status": _status(expression["literal"]), "bindings": []}
    if keys in ({"all"}, {"any"}):
        operation = next(iter(keys))
        children = [_evaluate(value, facts, f"{path}.{operation}[{index}]") for index, value in enumerate(expression[operation])]
        statuses = [child["status"] for child in children]
        verdict = _all(statuses) if operation == "all" else SATISFIED if SATISFIED in statuses else UNVERIFIED if UNVERIFIED in statuses else VIOLATED
        return {"path": path, "kind": operation, "status": verdict, "children": children,
                "bindings": [binding for child in children for binding in child["bindings"]]}
    if keys == {"not"}:
        child = _evaluate(expression["not"], facts, path + ".not")
        verdict = {SATISFIED: VIOLATED, VIOLATED: SATISFIED, UNVERIFIED: UNVERIFIED}[child["status"]]
        return {"path": path, "kind": "not", "status": verdict, "children": [child], "bindings": child["bindings"]}
    if keys == {"if", "then", "else"}:
        condition = _evaluate(expression["if"], facts, path + ".if")
        if condition["status"] == UNVERIFIED:
            branches = [_evaluate(expression[name], facts, path + "." + name) for name in ("then", "else")]
            verdict = branches[0]["status"] if branches[0]["status"] == branches[1]["status"] else UNVERIFIED
            children = [condition, *branches]
            selected_branch = "BOTH_CONDITION_UNKNOWN"
        else:
            selected_branch = "then" if condition["status"] == SATISFIED else "else"
            branch = _evaluate(expression[selected_branch], facts, path + "." + selected_branch)
            verdict, children = branch["status"], [condition, branch]
        return {"path": path, "kind": "if", "status": verdict, "selected_branch": selected_branch,
                "children": children, "bindings": [binding for child in children for binding in child["bindings"]]}
    left, right = _operand(expression["subject"], facts), _operand(expression["target"], facts)
    result = {"path": path, "kind": "atom", "operator": expression["operator"], "status": UNVERIFIED,
              "bindings": [*left["bindings"], *right["bindings"]], "subject": left, "target": right}
    if not left["known"] or not right["known"]:
        result["reason"] = "UNKNOWN_OPERAND"
        return result
    unit = expression.get("unit") or expression["target"].get("unit")
    references = [operand for operand in (left, right) if operand["kind"] == "reference"]
    if unit and any(operand["unit"] != unit for operand in references):
        result["reason"] = "UNIT_UNKNOWN_OR_MISMATCH"
        return result
    if len(references) == 2 and any(operand["unit"] for operand in references) and left["unit"] != right["unit"]:
        result["reason"] = "UNIT_UNKNOWN_OR_MISMATCH"
        return result
    target = right["value"]
    transform = expression["target"]
    if "scale" in transform or "offset" in transform:
        number = _number(target)
        if number is None:
            result["reason"] = "NONNUMERIC_TRANSFORM"
            return result
        target = number * _number(transform.get("scale", 1)) + _number(transform.get("offset", 0))
        result["transformed_target_fraction"] = _fraction_text(target)
    verdict = _compare(left["value"], expression["operator"], target, transform.get("tolerance"))
    result["status"] = _status(verdict)
    if verdict is None:
        result["reason"] = "UNSUPPORTED_OPERAND_TYPE"
    return result


def evaluate_expression(expression, facts):
    """Evaluate an AST against a FactStore and return a provenance-rich trace."""
    validate_expression(expression)
    if not isinstance(facts, FactStore):
        raise TypeError("Evaluator requires its independent FactStore")
    return _evaluate(expression, facts, "$")


def _build_case(case):
    if not isinstance(case, dict) or set(case) != {"case_id", "domain", "events", "options", "query", "format"}:
        raise ValueError("Expected raw task input fields only")
    if not isinstance(case["case_id"], str) or not case["case_id"] or not isinstance(case["query"], str):
        raise ValueError("Invalid case identity or current query")
    _json(case)
    if not isinstance(case["events"], list) or not case["events"]:
        raise ValueError("At least one causal event is required")
    events, facts, last = {}, FactStore(), -1
    for event in case["events"]:
        if not isinstance(event, dict) or not {"event_id", "step", "kind", "content"}.issubset(event):
            raise ValueError("Malformed source event")
        event_id, step = event["event_id"], event["step"]
        if not isinstance(event_id, str) or not event_id or event_id in events or type(step) is not int or step < 0 or step < last:
            raise ValueError("Events need unique IDs and nondecreasing causal steps")
        if event["kind"] not in {"user", "observation"}:
            raise ValueError("Runtime cannot supply interpreted semantic contracts")
        if event["kind"] == "user" and (not isinstance(event["content"], str) or event.get("facts")):
            raise ValueError("User sources must be raw text, not typed facts")
        last, events[event_id] = step, event
        declarations = event.get("facts", [])
        if not isinstance(declarations, list):
            raise ValueError("Fact declarations must be a list")
        for fact in declarations:
            if not isinstance(fact, dict) or not {"ref", "pointer"}.issubset(fact) or set(fact) - {"ref", "pointer", "unit", "entity_ref"}:
                raise ValueError("Malformed observed fact declaration")
            value = _pointer(event["content"], fact["pointer"])
            facts.append(fact["ref"], value, fact.get("unit"), step,
                         {"event_id": event_id, "pointer": fact["pointer"], "event_sha256": _hash(event["content"])}, fact.get("entity_ref"))
    # Reproduce the public source identity convention, without importing runtime.
    query_id = "current-query-" + hashlib.sha256((case["case_id"] + case["query"]).encode()).hexdigest()
    if query_id in events:
        raise ValueError("Current-query source identity collision")
    events[query_id] = {"event_id": query_id, "step": last + 1, "kind": "user", "content": case["query"]}
    if not isinstance(case["options"], list) or not case["options"]:
        raise ValueError("At least one offered option is required")
    options = {}
    for option in case["options"]:
        if not isinstance(option, dict) or set(option) != {"option_id", "event_id", "pointer", "properties"}:
            raise ValueError("Malformed offered-option declaration")
        identity = option["option_id"]
        if not isinstance(identity, str) or not identity or identity in options:
            raise ValueError("Offered identities must be unique and nonempty")
        source = events.get(option["event_id"])
        if source is None or source["kind"] != "observation":
            raise ValueError("Offer requires a causal observation source")
        row = _pointer(source["content"], option["pointer"])
        if not isinstance(row, dict) or row.get("option_id") != identity:
            raise ValueError("Offered identity differs from exact observed row")
        parent = _pointer_tokens(option["pointer"])
        if not isinstance(option["properties"], list):
            raise ValueError("Offered properties must be a list")
        properties = {}
        for field in option["properties"]:
            if not isinstance(field, dict) or not {"ref", "pointer"}.issubset(field) or set(field) - {"ref", "pointer", "unit"}:
                raise ValueError("Malformed offered-property declaration")
            reference = field["ref"]
            if not isinstance(reference, str) or not reference or reference in properties or reference in facts.histories:
                raise ValueError("Offered property reference collision")
            tokens = _pointer_tokens(field["pointer"])
            if len(tokens) <= len(parent) or tokens[:len(parent)] != parent:
                raise ValueError("Offered property escapes its exact selected row")
            value = _pointer(source["content"], field["pointer"])
            unit = field.get("unit")
            if unit is not None and (not isinstance(unit, str) or not unit):
                raise ValueError("Invalid offered property unit")
            properties[reference] = {"value": value, "unit": unit,
                                     "source": {"event_id": source["event_id"], "pointer": field["pointer"], "event_sha256": _hash(source["content"])}}
        options[identity] = {"declaration": option, "properties": properties, "row": row}
    return events, facts, options, last + 2


def _annotation_errors(annotation, events, options):
    errors = []
    if not isinstance(annotation, dict) or not {"case_id", "semantic_predicates"}.issubset(annotation):
        return ["Annotation needs case_id and semantic_predicates"]
    predicates = annotation["semantic_predicates"]
    if not isinstance(predicates, list) or not predicates or len(predicates) > 128:
        return ["Annotation requires nonempty bounded semantic predicates"]
    ids = []
    for index, predicate in enumerate(predicates):
        prefix = "predicate " + str(index)
        if not isinstance(predicate, dict) or not {"predicate_id", "source", "expression"}.issubset(predicate):
            errors.append(prefix + ": missing predicate fields")
            continue
        identity = predicate["predicate_id"]
        if not isinstance(identity, str) or not identity or identity in ids:
            errors.append(prefix + ": nonunique predicate ID")
        ids.append(identity)
        try:
            source = predicate["source"]
            if not isinstance(source, dict) or set(source) != {"event_id", "span", "quote"}:
                raise ValueError("source requires exact event, span and quote")
            event = events.get(source["event_id"])
            span = source["span"]
            if event is None or event["kind"] != "user" or not isinstance(event["content"], str):
                raise ValueError("semantic provenance must cite raw user text")
            if not isinstance(span, dict) or set(span) != {"start", "end"} or type(span["start"]) is not int or type(span["end"]) is not int:
                raise ValueError("source span requires exact integer boundaries")
            if not 0 <= span["start"] < span["end"] <= len(event["content"]):
                raise ValueError("source span outside raw text")
            if not isinstance(source["quote"], str) or event["content"][span["start"]:span["end"]] != source["quote"]:
                raise ValueError("source quote differs from exact raw text span")
            # The explicit span disambiguates repeated text; quote uniqueness is
            # not required for an offline annotation with a verified exact span.
        except (KeyError, TypeError, ValueError) as error:
            errors.append(prefix + ": " + str(error))
        try:
            validate_expression(predicate["expression"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(prefix + ": " + str(error))
    # Expected labels can be omitted for unseen scored cases. When present,
    # validate their shape; their truth values remain separate cross-checks.
    expected = annotation.get("expected_options")
    if expected is not None:
        if not isinstance(expected, dict) or set(expected) != set(options):
            errors.append("Expected-option map must cover exactly the observed offered IDs")
        else:
            for identity, value in expected.items():
                if not isinstance(value, dict) or value.get("status") not in STATUSES or not isinstance(value.get("predicate_statuses"), dict):
                    errors.append("Malformed expected-option label: " + identity)
                elif set(value["predicate_statuses"]) != set(ids) or any(status not in STATUSES for status in value["predicate_statuses"].values()):
                    errors.append("Expected predicate labels must cover exact criterion IDs: " + identity)
    decision = annotation.get("expected_decision")
    if decision is not None and (not isinstance(decision, dict) or set(decision) != {"option_ids", "abstain", "abstention_reason"}
                                 or not isinstance(decision["option_ids"], list) or type(decision["abstain"]) is not bool
                                 or decision["abstention_reason"] not in {None, "infeasible", "unverified"}
                                 or any(identity not in options for identity in decision["option_ids"])
                                 or len(decision["option_ids"]) != len(set(decision["option_ids"]))):
        errors.append("Malformed expected-decision label")
    return errors


def validate_annotation(case, annotation):
    """Validate raw sources and AST shape; this does not certify NL meaning."""
    try:
        events, _, options, _ = _build_case(case)
        errors = _annotation_errors(annotation, events, options)
        if not isinstance(annotation, dict) or annotation.get("case_id") != case["case_id"]:
            errors.append("Annotation case identity mismatch")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        errors = ["Invalid source case: " + str(error)]
    return {"valid": not errors, "errors": errors}


def _score_option(option_id, option, observed, predicates, action_step, events):
    # A fresh store for each counterfactual candidate prevents chosen-field
    # values from another offer or prior score leaking into this one.
    facts = copy.deepcopy(observed)
    for reference, field in option["properties"].items():
        facts.append(reference, field["value"], field["unit"], action_step, field["source"])
    criteria = []
    for predicate in predicates:
        trace = evaluate_expression(predicate["expression"], facts)
        provenance = {**copy.deepcopy(predicate["source"]),
                      "event_sha256": _hash(events[predicate["source"]["event_id"]]["content"])}
        criteria.append({"predicate_id": predicate["predicate_id"], "status": trace["status"],
                         "source": provenance, "expression_evidence": trace})
    return {"option_id": option_id, "status": _all([row["status"] for row in criteria]), "criteria": criteria}


def _check_labels(annotation, option_results, feasible, abstention_reason):
    discrepancies = []
    expected = annotation.get("expected_options")
    if expected is not None:
        for row in option_results:
            identity, actual = row["option_id"], {criterion["predicate_id"]: criterion["status"] for criterion in row["criteria"]}
            label = expected[identity]
            if label["status"] != row["status"] or label["predicate_statuses"] != actual:
                discrepancies.append({"option_id": identity, "expected_status": label["status"], "computed_status": row["status"],
                                      "expected_predicate_statuses": label["predicate_statuses"], "computed_predicate_statuses": actual})
    decision = annotation.get("expected_decision")
    actual_decision = {"option_ids": feasible, "abstain": not feasible, "abstention_reason": abstention_reason}
    if decision is not None and (set(decision["option_ids"]) != set(feasible) or decision["abstain"] != bool(not feasible)
                                 or decision["abstention_reason"] != abstention_reason):
        discrepancies.append({"expected_decision": decision, "computed_decision": actual_decision})
    return {"present": expected is not None or decision is not None, "consistent": not discrepancies,
            "discrepancies": discrepancies, "used_to_derive_verdict": False}


def grade_case(case, annotation, response):
    """Grade a committed raw response or commit_option metadata, offline.

    Supplied selected_facts/status/task_success metadata never establishes the
    answer: the offered identity is grounded afresh against the observed row.
    All criterion traces and expected-map cross-checks are post-commit only.
    """
    base = {"case_id": case.get("case_id") if isinstance(case, dict) else None, "passed": False,
            "grading_valid": False, "criteria": [], "selected_option_id": None}
    try:
        events, observed, options, action_step = _build_case(case)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {**base, "status": "INVALID_CASE", "error": str(error)}
    validation = validate_annotation(case, annotation)
    if not validation["valid"]:
        return {**base, "status": "INVALID_ANNOTATION", "annotation_validation": validation}
    # This is public runtime output metadata, not trusted evidence. Extra raw
    # response fields remain invalid; only an explicit commit wrapper unwraps.
    actual = response.get("actual_response") if isinstance(response, dict) and "status" in response and "actual_response" in response else response
    if not isinstance(actual, dict) or set(actual) != {"option_id"} or actual["option_id"] is not None and (not isinstance(actual["option_id"], str) or not actual["option_id"]):
        return {**base, "status": "INVALID_SUBMISSION", "annotation_validation": validation}
    selected = actual["option_id"]
    if selected is not None and selected not in options:
        return {**base, "status": "UNKNOWN_OPTION", "selected_option_id": selected, "annotation_validation": validation}
    option_results = [_score_option(identity, option, observed, annotation["semantic_predicates"], action_step, events) for identity, option in options.items()]
    feasible = [row["option_id"] for row in option_results if row["status"] == SATISFIED]
    abstention_reason = None if feasible else "unverified" if any(row["status"] == UNVERIFIED for row in option_results) else "infeasible"
    labels = _check_labels(annotation, option_results, feasible, abstention_reason)
    common = {**base, "grading_valid": True, "selected_option_id": selected, "annotation_validation": validation,
              "option_results": option_results, "feasible_option_ids": feasible, "abstention_reason": abstention_reason,
              "expected_labels_check": labels,
              "grounding": "observed source rows selected by declared offered identity; caller-supplied selected facts ignored"}
    if selected is None:
        return {**common, "status": "ABSTAINED", "passed": not feasible,
                "decision_status": SATISFIED if not feasible else VIOLATED}
    selected_result = next(row for row in option_results if row["option_id"] == selected)
    return {**common, "status": "EVALUATED", "decision_status": selected_result["status"],
            "passed": selected_result["status"] == SATISFIED, "criteria": selected_result["criteria"]}
