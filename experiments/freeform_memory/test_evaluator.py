"""Independent evaluator regression tests; no runtime/core or model imports."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("independent_freeform_evaluator", Path(__file__).with_name("evaluator.py"))
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)
ROOT = Path(__file__).resolve().parents[2]


def atom(reference="chosen.value", operator="EQ", target=None, unit=None):
    result = {"subject": {"ref": reference}, "operator": operator,
              "target": {"value": True} if target is None else target}
    if unit is not None:
        result["unit"] = unit
    return result


def store(value=True, unit=None):
    facts = ev.FactStore()
    facts.append("chosen.value", value, unit, 2, {"event_id": "offers", "pointer": "/0/value", "event_sha256": "test"})
    return facts


def fixture():
    case = {"case_id": "opaque-task", "domain": "opaque-domain", "format": "messages", "query": "Choose now.",
            "events": [
                {"event_id": "instruction", "step": 0, "kind": "user", "content": "Pick a true value."},
                {"event_id": "offers", "step": 1, "kind": "observation", "content": [
                    {"option_id": "bad", "value": False}, {"option_id": "good", "value": True}]},
            ],
            "options": [{"option_id": identity, "event_id": "offers", "pointer": "/" + str(index),
                         "properties": [{"ref": "chosen.value", "pointer": "/" + str(index) + "/value"}]}
                        for index, identity in enumerate(("bad", "good"))]}
    annotation = {"case_id": case["case_id"], "semantic_predicates": [
        {"predicate_id": "truth", "source": {"event_id": "instruction", "span": {"start": 0, "end": 18},
                                               "quote": "Pick a true value."}, "expression": atom()}]}
    return case, annotation


@pytest.mark.parametrize("left,right,status", [(True, 1, ev.VIOLATED), (1, True, ev.VIOLATED),
                                               ("1", 1, ev.VIOLATED), (1.0, 1, ev.SATISFIED),
                                               ([True], [1], ev.VIOLATED), ({"a": True}, {"a": 1}, ev.VIOLATED)])
def test_equality_preserves_json_types(left, right, status):
    assert ev.evaluate_expression(atom(target={"value": right}), store(left))["status"] == status


@pytest.mark.parametrize("operator,left,right,status", [
    ("IN", True, [1], ev.VIOLATED), ("SUPERSET_OF", [1], [True], ev.VIOLATED),
    ("SUPERSET_OF", ["a", "a", "b"], ["a"], ev.SATISFIED),
    ("SUBSET_OF", ["a"], ["a", "b"], ev.SATISFIED),
    ("DISJOINT", [True], [1], ev.SATISFIED),
    ("INTERSECTS", ["1"], [1], ev.VIOLATED), ("IN", "a", "a", ev.UNVERIFIED),
])
def test_type_aware_set_operations(operator, left, right, status):
    assert ev.evaluate_expression(atom(operator=operator, target={"value": right}), store(left))["status"] == status


def test_fraction_decimal_boundary_and_transform_are_exact():
    facts = store("0.3", "unit")
    expression = atom(operator="GTE", target={"value": "0.1", "scale": 3}, unit="unit")
    trace = ev.evaluate_expression(expression, facts)
    assert trace["status"] == ev.SATISFIED
    assert trace["transformed_target_fraction"] == {"numerator": 3, "denominator": 10}
    assert ev.evaluate_expression(atom(operator="LT", target={"value": "0.300000000000000000000001"}), facts)["status"] == ev.SATISFIED
    tolerance = atom(operator="ABS_DIFF_LTE", target={"value": "0.2", "tolerance": "0.1"})
    assert ev.evaluate_expression(tolerance, facts)["status"] == ev.SATISFIED


@pytest.mark.parametrize("value", [True, "NaN", "Infinity", "1/3", "1e1000000000"])
def test_nonnumeric_or_unbounded_values_do_not_satisfy_numeric_predicates(value):
    assert ev.evaluate_expression(atom(operator="LTE", target={"value": 1}), store(value))["status"] == ev.UNVERIFIED


def test_live_snapshot_and_provenance_are_distinct():
    facts = store(8, "EUR")
    facts.append("limit", 7, "EUR", 0, {"event_id": "approval", "pointer": "/cap", "event_sha256": "first"})
    facts.append("limit", 9, "EUR", 1, {"event_id": "revision", "pointer": "/cap", "event_sha256": "second"})
    current = atom(operator="LTE", target={"ref": "limit", "binding_mode": "LIVE"}, unit="EUR")
    snapshot = atom(operator="LTE", target={"ref": "limit", "binding_mode": "SNAPSHOT", "version": 1}, unit="EUR")
    assert ev.evaluate_expression(current, facts)["status"] == ev.SATISFIED
    old = ev.evaluate_expression(snapshot, facts)
    assert old["status"] == ev.VIOLATED
    assert old["target"]["bindings"][0]["source"]["event_id"] == "approval"


def test_conflicting_same_step_is_unknown_regardless_of_insertion_order_and_snapshot():
    for values in ((3, 4), (4, 3)):
        facts = store(3)
        for value in values:
            facts.append("location", value, None, 1, {"event_id": "source-" + str(value)})
        for target in ({"ref": "location"}, {"ref": "location", "binding_mode": "SNAPSHOT", "version": 1}):
            trace = ev.evaluate_expression(atom(target=target), facts)
            assert trace["status"] == ev.UNVERIFIED
            assert trace["target"]["reason"] == "CONFLICTING_SAME_STEP_FACTS"
            assert len(trace["target"]["bindings"]) == 2
        facts.append("location", 3, None, 2, {"event_id": "reconciled"})
        assert ev.evaluate_expression(atom(target={"ref": "location"}), facts)["status"] == ev.SATISFIED


def test_same_step_identical_reports_are_usable_but_bool_number_reports_conflict():
    facts = store(True)
    facts.append("status", True, None, 1, {"event_id": "a"})
    facts.append("status", True, None, 1, {"event_id": "b"})
    assert ev.evaluate_expression(atom(target={"ref": "status"}), facts)["status"] == ev.SATISFIED
    facts.append("status", 1, None, 1, {"event_id": "c"})
    assert ev.evaluate_expression(atom(target={"ref": "status"}), facts)["status"] == ev.UNVERIFIED


def test_missing_units_and_incompatible_units_are_unknown():
    expression = atom(operator="LTE", target={"value": 10}, unit="minute")
    for unit in (None, "hour"):
        assert ev.evaluate_expression(expression, store(2, unit))["reason"] == "UNIT_UNKNOWN_OR_MISMATCH"
    facts = store(2, "minute")
    facts.append("target", 5, "hour", 0, {"event_id": "sheet"})
    assert ev.evaluate_expression(atom(operator="LTE", target={"ref": "target"}), facts)["status"] == ev.UNVERIFIED


def test_three_valued_not_all_any_and_conditional_branch_selection():
    missing = atom("missing")
    true, false = {"literal": True}, {"literal": False}
    facts = store()
    for expression, expected in [
        ({"not": missing}, ev.UNVERIFIED), ({"all": [missing, false]}, ev.VIOLATED),
        ({"any": [missing, true]}, ev.SATISFIED), ({"any": [missing, false]}, ev.UNVERIFIED),
        ({"if": true, "then": true, "else": missing}, ev.SATISFIED),
        ({"if": missing, "then": true, "else": true}, ev.SATISFIED),
        ({"if": missing, "then": false, "else": false}, ev.VIOLATED),
        ({"if": missing, "then": true, "else": false}, ev.UNVERIFIED),
    ]:
        assert ev.evaluate_expression(expression, facts)["status"] == expected
    trace = ev.evaluate_expression({"if": true, "then": true, "else": missing}, facts)
    assert trace["selected_branch"] == "then"
    assert all(child["path"] != "$.else" for child in trace["children"])


def test_entity_revision_invalidates_bound_property():
    facts = ev.FactStore()
    facts.append("selected", "one", None, 0, {"event_id": "a"})
    facts.append("property", True, None, 1, {"event_id": "b"}, {"ref": "selected", "version": 1})
    assert ev.evaluate_expression(atom("property"), facts)["status"] == ev.SATISFIED
    facts.append("selected", "two", None, 2, {"event_id": "c"})
    trace = ev.evaluate_expression(atom("property"), facts)
    assert trace["status"] == ev.UNVERIFIED
    assert trace["subject"]["reason"] == "STALE_OR_UNKNOWN_ENTITY"


def test_entity_cannot_be_bound_to_unknown_or_future_evidence():
    facts = ev.FactStore()
    with pytest.raises(ValueError, match="causal evidence"):
        facts.append("property", True, None, 0, {"event_id": "p"}, {"ref": "future", "version": 1})
    facts.append("future", "x", None, 3, {"event_id": "f"})
    with pytest.raises(ValueError, match="causal evidence"):
        facts.append("property", True, None, 2, {"event_id": "p"}, {"ref": "future", "version": 1})


def test_grade_ignores_forged_commit_values_and_pass_flags():
    case, annotation = fixture()
    forged = {"status": "COMMITTED_UNVERIFIED", "actual_response": {"option_id": "bad"},
              "selected_facts": {"chosen.value": {"value": True}}, "passed": True, "task_success": True}
    result = ev.grade_case(case, annotation, forged)
    assert result["status"] == "EVALUATED" and not result["passed"]
    assert result["criteria"][0]["expression_evidence"]["subject"]["value"] is False
    assert result["criteria"][0]["expression_evidence"]["subject"]["bindings"][0]["source"]["pointer"] == "/0/value"
    assert ev.grade_case(case, annotation, {"option_id": "good"})["passed"]
    assert not ev.grade_case(case, annotation, {"option_id": None})["passed"]


def test_source_identity_and_row_boundaries_cannot_be_forged():
    case, annotation = fixture()
    case["options"][0]["properties"][0]["pointer"] = "/1/value"
    assert ev.grade_case(case, annotation, {"option_id": "bad"})["status"] == "INVALID_CASE"
    case, annotation = fixture()
    case["events"][1]["content"][0]["option_id"] = "pretend"
    assert ev.grade_case(case, annotation, {"option_id": "bad"})["status"] == "INVALID_CASE"


def test_annotation_quotes_ast_shape_and_snapshot_metadata_are_validated():
    case, annotation = fixture()
    annotation["semantic_predicates"][0]["source"]["quote"] = "different"
    assert ev.grade_case(case, annotation, {"option_id": "good"})["status"] == "INVALID_ANNOTATION"
    for expression in ({"all": []}, {"literal": 1}, atom(target={"ref": "cap", "version": 1}),
                       atom(target={"ref": "cap", "binding_mode": "SNAPSHOT"}), {"not": atom(), "all": [atom()]}):
        with pytest.raises(ValueError):
            ev.validate_expression(expression)


def test_expected_labels_never_drive_source_derived_grades():
    case, annotation = fixture()
    annotation["expected_options"] = {
        "bad": {"status": ev.SATISFIED, "predicate_statuses": {"truth": ev.SATISFIED}},
        "good": {"status": ev.VIOLATED, "predicate_statuses": {"truth": ev.VIOLATED}},
    }
    result = ev.grade_case(case, annotation, {"option_id": "good"})
    assert result["passed"] and result["grading_valid"]
    assert not result["expected_labels_check"]["consistent"]
    assert not result["expected_labels_check"]["used_to_derive_verdict"]


@pytest.mark.parametrize("response,status", [(None, "INVALID_SUBMISSION"), ({"option_id": "unknown"}, "UNKNOWN_OPTION"),
                                               ({"option_id": True}, "INVALID_SUBMISSION"),
                                               ({"option_id": "good", "value": True}, "INVALID_SUBMISSION")])
def test_submission_protocol(response, status):
    case, annotation = fixture()
    assert ev.grade_case(case, annotation, response)["status"] == status


def test_all_author_cases_and_all_offers_match_independent_labels_without_mutation():
    cases = json.loads((ROOT / "data/transfer/freeform_development.json").read_text(encoding="utf-8"))
    annotations = json.loads((ROOT / "data/transfer/freeform_annotations.json").read_text(encoding="utf-8"))["cases"]
    annotation_map = {row["case_id"]: row for row in annotations}
    pristine = copy.deepcopy((cases, annotations))
    offers = criteria = 0
    for case in cases:
        annotation = annotation_map[case["case_id"]]
        result = ev.grade_case(case, annotation, {"option_id": None})
        assert result["grading_valid"] and result["expected_labels_check"]["consistent"]
        assert result["passed"] == annotation["expected_decision"]["abstain"]
        for option in case["options"]:
            selected = ev.grade_case(case, annotation, {"option_id": option["option_id"]})
            assert selected["expected_labels_check"]["consistent"]
            assert selected["passed"] == (annotation["expected_options"][option["option_id"]]["status"] == ev.SATISFIED)
        offers += len(case["options"])
        criteria += len(annotation["semantic_predicates"])
    assert (len(cases), offers, criteria) == (10, 35, 32)
    assert (cases, annotations) == pristine


def test_evaluator_has_no_runtime_or_core_imports():
    import ast
    tree = ast.parse(Path(__file__).with_name("evaluator.py").read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert all("constraint_memory" not in name and "runtime" not in name for name in imports)
