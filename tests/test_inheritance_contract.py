"""Declared inheritance preserves actual evidence and fails open on uncertainty."""
import copy

import pytest

from constraint_memory.memory.inheritance import (
    compile_inheritance_obligations, declare_inheritance_contract, validate_inheritance,
)
from constraint_memory.memory.subject_scope import compile_scope_coverage


def inputs():
    contract = declare_inheritance_contract("public-base-v1", "g", "Keep unmodified base slots.",
        ["event:contract"], enabled=True)
    base = [{"group_id": "g", "traveler": "Base", "day": 1, "slot": slot,
             "value": value, "version": 1, "created_at": 0, "status": "COMMITTED",
             "source_refs": ["event:base"]}
            for slot, value in (("breakfast", "Base Cafe, Boston"), ("dinner", "Base Grill, Boston"))]
    request = {"group_id": "g", "traveler": "Current", "step": 1}
    parsed = {"parse_success": True, "interpretation_complete": True, "constraints": [], "unresolved": []}
    return contract, base, request, parsed


def test_absent_or_disabled_contract_creates_no_requirements():
    contract, base, request, parsed = inputs()
    for value in (None, {**contract, "enabled": False}):
        result = compile_inheritance_obligations(value, base, request, parsed)
        assert result["status"] == "DISABLED"
        assert result["constraints"] == []
        assert not result["hard_checks_enabled"]


def test_valid_json_without_independent_coverage_is_only_advisory():
    contract, base, request, parsed = inputs()
    del parsed["interpretation_complete"]
    result = compile_inheritance_obligations(contract, base, request, parsed)
    assert result["status"] == "UNCERTAIN"
    assert not result["hard_checks_enabled"]
    assert len(result["constraints"]) == 2
    assert all(row["status"] == "UNRESOLVED" for row in result["constraints"])
    assert validate_inheritance(result, [])["status"] == "UNVERIFIED"


def test_declared_obligations_group_by_day_and_keep_exact_sources():
    contract, base, request, parsed = inputs()
    result = compile_inheritance_obligations(contract, base, request, parsed)
    assert result["status"] == "ACTIVE"
    assert len(result["obligations"]) == 1
    assert len(result["obligations"][0]["slots"]) == 2
    for row in result["constraints"]:
        assert row["source_kind"] == "TASK_CONTRACT"
        assert row["source_refs"] == ["event:contract", "event:base"]
        assert row["base_decision_ref"].endswith("@v1")


@pytest.mark.parametrize("attribute", ["value", "price", "rating", "cuisine"])
def test_any_explicit_slot_constraint_exempts_that_slot(attribute):
    contract, base, request, parsed = inputs()
    parsed["constraints"] = [{"constraint_id": "change", "subject": {
        "traveler": "Current", "day": 1, "slot": "dinner", "attribute": attribute},
        "source_refs": ["event:request"]}]
    result = compile_inheritance_obligations(contract, base, request, parsed)
    assert [row["subject"]["slot"] for row in result["constraints"]] == ["breakfast"]
    assert result["exceptions"][0]["slot"] == "dinner"


@pytest.mark.parametrize("change", ["parser_failure", "unscoped_uncertainty", "malformed_constraint"])
def test_uncertain_request_never_freezes_legitimate_unparsed_changes(change):
    contract, base, request, parsed = inputs()
    if change == "parser_failure":
        parsed["parse_success"] = False
    elif change == "unscoped_uncertainty":
        parsed["unresolved"] = [{"reason": "Unknown changed slot"}]
    else:
        parsed["constraints"] = [{"subject": {"traveler": "Current"}}]
    result = compile_inheritance_obligations(contract, base, request, parsed)
    assert not result["hard_checks_enabled"]
    assert validate_inheritance(result, [])["violations"] == []


def test_validation_reports_actual_missing_changed_slots_without_repair_or_success_claim():
    contract, base, request, parsed = inputs()
    compiled = compile_inheritance_obligations(contract, base, request, parsed)
    actual = [{"traveler": "Current", "day": 1, "slot": "dinner", "value": "Different Grill, Boston"}]
    before = copy.deepcopy((compiled, actual))
    result = validate_inheritance(compiled, actual)
    assert result["status"] == "VIOLATED"
    assert {row["code"] for row in result["violations"]} == {"INHERITED_SLOT_CHANGED", "INHERITED_SLOT_MISSING"}
    assert result["task_success"] is None
    assert (compiled, actual) == before


def test_base_version_and_visibility_do_not_use_future_choice():
    contract, base, request, parsed = inputs()
    base += [{**base[1], "version": 2, "created_at": 3, "value": "Future Grill, Boston"}]
    compiled = compile_inheritance_obligations(contract, base, request, parsed)
    row = next(row for row in compiled["constraints"] if row["subject"]["slot"] == "dinner")
    assert row["target"]["value"] == "Base Grill, Boston"
    assert row["base_decision_ref"].endswith("@v1")


def test_mismatched_group_and_falsely_quoted_user_contract_are_rejected():
    contract, base, request, parsed = inputs()
    for changed in ({**contract, "group_id": "other"}, {**contract, "source_kind": "USER_REQUEST"}):
        with pytest.raises(ValueError):
            compile_inheritance_obligations(changed, base, request, parsed)


def scoped_inputs(query="For dinner on day1, I want a restaurant that bans smoking."):
    contract, base, request, _ = inputs()
    request.update(query=query, prior_travelers=["Base", "Benjamin"], source_refs=["event:g:1"])
    certificate = compile_scope_coverage(request, request["prior_travelers"], request["source_refs"])
    return contract, base, request, certificate


def test_independent_scope_enables_only_unchanged_slot_checks_without_semantic_interpretation():
    contract, base, request, certificate = scoped_inputs()
    advisory = {"parse_success": True, "constraints": [], "unresolved": [{"reason": "Smoking policy unsupported"}]}
    result = compile_inheritance_obligations(contract, base, request, advisory, scope_coverage=certificate)
    assert result["hard_checks_enabled"]
    assert result["scope_coverage_verified"]
    assert not result["semantic_interpretation_verified"]
    assert result["exception_basis"] == "INDEPENDENT_REQUEST_SCOPE"
    assert [row["subject"]["slot"] for row in result["constraints"]] == ["breakfast"]
    exception = result["exceptions"][0]
    assert exception["slot"] == "dinner"
    assert exception["source_kind"] == "REQUEST_SCOPE"
    assert exception["reasons"] == ["INDEPENDENT_CURRENT_SCOPE"]
    assert exception["scope_evidence"][0]["evidence_quote"] == request["query"]
    assert result["advisory_diagnostics"][0]["code"] == "ADVISORY_REQUEST_SEMANTICS_UNRESOLVED"
    assert "interpretation_complete" not in advisory


@pytest.mark.parametrize("advisory", [None, {}, {"parse_success": False}, {
    "parse_success": True, "interpretation_complete": True,
    "constraints": [{"subject": {"traveler": "Current", "day": 1, "slot": "breakfast"}, "source_refs": ["event:wrong"]}],
    "unresolved": [{"subject": {"traveler": "Current", "day": 1, "slot": "breakfast"}, "source_refs": ["event:wrong"]}],
}, {"group_id": "other", "parse_success": True}])
def test_advisory_writer_cannot_override_independent_changed_subjects(advisory):
    contract, base, request, certificate = scoped_inputs()
    result = compile_inheritance_obligations(contract, base, request, advisory, scope_coverage=certificate)
    assert result["hard_checks_enabled"]
    assert {(row["day"], row["slot"]) for row in result["exceptions"]} == {(1, "dinner")}
    actual = [{"traveler": "Current", "day": 1, "slot": "breakfast", "value": "Different Cafe, Boston"}]
    checked = validate_inheritance(result, actual)
    assert checked["checked_slot_count"] == 1
    assert checked["violations"][0]["code"] == "INHERITED_SLOT_CHANGED"
    assert checked["task_success"] is None


def test_named_target_day_slot_does_not_become_an_inheritance_exception():
    contract, base, request, certificate = scoped_inputs("For dinner on the third day, I want a place rated above Benjamin's first-day lunch.")
    base.extend([{**base[1], "day": 3, "value": "Day3 Grill"}, {**base[0], "slot": "lunch", "value": "Day1 Lunch"}])
    result = compile_inheritance_obligations(contract, base, request, scope_coverage=certificate)
    assert result["hard_checks_enabled"]
    assert {(row["day"], row["slot"]) for row in result["exceptions"]} == {(3, "dinner")}
    assert (1, "lunch") in {(row["subject"]["day"], row["subject"]["slot"]) for row in result["constraints"]}


@pytest.mark.parametrize("field,value", [
    ("coverage_complete", False), ("group_id", "other"), ("traveler", "Other"),
    ("step", 9), ("step", True), ("source_kind", "TASK_CONTRACT"),
    ("request_sha256", "0" * 64), ("source_refs", ["event:other"]),
    ("prior_travelers", ["Future"]), ("extra_field", "not generated"),
    ("gold_plan", []),
])
def test_any_altered_certificate_identity_flag_source_or_extra_field_disables_hard_checks(field, value):
    contract, base, request, certificate = scoped_inputs()
    certificate[field] = value
    legacy_assertion = {"parse_success": True, "interpretation_complete": True, "constraints": [], "unresolved": []}
    result = compile_inheritance_obligations(contract, base, request, legacy_assertion, scope_coverage=certificate)
    assert not result["hard_checks_enabled"]
    assert not result["scope_coverage_verified"]
    assert any(row["code"] == "INDEPENDENT_SCOPE_CERTIFICATE_MISMATCH" for row in result["diagnostics"])
    assert all(not row["hard_check"] for row in result["constraints"])


@pytest.mark.parametrize("change", ["span", "quote", "cue", "scope_subject", "scope_sources", "metadata"])
def test_exact_scope_quotes_spans_cues_and_subjects_are_recomputed(change):
    contract, base, request, certificate = scoped_inputs("I am Current.\nFor dinner on day1, I want a cheap place.")
    scope = certificate["scopes"][0]
    if change == "span":
        scope["evidence_span"]["start"] += 1
    elif change == "quote":
        scope["evidence_quote"] = "Different request"
    elif change == "cue":
        scope["scope_cues"][0]["start"] += 1
    elif change == "scope_subject":
        scope["subject"]["slot"] = "breakfast"
    elif change == "scope_sources":
        scope["source_refs"] = ["event:other"]
    else:
        certificate["ignored_metadata"][0]["evidence"]["quote"] = "I am Other."
    result = compile_inheritance_obligations(contract, base, request, scope_coverage=certificate)
    assert not result["hard_checks_enabled"]
    assert not result["scope_coverage_verified"]


@pytest.mark.parametrize("missing", ["prior_travelers", "source_refs", "query"])
def test_caller_must_declare_visible_inputs_independently_of_the_certificate(missing):
    contract, base, request, certificate = scoped_inputs()
    del request[missing]
    result = compile_inheritance_obligations(contract, base, request, scope_coverage=certificate)
    assert not result["hard_checks_enabled"]
    assert not result["scope_coverage_verified"]


def test_incomplete_or_forged_complete_scope_never_uses_legacy_completeness_assertion():
    contract, base, request, certificate = scoped_inputs("For dinner on day1, I want a cheap place.\nI need every other meal to be vegetarian.")
    advisory = {"parse_success": True, "interpretation_complete": True, "constraints": [], "unresolved": []}
    valid = compile_inheritance_obligations(contract, base, request, advisory, scope_coverage=certificate)
    assert valid["scope_coverage_verified"]
    assert not valid["hard_checks_enabled"]
    certificate["coverage_complete"] = True
    forged = compile_inheritance_obligations(contract, base, request, advisory, scope_coverage=certificate)
    assert not forged["scope_coverage_verified"]
    assert not forged["hard_checks_enabled"]


def test_request_change_after_certificate_compilation_is_uncertain_and_inputs_unchanged():
    contract, base, request, certificate = scoped_inputs()
    request["query"] = "For breakfast on day1, I want a cheap place."
    before = copy.deepcopy((contract, base, request, certificate))
    result = compile_inheritance_obligations(contract, base, request, scope_coverage=certificate)
    assert not result["hard_checks_enabled"]
    assert {(row["day"], row["slot"]) for row in result["exceptions"]} == {(1, "breakfast")}
    assert (contract, base, request, certificate) == before


@pytest.mark.parametrize("descriptor", [
    {"group_id": "g", "traveler": "Future", "created_at": 2},
    {"group_id": "other", "traveler": "Future", "created_at": 0},
])
def test_future_or_other_group_visible_declarations_cannot_enable_inheritance(descriptor):
    contract, base, request, certificate = scoped_inputs()
    request["prior_travelers"] = [descriptor]
    result = compile_inheritance_obligations(contract, base, request, scope_coverage=certificate)
    assert not result["hard_checks_enabled"]
