"""Independent mechanical checks with opaque identities and unfamiliar JSON shapes.

These use scripted interpretations only. They do not score language understanding,
certify source semantics, load author annotations, or issue model calls.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from experiments.freeform_memory.runtime import (
    build_observed,
    commit_option,
    encoded,
    ingest_interpretation,
    planner_messages,
    prepare_context,
    writer_messages,
)


FACT = "basis:ι/3~r"
OUTPUT = "out:δ/7~r"
ROW_A = "/inventory/edge~1~0A"
ROW_B = "/inventory/edge~1~0B"
OPTION_A = "opaque:Ω/7~q"
OPTION_B = "opaque:Ж/9~m"
UNIT = "µs"
USER_ID = "message:ξ/#"
QUOTE = "Use the latest bound for the selected channel"


def opaque_case():
    """No ordinal traveler/day/slot names or option-name semantics are supplied."""
    rows = {
        "edge/~A": {"option_id": OPTION_A, "payload": {"s/~q": {"value": 11, "unit": UNIT}}, "tags": ["x", "y"]},
        "edge/~B": {"option_id": OPTION_B, "payload": {"s/~q": {"value": 18, "unit": UNIT}}, "tags": ["z"]},
    }
    return {
        "case_id": "project:κ/19~",
        "domain": "signal-routing",
        "format": "ticket",
        "query": "Choose a current identity only if the preserved requirements justify it.",
        "events": [
            {"event_id": "sample:old/~", "kind": "observation", "step": 0,
             "content": {"settings": {"bound": {"value": 20, "unit": UNIT}}},
             "facts": [{"ref": FACT, "pointer": "/settings/bound/value", "unit": UNIT}]},
            {"event_id": USER_ID, "kind": "user", "step": 1,
             "content": QUOTE + ". If the phase witness is absent, wait; do not infer it from a low number."},
            {"event_id": "sample:new/~", "kind": "observation", "step": 2,
             "content": {"replacement": [14, {"opaque": True}]},
             "facts": [{"ref": FACT, "pointer": "/replacement/0", "unit": UNIT}]},
            {"event_id": "catalogue:α/~", "kind": "observation", "step": 3,
             "content": {"inventory": rows}},
            {"event_id": "message:amendment/~", "kind": "user", "step": 4,
             "content": "A threshold alone is insufficient: the phase witness still applies."},
        ],
        "options": [
            {"option_id": identity, "event_id": "catalogue:α/~", "pointer": pointer,
             "properties": [{"ref": OUTPUT, "pointer": pointer + "/payload/s~1~0q/value", "unit": UNIT}]}
            for identity, pointer in ((OPTION_A, ROW_A), (OPTION_B, ROW_B))
        ],
    }


def advisory(*, target=None, operator="LTE", quote=QUOTE):
    return {
        "constraints": [{"source_event_id": USER_ID, "evidence_quote": quote,
                         "subject_ref": OUTPUT, "operator": operator,
                         "target": {"ref": FACT} if target is None else target, "unit": UNIT}],
        "unresolved": [],
    }


def contexts(case, response=None):
    flat = prepare_context(case, "FACT_LEDGER", len, 100000)
    interpreted = prepare_context(case, "ADVISORY_IR", len, 100000, advisory() if response is None else response)
    return flat, interpreted


def test_identical_core_evidence_is_preserved_with_opaque_refs_nested_rows_and_amendments():
    case = opaque_case()
    original = copy.deepcopy(case)
    flat, interpreted = contexts(case)
    for section in ("RAW_USER_REQUIREMENTS", "CURRENT_FACTS", "UNKNOWN_FACTS", "FACT_REVISIONS"):
        assert flat["sections"][section] == interpreted["sections"][section]
    assert [x["value"] for x in flat["sections"]["FACT_REVISIONS"][FACT]] == [20, 14]
    assert flat["sections"]["CURRENT_FACTS"][0]["reference"] == FACT
    assert flat["sections"]["CURRENT_FACTS"][0]["value"] == 14
    assert case == original


def test_unicode_quote_spans_keep_raw_source_and_model_provenance_separate():
    case = opaque_case()
    case["events"][1]["content"] = "λ→ " + case["events"][1]["content"]
    memory, step, _ = build_observed(case)
    parsed = ingest_interpretation(memory, case, advisory(), step + 1)
    assert parsed["rejected"] == []
    record = memory.resolve(parsed["references"][0], step + 1)
    span = record["source"]["span"]
    assert case["events"][1]["content"][span["start"]:span["end"]] == QUOTE
    assert record["source"]["event_id"] == USER_ID
    assert record["interpretation_source"]["event_id"] != USER_ID
    assert memory.events[record["interpretation_source"]["event_id"]]["kind"] == "model_output"
    assert record["semantic_status"] == "UNVERIFIED"
    assert memory.check(record["reference"], step + 1)["reason"] == "SEMANTICS_NOT_VERIFIED"


def test_structurally_valid_but_wrong_predicate_never_certifies_the_compound_requirement():
    flat, interpreted = contexts(opaque_case(), advisory(target={"value": "invented-semantic-answer"}, operator="EQUALS"))
    assert interpreted["sections"]["RAW_USER_REQUIREMENTS"] == flat["sections"]["RAW_USER_REQUIREMENTS"]
    assert interpreted["hard_semantic_checks"] is False
    assert interpreted["task_success"] is None
    assert interpreted["sections"]["ADVISORY_CONSTRAINTS"][0]["semantic_status"] == "UNVERIFIED"
    assert any(row["reason"] == "SEMANTICS_NOT_VERIFIED" for row in interpreted["sections"]["INTERPRETATION_UNKNOWN"])
    assert not interpreted["source_snapshot"]["check_history"]


@pytest.mark.parametrize("source,quote", [
    ("catalogue:α/~", QUOTE), ("future:unseen", QUOTE), (USER_ID, "quote that never appeared"),
])
def test_rejected_source_cannot_become_a_constraint_or_hide_raw_requirements(source, quote):
    case = opaque_case()
    response = advisory(quote=quote)
    response["constraints"][0]["source_event_id"] = source
    flat, interpreted = contexts(case, response)
    assert interpreted["interpretation"]["rejected"]
    assert interpreted["sections"]["ADVISORY_CONSTRAINTS"] == []
    assert interpreted["source_snapshot"]["constraints"] == {}
    for name in flat["sections"]:
        assert interpreted["sections"][name] == flat["sections"][name]


def test_nonunique_exact_quote_is_rejected_without_semantic_guessing():
    case = opaque_case()
    case["events"][1]["content"] = QUOTE + "; original note: " + QUOTE
    _, interpreted = contexts(case)
    assert interpreted["interpretation"]["rejected"]
    assert interpreted["source_snapshot"]["constraints"] == {}


def test_unknown_clause_and_model_reason_remain_visible_but_unverified():
    response = {"constraints": [], "unresolved": [{"source_event_id": USER_ID,
        "evidence_quote": "If the phase witness is absent, wait",
        "reason": "No observed witness; this model text is not a success certificate."}]}
    _, interpreted = contexts(opaque_case(), response)
    assert interpreted["sections"]["UNMODELED_CLAUSES"] == response["unresolved"]
    assert interpreted["sections"]["ADVISORY_CONSTRAINTS"] == []
    assert interpreted["task_success"] is None


def test_simultaneous_incompatible_facts_are_unknown_in_both_controls_with_full_revisions():
    case = opaque_case()
    case["events"].insert(3, {"event_id": "same-step:other", "kind": "observation", "step": 2,
        "content": {"bound": 99}, "facts": [{"ref": FACT, "pointer": "/bound", "unit": UNIT}]})
    flat, interpreted = contexts(case)
    assert flat["sections"]["CURRENT_FACTS"] == interpreted["sections"]["CURRENT_FACTS"] == []
    assert flat["sections"]["UNKNOWN_FACTS"] == interpreted["sections"]["UNKNOWN_FACTS"]
    assert flat["sections"]["UNKNOWN_FACTS"] == [{"reference": FACT, "status": "UNKNOWN_OR_CONFLICTING"}]
    assert [x["value"] for x in interpreted["sections"]["FACT_REVISIONS"][FACT]] == [20, 14, 99]


def test_option_values_and_types_come_from_the_exact_escaped_source_pointer():
    case = opaque_case()
    original = copy.deepcopy(case)
    committed = commit_option(case, {"option_id": OPTION_B})
    selected = committed["selected_facts"][OUTPUT]
    assert selected["value"] == 18 and type(selected["value"]) is int
    assert selected["source"]["pointer"] == ROW_B + "/payload/s~1~0q/value"
    assert selected["source"]["event_id"] == "catalogue:α/~"
    assert selected["unit"] == UNIT
    assert committed["task_success"] is None
    assert committed["status"] == "COMMITTED_UNVERIFIED"
    assert case == original


@pytest.mark.parametrize("pointer", [ROW_B + "/payload/s~1~0q/value", "/settings/bound/value", ROW_A + "garbage/value"])
def test_option_cannot_alias_a_different_row_or_an_unoffered_antecedent(pointer):
    case = opaque_case()
    case["options"][0]["properties"][0]["pointer"] = pointer
    with pytest.raises(ValueError):
        build_observed(case)


def test_output_ref_cannot_shadow_observed_fact_even_when_opaque():
    case = opaque_case()
    case["options"][0]["properties"][0]["ref"] = FACT
    with pytest.raises(ValueError, match="shadow"):
        build_observed(case)


def test_planner_receives_untouched_rows_and_writer_receives_no_option_answer_key():
    case = opaque_case()
    plain = prepare_context(case, "FACT_LEDGER", len, 100000)
    planner = json.loads(planner_messages(case, plain)[1]["content"])
    observed = case["events"][3]["content"]["inventory"]
    assert planner["offers"] == [observed["edge/~A"], observed["edge/~B"]]
    writer = json.loads(writer_messages(case)[1]["content"])
    assert writer["output_fields"] == [OUTPUT]
    assert writer["fact_histories"][FACT][-1]["value"] == 14
    assert not {"offers", "expected_options", "annotations", "semantic_predicates"}.intersection(writer)


@pytest.mark.parametrize("condition", ["FACT_LEDGER", "ADVISORY_IR"])
def test_budget_boundary_is_exact_and_never_evicts_raw_rules_or_revision_history(condition):
    case = opaque_case()
    full = prepare_context(case, condition, len, 100000, advisory())
    exact = prepare_context(case, condition, len, full["required_tokens"], advisory())
    overflow = prepare_context(case, condition, len, full["required_tokens"] - 1, advisory())
    assert exact["status"] == "OK" and exact["context"] == full["context"]
    assert overflow["status"] == "BUDGET_OVERFLOW"
    assert overflow["sections"] == full["sections"]
    assert overflow["context"] == "BUDGET_OVERFLOW"


def test_opaque_observation_id_cannot_collide_with_the_runtime_commit_event():
    case = opaque_case()
    case["events"][0]["event_id"] = "committed-option"
    memory, _, _ = build_observed(case)
    original = memory.events["committed-option"]
    result = commit_option(case, {"option_id": OPTION_A})
    assert result["status"] == "COMMITTED_UNVERIFIED"
    assert result["snapshot"]["events"]["committed-option"] == original


def test_opaque_input_fact_cannot_be_revised_by_the_internal_selected_option_alias():
    case = opaque_case()
    case["events"][0]["facts"].append({"ref": "actual-selected-option", "pointer": "/settings/bound/value", "unit": UNIT})
    memory, _, _ = build_observed(case)
    original = memory.facts["actual-selected-option"]
    result = commit_option(case, {"option_id": OPTION_A})
    assert result["snapshot"]["facts"]["actual-selected-option"] == original
    selected = result["selected_facts"][OUTPUT]
    assert selected["entity_ref"]["ref"] != "actual-selected-option"


@pytest.mark.parametrize("unit", ["", 7, True, [], {}])
def test_malformed_option_unit_is_rejected_before_the_planner_can_choose(unit):
    case = opaque_case()
    case["options"][0]["properties"][0]["unit"] = unit
    with pytest.raises(ValueError):
        build_observed(case)


@pytest.mark.parametrize("target,operator", [
    ({"ref": FACT, "binding_mode": "SNAPSHOT"}, "LTE"),
    ({"ref": FACT, "tolerance": -1}, "ABS_DIFF_LTE"),
    ({"ref": FACT, "tolerance": True}, "ABS_DIFF_LTE"),
    ({"ref": FACT, "tolerance": "uninterpretable"}, "ABS_DIFF_LTE"),
    ({"value": 3, "binding_mode": "SNAPSHOT", "version": False}, "LTE"),
])
def test_malformed_binding_and_tolerance_metadata_stays_out_of_the_advisory_ir(target, operator):
    _, interpreted = contexts(opaque_case(), advisory(target=target, operator=operator))
    assert interpreted["interpretation"]["rejected"]
    assert interpreted["sections"]["ADVISORY_CONSTRAINTS"] == []
    assert interpreted["source_snapshot"]["constraints"] == {}


def test_overflow_descriptor_cannot_be_used_as_a_requirement_free_planner_context():
    case = opaque_case()
    overflow = prepare_context(case, "FACT_LEDGER", len, 1)
    assert overflow["status"] == "BUDGET_OVERFLOW"
    with pytest.raises(ValueError):
        planner_messages(case, overflow)


def test_writer_generated_event_collision_uses_fresh_identity_without_relabeling_observed_evidence():
    case = opaque_case()
    response = advisory()
    identifier = "writer-" + hashlib.sha256(encoded(response).encode()).hexdigest()
    case["events"][0]["event_id"] = identifier
    _, interpreted = contexts(case, response)
    assert interpreted["interpretation"]["rejected"] == []
    assert interpreted["interpretation"]["model_event_id"] != identifier
    assert interpreted["source_snapshot"]["events"][identifier]["kind"] == "observation"


def test_current_query_uses_fresh_event_identity_when_the_input_contains_its_hash_name():
    case = opaque_case()
    identifier = "current-query-" + hashlib.sha256((case["case_id"] + case["query"]).encode()).hexdigest()
    case["events"][0]["event_id"] = identifier
    memory, _, users = build_observed(case)
    assert memory.events[identifier]["kind"] == "observation"
    assert users[-1]["event_id"] != identifier
    assert users[-1]["content"] == case["query"]


@pytest.mark.parametrize("collision_kind", ["fact", "output"])
def test_advisory_identity_does_not_occupy_an_opaque_input_or_output_reference(collision_kind):
    case = opaque_case()
    identifier = "advisory-" + hashlib.sha256((case["case_id"] + "0").encode()).hexdigest()
    response = advisory()
    if collision_kind == "fact":
        case["events"][0]["facts"].append({"ref": identifier, "pointer": "/settings/bound/value", "unit": UNIT})
    else:
        for option in case["options"]:
            option["properties"][0]["ref"] = identifier
        response["constraints"][0]["subject_ref"] = identifier
    before, _, _ = build_observed(case)
    _, interpreted = contexts(case, response)
    assert interpreted["interpretation"]["rejected"] == []
    generated = interpreted["interpretation"]["references"][0]
    assert generated != identifier
    assert interpreted["source_snapshot"]["facts"] == before.snapshot()["facts"]
    if collision_kind == "output":
        assert any(row["reference"] == identifier and row["reason"] == "UNKNOWN"
                   for row in interpreted["sections"]["INTERPRETATION_UNKNOWN"])


def test_invented_snapshot_version_is_exposed_as_unknown_without_a_success_claim():
    _, interpreted = contexts(opaque_case(), advisory(target={"ref": FACT, "binding_mode": "SNAPSHOT", "version": 999}))
    assert interpreted["interpretation"]["rejected"] == []
    unknown = interpreted["sections"]["INTERPRETATION_UNKNOWN"]
    assert {"reference": FACT, "version": 999, "reason": "UNKNOWN"} in unknown
    assert interpreted["task_success"] is None
    assert interpreted["hard_semantic_checks"] is False


def test_option_unit_is_explicit_schema_metadata_and_not_an_independent_source_certificate():
    case = opaque_case()
    case["events"][3]["content"]["inventory"]["edge/~A"]["payload"]["s/~q"]["unit"] = "ms"
    result = commit_option(case, {"option_id": OPTION_A})
    assert result["selected_facts"][OUTPUT]["value"] == 11
    assert result["selected_facts"][OUTPUT]["unit"] == UNIT
    assert result["unit_authority"].startswith("CALLER_DECLARED_SCHEMA")
    assert result["task_success"] is None


def test_unused_property_metadata_cannot_carry_offline_labels_into_planner_bindings():
    case = opaque_case()
    injected = "OFFLINE_ORACLE_SHOULD_NOT_BE_RUNTIME_BINDING_METADATA"
    case["options"][0]["properties"][0].update({
        "semantic_predicates": [{"answer": injected}], "expected_option_id": OPTION_A,
    })
    # A strict input rejection or a sanitized binding view are both sound.
    # Arbitrary observed row CONTENT remains task data, outside this schema check.
    try:
        context = prepare_context(case, "FACT_LEDGER", len, 100000)
    except ValueError:
        return
    data = json.loads(planner_messages(case, context)[1]["content"])
    assert injected not in encoded(data["option_property_bindings"])
    for option in data["option_property_bindings"]:
        for field in option["properties"]:
            assert set(field) <= {"ref", "pointer", "unit"}
