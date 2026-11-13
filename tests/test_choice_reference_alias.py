"""Regressions for choice aliases, causal versions, and invalidation."""
from __future__ import annotations

import copy

import pytest

from constraint_memory.memory import ledger as original


RID = "decision:Benjamin:day1:accommodation"
ALIAS = RID + ":value"


def memory(module, variant="E3"):
    result = module.LedgerMemory(variant, {"token_budget": 10000})
    result.reset("g")
    return result


def request(name="Benjamin", step=0, **extra):
    return {"group_id": "g", "traveler": name, "query": "", "step": step, **extra}


def choose(memory, value="Hotel A, Boston", step=0, **extra):
    return memory.commit_subtask({**request(step=step), "action": value,
        "decisions": [{"traveler": "Benjamin", "day": 1, "slot": "accommodation",
                       "value": value, "source_refs": ["event:action"]}], **extra})


def join(memory, ref=ALIAS, step=1):
    memory.observe({"event_id": f"join-{step}", "group_id": "g", "step": step,
                    "type": "user_request", "content": "Andrew stays with Benjamin."})
    row = {"constraint_id": "join", "kind": "JOIN",
           "scope": {"group": "g", "traveler": "Andrew"},
           "subject": {"traveler": "Andrew", "day": 1, "slot": "accommodation", "attribute": "value"},
           "operator": "EQUALS", "target": {"ref": ref}, "source_refs": [f"event:join-{step}"]}
    return memory.begin_subtask(request("Andrew", step, query="stay with Benjamin",
                                        extraction={"constraints": [row]}))


def codes(memory):
    return {row["code"] for row in memory._diagnostics}


@pytest.mark.parametrize("suffix,version", [("", None), ("@v1", 1), ("@v12", 12)])
def test_alias_preserves_exact_version(proposed, suffix, version):
    assert proposed._split_ref(ALIAS + suffix) == (RID, version)


@pytest.mark.parametrize("ref", [
    RID, RID + "@v2", RID + ":price", RID + ":rating@v2", RID + ":cuisine",
    RID + ":room_type@v3", RID + ":price:value", "constraint:Benjamin:value@v1",
    "definition:value@v1", "event:choice:value", "decision:Benjamin:accommodation:value",
    "decision:Benjamin:day:accommodation:value", ALIAS + "@vbad",
])
def test_properties_nondecisions_and_malformed_aliases_are_unchanged(proposed, ref):
    assert proposed._split_ref(ref) == original._split_ref(ref)


@pytest.mark.parametrize("variant", ["E2", "E3", "E4"])
@pytest.mark.parametrize("suffix", ["", "@v1"])
def test_visible_committed_choice_alias_resolves_and_closes(proposed, variant, suffix):
    m = memory(proposed, variant)
    choose(m)
    response = join(m, ALIAS + suffix)
    assert response["missing_refs"] == []
    assert RID + "@v1" in response["closure_ids"]
    constraint = next(r for r in response["records"] if r.get("constraint_id") == "join")
    assert constraint["target"]["ref"] == ALIAS + suffix  # evidence is not rewritten
    assert constraint["resolved_target_ref"] == RID + "@v1"
    assert not {"MISSING_REFERENCE", "PROPERTY_EVIDENCE_REQUIRED", "UNRESOLVED_REFERENCE"} & codes(m)


@pytest.mark.parametrize("suffix", ["", "@v1"])
def test_alias_cannot_expose_future_record(proposed, suffix):
    m = memory(proposed, "E4")
    choose(m, step=5)
    assert m._resolve(ALIAS + suffix, 4) is None
    assert "FUTURE_REFERENCE" in codes(m)
    records, missing = m._closure([ALIAS + suffix], 4)
    assert records == []
    assert missing == [ALIAS + suffix]


def test_alias_cannot_expose_future_version_of_visible_choice(proposed):
    m = memory(proposed, "E4")
    choose(m, step=0)
    choose(m, "Hotel B, Boston", step=5)
    assert m._resolve(ALIAS + "@v2", 4) is None
    assert "MISSING_VERSION" in codes(m)
    assert m._resolve(ALIAS, 4)["value"] == "Hotel A, Boston"


def test_explicit_stale_seed_still_rejected_and_latest_alias_is_valid(proposed):
    m = memory(proposed, "E4")
    choose(m)
    choose(m, "Hotel B, Boston", step=2)
    records, missing = m._closure([ALIAS + "@v1"], 3)
    assert records == [] and missing == [ALIAS + "@v1"]
    assert "STALE_DEPENDENCY" in codes(m)
    assert m._resolve(ALIAS, 3)["value"] == "Hotel B, Boston"


@pytest.mark.parametrize("variant", ["E2", "E3", "E4"])
def test_live_dependency_rebinds_to_latest_without_resurrecting_old_value(proposed, variant):
    m = memory(proposed, variant)
    choose(m)
    choose(m, "Hotel B, Boston", step=2)
    response = join(m, ALIAS + "@v1", step=3)
    record = next(r for r in response["records"] if r.get("decision_id") == RID)
    assert record["value"] == "Hotel B, Boston" and record["version"] == 2
    assert RID + "@v1" not in response["closure_ids"]
    assert "DEPENDENCY_REBOUND" in codes(m)


@pytest.mark.parametrize("suffix", ["", "@v1", "@v2"])
def test_invalidated_alias_never_resurrects_a_choice(proposed, suffix):
    m = memory(proposed, "E5")
    choose(m)
    result = m.commit_subtask({**request(step=1), "action": "observation invalidates choice",
         "invalidate_decision_refs": [ALIAS], "remaining_action_budget": 2})
    assert result["recovery"]
    assert RID + "@v1" in m.invalidations
    assert m._resolve(ALIAS + suffix, 2) is None
    assert m._resolve_binding(ALIAS + suffix, 2) is None
    records, _ = m._closure([ALIAS + suffix], 2)
    assert not any(r.get("value") == "Hotel A, Boston" for r in records)
    assert m._resolve(ALIAS + "@v1", 0)["value"] == "Hotel A, Boston"  # causal past retained
    assert len(m.decisions[RID]) == 2  # history survives the tombstone


def test_failed_proposed_branch_preserves_only_previous_valid_choice(proposed):
    m = memory(proposed, "E4")
    choose(m)
    choose(m, "Rejected Hotel, Boston", step=1, branch_status="FAILED", status="FAILED")
    assert m._resolve(ALIAS, 2)["value"] == "Hotel A, Boston"
    assert m._resolve(ALIAS + "@v2", 2) is None
    assert "STALE_DEPENDENCY" in codes(m) or "INVALID_REFERENCE" in codes(m)


def test_property_id_and_entity_anchor_are_preserved_and_retired_with_entity(proposed):
    m = memory(proposed, "E4")
    m.observe({"event_id": "tool", "group_id": "g", "step": 0, "type": "tool_result",
               "content": {"Name": "Hotel A", "City": "Boston", "Price": 80}})
    choose(m)
    m.commit_subtask({**request(step=1), "action": "Hotel A, Boston",
         "decisions": [{"traveler": "Benjamin", "day": 1, "slot": "accommodation",
                        "attribute": "price", "value": 80,
                        "source_refs": ["event:tool", "event:action"]}]})
    price = m._resolve(RID + ":price@v1", 1)
    assert price["value"] == 80 and price["decision_id"] == RID + ":price"
    assert price["entity_ref"] == RID + "@v1"
    choose(m, "Hotel B, Boston", step=2)
    assert m._resolve(RID + ":price", 3) is None
    assert "STALE_PROPERTY_EVIDENCE" in codes(m)
    assert m._resolve(ALIAS, 3)["value"] == "Hotel B, Boston"


def test_alias_deduplicates_with_canonical_reference_and_does_not_change_archive(proposed):
    m = memory(proposed)
    choose(m)
    before = copy.deepcopy(m.decisions)
    records, missing = m._closure([RID, ALIAS, ALIAS + "@v1"], 1)
    assert missing == []
    assert [r["record_ref"] for r in records] == [RID + "@v1"]
    assert m.decisions == before


def test_missing_nonexistent_alias_remains_missing_without_fallback(proposed):
    m = memory(proposed)
    records, missing = m._closure([ALIAS], 0)
    assert records == [] and missing == [ALIAS]
    assert "MISSING_REFERENCE" in codes(m)
    assert "PROPERTY_EVIDENCE_REQUIRED" not in codes(m)


@pytest.fixture(scope="module")
def proposed():
    return original
