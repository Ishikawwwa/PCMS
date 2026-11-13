import pytest

from .memory import ReadScope, Snapshot, read_context
from .run import MERMAID, SHOPPING, TRAVEL


def read(snapshot, initial, scope, mode="subspaces+table"):
    return read_context(snapshot, initial, scope, mode=mode, token_limit=1000, count_tokens=len)


def test_shared_purchase_appears_once_across_initial_and_table_reads():
    snapshot = Snapshot(SHOPPING, through=31)
    initial = snapshot.recall(("bundle24", "slot3", "B071S91WRZ"), limit=1)
    result = read(snapshot, initial, ReadScope("bundle24", ("slot3",)))
    assert result["delivered_ids"] == ["p3"]
    assert result["context"].count("B071S91WRZ") == 1


def test_considered_item_does_not_authorize_expansion():
    snapshot = Snapshot(SHOPPING + (MERMAID,), through=31)
    result = read(snapshot, (MERMAID,), ReadScope("bundle24", ("slot3",)))
    assert result["table_anchor"] is False
    assert result["delivered_ids"] == ["mermaid"]


@pytest.mark.parametrize(("step", "expected"), [(3, "budget80"), (4, "budget100")])
def test_travel_expansion_is_scoped_and_uses_the_latest_confirmation(step, expected):
    snapshot = Snapshot(TRAVEL, through=step)
    result = read(snapshot, TRAVEL[:2], ReadScope("trip1", ("destination", "access", "budget")))
    assert result["delivered_ids"] == ["destination", "access", expected]


@pytest.mark.parametrize(("step", "invalid"), [(3, TRAVEL[4]), (4, TRAVEL[2])])
def test_forced_initial_recall_rejects_future_and_superseded_confirmations(step, invalid):
    with pytest.raises(ValueError, match="outside the active snapshot"):
        read(Snapshot(TRAVEL, through=step), (invalid,), ReadScope("trip1", ("budget",)))


def test_budget_requires_every_prior_receipt():
    scope = ReadScope("bundle24", ("slot3",), 10000, ("slot1", "slot2", "slot3"))
    full = read(Snapshot(SHOPPING + (MERMAID,), through=31), (SHOPPING[1], MERMAID), scope)
    assert "Spent $65.26. Remaining $34.74." in full["context"]
    partial = read(Snapshot(SHOPPING[1:], through=31), (SHOPPING[1],), scope)
    assert partial["delivered_ids"] == ["p2", "p3"]
    assert "Remaining" not in partial["context"]


def test_priority_packing_preserves_decision_evidence_at_the_direct_read_budget():
    snapshot = Snapshot(SHOPPING + (MERMAID,), through=31)
    scope = ReadScope("bundle24", ("slot3",), 10000, ("slot1", "slot2", "slot3"))
    initial = (SHOPPING[1], MERMAID)
    direct = read(snapshot, (), scope, mode="flat-table")
    results = [
        read_context(snapshot, initial, scope, mode=mode, token_limit=direct["tokens"], count_tokens=len)
        for mode in ("subspaces+table", "subspaces+table-priority")
    ]
    ordinary, priority = results
    assert ordinary["initial_ids"] == priority["initial_ids"] == ["p2", "mermaid"]
    assert ordinary["context"] != direct["context"]
    assert priority["context"] == direct["context"]


def test_packing_diagnostics_report_rejection_without_changing_context():
    snapshot = Snapshot(SHOPPING, through=31)
    initial = snapshot.recall(("bundle24", "slot3"), limit=2)
    decisions = []
    result = read_context(snapshot, initial, ReadScope("bundle24", ("slot3",)), mode="subspaces",
                          token_limit=len(initial[0].line()), count_tokens=len, decisions=decisions)
    assert result["context"] == initial[0].line()
    assert [(row["candidate"], row["status"]) for row in decisions] == [
        (initial[0].id, "included"), (initial[1].id, "over_budget"),
    ]
