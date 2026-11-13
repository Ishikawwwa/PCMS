"""Deterministic reference-kernel tests; no inference or comparator claims."""
import json

import pytest

from constraint_memory.reference import MageReferenceKernel


def validator(instruction, trace, summary):
    return ("bad" not in summary, {"instruction": instruction, "reason": "failed segment"})


def prefix_and_failure():
    kernel = MageReferenceKernel(validator=validator, public_anchor={"fixed": "base itinerary"})
    first = kernel.grow("search prefix", {"result": "valid anchor"})
    second = kernel.grow("select prefix", {"result": "accepted prefix"})
    good = kernel.compress("valid prefix")
    assert kernel.maintain("visible requirements")["status"] == "PASS"
    bad_first = kernel.grow("search suffix", {"result": "wrong candidate"})
    bad_last = kernel.grow("select suffix", {"result": "bad choice"})
    bad = kernel.compress("bad suffix")
    failure = kernel.maintain("visible requirements")
    assert failure["status"] == "FAIL"
    return kernel, first, second, good, bad_first, bad_last, bad, failure


def test_revision_preserves_valid_prefix_restores_exact_raw_pointer_and_keeps_history():
    kernel, first, second, good, bad_first, bad_last, bad, failure = prefix_and_failure()
    before = kernel.snapshot()
    result = kernel.revise(failure["feedback"], failure["summary_id"])
    assert result["pt"] == good["id"] and result["pb"] == second["id"]
    state = kernel.state()
    assert [summary["content"] for summary in state["C"]] == ["valid prefix"]
    assert state["R"] == []
    assert "bad suffix" not in json.dumps(state["C"])
    assert all(hint["active"] is False for hint in state["H"])
    assert any(hint.get("content") == "bad suffix" and hint.get("note") for hint in state["H"])
    after = kernel.snapshot()
    assert after["raw_nodes"] == before["raw_nodes"]
    assert after["summary_nodes"] == before["summary_nodes"]
    assert len(after["history"]) == len(before["history"]) + 1


def test_repaired_suffix_becomes_sibling_branch_without_deleting_failed_branch():
    kernel, first, second, good, bad_first, bad_last, bad, failure = prefix_and_failure()
    kernel.revise(failure["feedback"], bad["id"])
    repaired_raw = kernel.grow("search corrected suffix", {"result": "valid replacement"})
    assert repaired_raw["parent"] == second["id"]
    repaired = kernel.compress("corrected suffix")
    kernel.maintain("visible requirements")
    snapshot = kernel.snapshot()
    assert snapshot["raw_nodes"][second["id"]]["children"] == [bad_first["id"], repaired_raw["id"]]
    assert snapshot["summary_nodes"][good["id"]]["children"] == [bad["id"], repaired["id"]]
    assert bad["id"] in snapshot["summary_nodes"] and bad_last["id"] in snapshot["raw_nodes"]
    assert [item["content"] for item in snapshot["state"]["C"]] == ["valid prefix", "corrected suffix"]


def test_identical_raw_siblings_and_same_cover_summary_reuse_stable_ids():
    kernel, first, second, good, bad_first, bad_last, bad, failure = prefix_and_failure()
    kernel.revise(failure["feedback"], bad["id"])
    raw_count = len(kernel.snapshot()["raw_nodes"])
    summary_count = len(kernel.snapshot()["summary_nodes"])
    replay_first = kernel.grow("search suffix", {"result": "wrong candidate"})
    replay_last = kernel.grow("select suffix", {"result": "bad choice"})
    rewritten = kernel.compress("reinterpreted suffix")
    assert replay_first["id"] == bad_first["id"] and replay_last["id"] == bad_last["id"]
    assert rewritten["id"] == bad["id"] and rewritten["version"] == 2
    assert rewritten["cover_nodes"] == bad["cover_nodes"]
    assert rewritten["boundary_raw_id"] == second["id"] and rewritten["boundary_step"] == second["step"]
    assert rewritten["note"] == failure["feedback"]  # retained until revalidation
    kernel.maintain("visible requirements")
    snapshot = kernel.snapshot()
    assert len(snapshot["raw_nodes"]) == raw_count and len(snapshot["summary_nodes"]) == summary_count
    assert snapshot["summary_nodes"][bad["id"]]["note"] is None
    overwrite = [item for item in snapshot["history"] if item["operation"] == "Compress"][-1]
    assert overwrite["reused"] and overwrite["previous"]["content"] == "bad suffix"
    assert any(hint.get("content") == "bad suffix" for hint in snapshot["state"]["H"])


def test_revision_to_earlier_declared_boundary_excludes_entire_later_suffix():
    kernel = MageReferenceKernel(validator=lambda *args: (True, None))
    raw_ids, summary_ids = [], []
    for label in ["prefix", "middle", "late"]:
        raw_ids.append(kernel.grow(label, {"observed": label})["id"])
        summary_ids.append(kernel.compress(label)["id"])
        kernel.maintain("visible task")
    before = kernel.snapshot()
    result = kernel.revise("middle is incorrect", summary_ids[1])
    assert result["excluded_summary_ids"] == summary_ids[1:]
    assert kernel.pb == raw_ids[0] and kernel.pt == summary_ids[0]
    assert [item["content"] for item in kernel.state()["C"]] == ["prefix"]
    assert kernel.snapshot()["raw_nodes"] == before["raw_nodes"]
    assert kernel.snapshot()["summary_nodes"] == before["summary_nodes"]


def test_callbacks_receive_only_the_covered_current_segment_and_snapshots_are_deterministic():
    seen = []
    def summarize(instruction, trace):
        seen.append((instruction, trace))
        return {"summary": [item["action"] for item in trace]}
    def execute():
        kernel = MageReferenceKernel(summarizer=summarize, validator=lambda *args: (True, "checked"))
        kernel.grow({"tool": "search", "city": "visible"}, {"rows": ["observed"]})
        kernel.compress(task_instruction="current only")
        kernel.maintain("current only")
        return kernel.snapshot()
    first, second = execute(), execute()
    assert first == second
    assert len(seen[0][1]) == 1 and seen[0][1][0]["id"] == "r1"
    assert seen[0][0] == "current only"
    json.dumps(first, allow_nan=False)
    first["raw_nodes"]["r1"]["content"]["action"]["city"] = "mutated snapshot"
    assert second["raw_nodes"]["r1"]["content"]["action"]["city"] == "visible"


def test_invalid_boundary_empty_compression_and_nonstrict_validator_do_not_create_hidden_state():
    kernel = MageReferenceKernel()
    with pytest.raises(ValueError, match="nonempty"):
        kernel.compress("empty")
    with pytest.raises(ValueError, match="nonroot"):
        kernel.revise("feedback", "s0")
    kernel.grow("one", "observed")
    kernel.compress("one")
    with pytest.raises(ValueError, match="injected validator"):
        kernel.maintain("instruction")
    with pytest.raises(ValueError, match="bool"):
        kernel.maintain("instruction", validator=lambda *args: ("false", "not a boolean"))
    with pytest.raises(ValueError, match="nonroot"):
        kernel.revise("feedback", "missing")
    with pytest.raises(ValueError, match="finite JSON"):
        kernel.grow("one", float("nan"))
