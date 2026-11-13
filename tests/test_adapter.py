from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from constraint_memory.benchmark_adapter.data import ChronologicalAdapter, assert_visible_request, load_base, load_splits, load_turn
from constraint_memory.benchmark_adapter.planner import PlannerWorker
from constraint_memory.benchmark_adapter.scoring import score_official

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/MemoryArena"
DB = UPSTREAM / "env/env_systems/travel_planner_env/database"


def test_request_rejects_gold_and_future_fields():
    request = {"group_id": "1", "traveler_index": 1, "name": "A", "query": "I am A."}
    for field in ["questions", "answers", "daily_plans", "oracle_annotations", "future_queries"]:
        with pytest.raises(ValueError, match="Unpermitted"):
            assert_visible_request({**request, field: []})


@pytest.mark.skipif(not (UPSTREAM / "agent/travel_planner.py").exists(), reason="official source not prepared")
def test_worker_hidden_reads_and_scorer_imports_are_denied():
    code = """
import importlib
from pathlib import Path
from constraint_memory.benchmark_adapter.planner_worker import deny_hidden_reads
from constraint_memory.benchmark_adapter.upstream import mount_upstream, TRAVEL_PACKAGE
mount_upstream(Path('external/MemoryArena'))
deny_hidden_reads([str(Path('data/hidden').resolve()), str(Path('data/visible').resolve())])
for path in ['data/hidden/memoryarena_travel.jsonl', 'data/visible/6/turn_02.json']:
    try:
        open(path, encoding='utf-8').read()
        raise AssertionError('hidden/future path was readable')
    except PermissionError:
        print('blocked')
for name in [TRAVEL_PACKAGE + '.eval', TRAVEL_PACKAGE + '.data_loader']:
    try:
        importlib.import_module(name)
        raise AssertionError('hidden module was importable')
    except ImportError:
        print('blocked')
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert process.stdout.splitlines() == ["blocked"] * 4


@pytest.mark.skipif(not (ROOT / "data/manifests/splits.json").exists(), reason="run official data preparation first")
def test_frozen_splits_have_disjoint_base_families():
    splits = load_splits(ROOT)
    assert splits["family_counts"] == {"dev": 8, "validation": 7, "holdout": 30}
    assert len(splits["partitions"]["all"]) == 270
    assert len(splits["partitions"]["smoke"]) == 10
    assert set(splits["partitions"]["smoke"]) <= set(splits["partitions"]["dev"])
    family_sets = [set(splits["families"][key]) for key in ["dev", "validation", "holdout"]]
    assert not family_sets[0] & family_sets[1]
    assert not family_sets[0] & family_sets[2]
    assert not family_sets[1] & family_sets[2]
    for gid in splits["partitions"]["smoke"]:
        request = load_turn(ROOT, str(gid), 1)
        assert set(request) == {"group_id", "base_template_id", "traveler_index", "name", "query"}


@pytest.mark.skipif(not (ROOT / "data/manifests/splits.json").exists(), reason="run official data preparation first")
def test_chronological_adapter_blocks_future_and_cross_group_commits():
    gid = str(load_splits(ROOT)["partitions"]["smoke"][0])
    adapter = ChronologicalAdapter(ROOT, gid)
    first = adapter.next_request()
    assert first["traveler_index"] == 1
    with pytest.raises(RuntimeError, match="Commit the current traveler"):
        adapter.next_request()
    with pytest.raises(ValueError, match="does not match"):
        adapter.commit_result({"group_id": "wrong", "traveler_index": 1})
    with pytest.raises(ValueError, match="does not match"):
        adapter.commit_result({"group_id": gid, "traveler_index": 2})
    adapter.commit_result({"group_id": gid, "traveler_index": 1, "plan": None})
    second = adapter.next_request()
    assert second["traveler_index"] == 2
    assert len(adapter.past_results) == 1


@pytest.mark.skipif(not (DB / "flights/clean_Flights_2022.csv").exists(), reason="official tool DB not prepared")
def test_official_planner_worker_tools_parser_and_visibility():
    # A scripted client validates the real loop and real CSV tools. It is not an
    # inference experiment or an oracle performance claim.
    seen_messages = []
    request = {"group_id": "999", "traveler_index": 2, "name": "Auditor", "query": "I am Auditor. CURRENT_PUBLIC_REQUEST."}
    base = {"name": "Base", "query": "BASE_PUBLIC_REQUEST", "daily_plans": [{"days": 1, "current_city": "Rockford"}]}

    class ScriptedClient:
        def chat_with_tools(self, messages, tools):
            seen_messages.append(messages)
            assert any(tool["function"]["name"] == "RestaurantSearch" for tool in tools)
            if len(seen_messages) == 1:
                call = SimpleNamespace(id="real-search", name="RestaurantSearch", arguments={"city": "Rockford"})
                return SimpleNamespace(content=None, tool_calls=[call], raw_response={})
            assert any(message["role"] == "tool" and "Average Cost" in message["content"] for message in messages)
            return SimpleNamespace(content="=== Auditor's Plan ===\nDay 1:\nCurrent City: Rockford\nBreakfast: scripted fixture\n", tool_calls=[], raw_response={})

    planner = PlannerWorker(UPSTREAM, DB, ScriptedClient(), max_steps=3, timeout_seconds=60)
    result = planner.run_turn(request, "PERMITTED_PAST_MEMORY", base)
    assert result["success"] and result["total_steps"] == 2
    assert result["plan"][0]["breakfast"] == "scripted fixture"
    observation = result["structured_tool_results"][0]
    assert observation["tool_call_id"] == "real-search"
    assert observation["tool_name"] == "RestaurantSearch"
    assert observation["latency_seconds"] >= 0
    assert observation["displayed_columns"] == ["Name", "Average Cost", "Cuisines", "Aggregate Rating", "City"]
    assert observation["displayed_rows"]
    assert all(row["City"] == "Rockford" for row in observation["displayed_rows"])
    assert observation["formatted_result"] == result["scratchpad"][0]["tool_results"][0]["result"]
    assert not result["worker_audit"]["hidden_modules_loaded"]
    assert not result["worker_audit"]["model_credentials_present"]
    joined = json.dumps(seen_messages[0])
    assert "CURRENT_PUBLIC_REQUEST" in joined
    assert "BASE_PUBLIC_REQUEST" in joined
    assert "PERMITTED_PAST_MEMORY" in joined
    assert "FUTURE_SECRET" not in joined


def fake_hidden_dataset(path: Path):
    base_day = {"days": 1, "breakfast": "base", "lunch": "-", "dinner": "-", "attraction": "-", "transportation": "-", "accommodation": "-"}
    answer_one = {**base_day, "breakfast": "preferred restaurant"}
    answer_two = {**base_day, "breakfast": "another preference"}
    rows = [{"id": 999, "base_person": {"name": "Base", "query": "public", "daily_plans": [base_day]},
             "questions": ["I am A.", "I am B."], "answers": [[answer_one], [answer_two]]}]
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    return answer_one, answer_two


@pytest.mark.skipif(not (UPSTREAM / "agent/travel_planner.py").exists(), reason="official source not prepared")
def test_official_scorer_complete_grid_missing_members_and_exact_oracle(tmp_path):
    dataset = tmp_path / "hidden.jsonl"
    answer_one, answer_two = fake_hidden_dataset(dataset)
    predictions = [{"group_id": "999", "traveler_index": 1, "plan": [answer_one]},
                   {"group_id": "999", "traveler_index": 2, "plan": [answer_two]}]
    oracle = score_official(UPSTREAM, dataset, predictions, ["999"])
    assert oracle["official_metrics"] == {"SR": 1, "PS": 1, "SPS": 1}
    assert oracle["metric_metadata"]["planner_modules_loaded"] == []
    partial = score_official(UPSTREAM, dataset, predictions[:1], ["999"])
    assert partial["official_metrics"] == {"SR": 0, "PS": 0.5, "SPS": 0.5}
    assert partial["per_group"][0]["traveler_count"] == 2
    assert partial["per_group"][0]["traveler_metrics"][1]["missing_or_unparsed"]
    absent = score_official(UPSTREAM, dataset, [], ["999"])
    assert absent["official_metrics"] == {"SR": 0, "PS": 0, "SPS": 0}
    with pytest.raises(ValueError, match="Duplicate prediction"):
        score_official(UPSTREAM, dataset, predictions + predictions[:1], ["999"])
    with pytest.raises(ValueError, match="Hidden dataset checksum"):
        score_official(UPSTREAM, dataset, predictions, ["999"], expected_dataset_sha="incorrect")
