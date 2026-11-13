"""Offline execution smoke on public development requests and the real scorer.

The injected planner copies the public base itinerary. It is a deliberately
limited scripted diagnostic, never model completion or research evidence.
Answer keys are accessible only to the production isolated scoring worker.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from constraint_memory.benchmark_adapter import load_splits, score_official
from constraint_memory.config import load_config
from constraint_memory.orchestration import runner
from constraint_memory.orchestration.logging import read_jsonl
from constraint_memory.orchestration.replay import replay_run

ROOT = Path(__file__).resolve().parents[1]
PREPARED = all((ROOT / relative).exists() for relative in (
    "data/hidden/memoryarena_travel.jsonl", "data/manifests/provenance.json",
    "data/manifests/qwen_tokenizer.json",
    "external/qwen-tokenizer/tokenizer.json", "external/MemoryArena/env/env_systems/travel_planner_env/eval.py",
))
pytestmark = pytest.mark.skipif(not PREPARED, reason="run pinned official source/data preparation first")


@pytest.fixture(scope="module")
def prepared_workspace(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("official-runner-diagnostic")
    for relative in ("data/hidden/memoryarena_travel.jsonl", "data/manifests/provenance.json",
                     "data/manifests/qwen_tokenizer.json",
                     "data/manifests/splits.json", "data/manifests/splits.sha256",
                     "external/qwen-tokenizer/tokenizer.json"):
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    splits = load_splits(workspace)
    for group in splits["partitions"]["smoke"][:2]:
        shutil.copytree(ROOT / "data/visible" / str(group), workspace / "data/visible" / str(group))
    return workspace


class NoNetworkClient:
    def __init__(self):
        self.records = []
        self.group = None
        self.traveler = None

    def verify_model(self):
        return {"model": "qwen/qwen3.6-27b", "scripted_diagnostic": True, "network_calls": 0}

    def set_scope(self, group, traveler=None):
        self.group, self.traveler = group, traveler

    def complete(self, *args, **kwargs):
        raise AssertionError("The scripted diagnostic must not use model inference")


def diagnostic_config():
    config = load_config(ROOT / "configs/travel.yaml")
    config["experiment"]["upstream_root"] = str(ROOT / "external/MemoryArena")
    config["memory"]["backend"] = "E2"
    config["memory"]["parser"] = "rules"
    config["agent"]["max_agent_visible_memory_tokens"] = 262144
    config["evaluation"]["bootstrap_iterations"] = 100
    return config


def preflight_without_database_scan(workspace):
    # Prepared files are byte-for-byte copies of pinned data. Only expensive
    # real CSV database scans are replaced; this planner never calls tools.
    return {"provenance": json.loads((workspace / "data/manifests/provenance.json").read_text()),
            "splits": load_splits(workspace), "database": {"database_path": str(ROOT / "external/MemoryArena/env/env_systems/travel_planner_env/database")}}


def test_two_development_groups_memory_reset_replay_and_official_score_roundtrip(prepared_workspace, monkeypatch):
    workspace = prepared_workspace
    monkeypatch.setattr(runner, "verify_readiness", preflight_without_database_scan)
    seen = []

    class PublicBasePlanner:
        def __init__(self, upstream, database, client, **kwargs):
            self.client = client

        def run_turn(self, packet, context, base):
            assert set(packet) == {"group_id", "base_template_id", "traveler_index", "name", "query"}
            assert "answers" not in packet and "questions" not in packet
            seen.append((packet["group_id"], packet["traveler_index"]))
            plan = copy.deepcopy(base["daily_plans"])
            action = runner.format_base({**base, "name": packet["name"]})
            self.client.records.append({"group_id": self.client.group, "traveler": packet["name"],
                "operation": "agent", "prompt_tokens": 10, "completion_tokens": 5,
                "cost_rub": 0, "latency_seconds": 0, "scripted_diagnostic": True})
            return {"success": True, "final_plan": action,
                    "submitted_text":action.split("\n",1)[1], "plan": plan, "scratchpad": [],
                    "total_steps": 1, "error_message": None, "latency_seconds": 0}

    run_dir = runner.run_experiment(diagnostic_config(), root=workspace,
        variants=["E0", "E2"], limit=2, run_id="scripted-public-base-diagnostic",
        client=NoNetworkClient(), planner_factory=PublicBasePlanner)
    rows = read_jsonl(run_dir / "results.jsonl")
    assert len(rows) == 4
    assert all(row["official_metric_metadata"]["scorer_isolated"] for row in rows)
    assert all(row["storage_bytes"] > 0 and row["record_count"] > 0 for row in rows)
    assert all(row["travelers"][0]["traveler_index"] == 0 for row in rows)
    assert all("PS" in row["travelers"][-1]["official_metrics"] for row in rows)
    reset_ops = [op for op in read_jsonl(run_dir / "memory_ops.jsonl") if op["op"] == "reset"]
    assert len(reset_ops) == 4
    for row in rows:
        snapshot = json.loads((run_dir / f"snapshots/{row['variant']}/{row['group_id']}.json").read_text())
        assert all(event["group_id"] == row["group_id"] for event in snapshot["evidence"])
        assert snapshot["decisions"]  # actual submitted public choices are recorded
    assert all(result["match"] for result in replay_run(run_dir))
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    groups = sorted({row["group_id"] for row in rows})
    for variant in ("E0", "E2"):
        rescored = score_official(ROOT / "external/MemoryArena", workspace / "data/hidden/memoryarena_travel.jsonl",
            [prediction for prediction in predictions if prediction["variant"] == variant], groups)
        original = json.loads((run_dir / f"official_scores_{variant}.json").read_text())
        assert rescored["official_metrics"] == original["official_metrics"]
    report = json.loads((run_dir / "reports/summary.json").read_text())
    assert report["comparison"]["paired_groups"] == 2
    assert report["comparison"]["metrics"]["SR"]["paired_difference"]["mean"] == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["model_catalog_snapshot"]["scripted_diagnostic"]
    assert len(seen) == sum(row["traveler_count"] for row in rows)


def test_missing_named_submission_cannot_hallucinate_committed_decisions(prepared_workspace, monkeypatch):
    workspace = prepared_workspace
    monkeypatch.setattr(runner, "verify_readiness", preflight_without_database_scan)

    class WrongNamePlanner:
        def __init__(self, *args, **kwargs):
            pass

        def run_turn(self, packet, context, base):
            return {"success": True, "final_plan": "=== SomeoneElse's Plan ===\nDay 1:\nBreakfast: invented choice\n",
                    "plan": None, "scratchpad": [], "total_steps": 1, "error_message": None}

    run_dir = runner.run_experiment(diagnostic_config(), root=workspace, variants=["E2"],
        limit=1, run_id="scripted-missing-name-diagnostic", client=NoNetworkClient(), planner_factory=WrongNamePlanner)
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    assert all(prediction["plan"] is None for prediction in predictions)
    assert all(prediction["error_message"] for prediction in predictions)
    group = predictions[0]["group_id"]
    snapshot = json.loads((run_dir / f"snapshots/E2/{group}.json").read_text())
    assert all(decision["traveler"] != "SomeoneElse" for decision in snapshot["decisions"])
    assert all(outcome["status"] == "FAILED" for outcome in snapshot["outcomes"][1:])
    assert read_jsonl(run_dir / "results.jsonl")[0]["official_metrics"]["SR"] == 0
