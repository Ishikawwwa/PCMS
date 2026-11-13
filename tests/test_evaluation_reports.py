import builtins
import csv
import json

import pytest

from constraint_memory.evaluation import MissingMetricError, difficulty_breakdown, report_run, summarize_variants


def row(group, variant, sr, *, person_count=2, ps=.5):
    return {"group_id": group, "base_template_id": group, "variant": variant,
            "SR": sr, "PS": ps, "SPS": .6, "person_count": person_count,
            "dependency_depth": 3, "constraint_type": "JOIN", "wall_clock_seconds": 4,
            "usage": [{"operation": "agent", "prompt_tokens": 100, "completion_tokens": 20,
                       "cost_rub": .1, "latency_seconds": 2}],
            "official_metric_metadata": {"evaluator_version": "pinned-test-scorer"}}


def test_official_pooled_ps_is_distinguished_from_mean_group_ps():
    result = summarize_variants([row("a", "E0", 0, person_count=1, ps=0),
                                 row("b", "E0", 0, person_count=3, ps=1)], bootstrap_samples=100)
    assert result["E0"]["metrics"]["PS"]["mean"] == .5
    assert result["E0"]["official_pooled_PS"] == .75


def test_report_exports_raw_pairs_official_provenance_and_difficulty(tmp_path):
    rows = [row("a", "E0", 0), row("a", "E2", 1)]
    report = report_run(rows, tmp_path, bootstrap_samples=100)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["comparison"]["metrics"]["SR"]["absolute_improvement_pp"] == 100
    assert report["variants"]["E0"]["official_metric_provenance"] == [{"evaluator_version": "pinned-test-scorer"}]
    assert "E0:3" in report["breakdowns"]["group_difficulty"]["dependency_depth"]
    with (tmp_path / "paired_differences.csv").open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 1
    assert json.loads(pairs[0]["differences"])["SR"] == 1
    assert (tmp_path / "cost_performance.svg").exists()
    assert (tmp_path / "sr_by_depth.svg").exists()
    assert "undefined (zero baseline)" in (tmp_path / "report.md").read_text()


def test_optional_parquet_export(tmp_path):
    pytest.importorskip("pyarrow")
    report_run([row("a", "E0", 0)], tmp_path, bootstrap_samples=10, write_parquet=True)
    assert (tmp_path / "per_group.parquet").exists()


def test_requested_unavailable_parquet_fails_explicitly(tmp_path, monkeypatch):
    original_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("blocked optional dependency")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="pyarrow is unavailable"):
        report_run([row("a", "E0", 0)], tmp_path, bootstrap_samples=10, write_parquet=True)


def test_run_directory_loading_and_missing_metric_failure(tmp_path):
    (tmp_path / "results.jsonl").write_text(json.dumps(row("a", "E0", 0)) + "\n", encoding="utf-8")
    report = report_run(tmp_path, bootstrap_samples=10)
    assert report["comparison"] is None
    assert (tmp_path / "reports" / "report.md").exists()
    broken = row("a", "E0", 0)
    del broken["SPS"]
    with pytest.raises(MissingMetricError, match="SPS"):
        report_run([broken], bootstrap_samples=10)


def test_early_middle_late_official_traveler_scores_are_not_copied_from_group():
    group = row("a", "E2", 1, person_count=3)
    group["travelers"] = [{"traveler_index": i, "official_metrics": {"PS": score}} for i, score in enumerate((0, .5, 1))]
    report = difficulty_breakdown([group])
    assert report["traveler_position"]["E2:early"]["metrics"]["PS"]["mean"] == 0
    assert report["traveler_position"]["E2:middle"]["metrics"]["PS"]["mean"] == .5
    assert report["traveler_position"]["E2:late"]["metrics"]["PS"]["mean"] == 1
    assert report["traveler_position"]["E2:late"]["metrics"]["SR"]["mean"] is None
