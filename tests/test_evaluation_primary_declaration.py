"""A comparison must not invent or silently change predeclared endpoints."""

import json

import pytest

from constraint_memory.evaluation import compare_runs, markdown_report, report_run


def rows():
    return [{"group_id": "6", "base_template_id": "one-base-family", "variant": variant,
             "official_metrics": {"SR": 0, "PS": 0, "SPS": .4 if variant == "E0" else .3}}
            for variant in ("E0", "E3")]


def test_named_comparison_without_declaration_remains_exploratory():
    comparison = compare_runs(rows(), "E0", "E3", bootstrap_samples=10)
    assert comparison["primary_contrast"] is None
    assert comparison["primary_endpoint"] is None
    assert all(metric["status"] == "secondary_unadjusted" for metric in comparison["metrics"].values())
    report = report_run(rows(), baseline="E0", treatment="E3", bootstrap_samples=10)
    assert "Exploratory comparison" in markdown_report(report)


def test_caller_declared_e3_sr_contrast_is_preserved_without_upgrading_sps():
    report = report_run(rows(), primary_contrast=("E0", "E3"), primary_endpoint="SR", bootstrap_samples=10)
    comparison = report["comparison"]
    assert comparison["primary_contrast"] == {"baseline": "E0", "treatment": "E3"}
    assert comparison["metrics"]["SR"]["status"] == "primary"
    assert comparison["metrics"]["PS"]["status"] == comparison["metrics"]["SPS"]["status"] == "secondary_unadjusted"
    assert report["primary_declaration_source"] == "explicit_arguments"
    assert "cannot estimate between-family uncertainty" in comparison["pilot_caveat"]


def test_saved_manifest_supplies_exact_contrast_and_primary_metric(tmp_path):
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows()) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"experiment": {
        "primary_contrast": ["E0", "E3"], "primary_endpoint": "PS"}}}), encoding="utf-8")
    report = report_run(tmp_path, tmp_path / "new-analysis", bootstrap_samples=10)
    comparison = report["comparison"]
    assert comparison["treatment"] == "E3"
    assert comparison["primary_endpoint"] == "PS"
    assert comparison["metrics"]["PS"]["status"] == "primary"
    assert comparison["metrics"]["SR"]["status"] == "secondary_unadjusted"
    assert report["primary_declaration_source"] == "run_manifest"


def test_other_requested_comparison_stays_secondary_to_saved_declaration():
    data = rows() + [{**rows()[1], "variant": "E2"}]
    report = report_run(data, baseline="E0", treatment="E3", primary_contrast=("E0", "E2"), primary_endpoint="SR", bootstrap_samples=10)
    assert all(metric["status"] == "secondary_unadjusted" for metric in report["comparison"]["metrics"].values())
    assert "Secondary comparison: E3 versus E0" in markdown_report(report)


@pytest.mark.parametrize("declaration", [{"primary_contrast": ("E0", "E3")}, {"primary_endpoint": "SR"}])
def test_partial_declaration_fails_explicitly(declaration):
    with pytest.raises(ValueError, match="declared together"):
        compare_runs(rows(), "E0", "E3", bootstrap_samples=10, **declaration)
    with pytest.raises(ValueError, match="declared together"):
        report_run(rows(), baseline="E0", treatment="E3", bootstrap_samples=10, **declaration)
