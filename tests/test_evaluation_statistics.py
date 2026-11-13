"""Statistical invariants, independent of model/network calls."""

import pytest

from constraint_memory.evaluation import (
    IncompletePairsError, MissingMetricError, clustered_bootstrap, compare_runs,
    exact_mcnemar, normalize_official_metrics, require_metrics,
)


def observation(group, variant, sr, *, family=None, ps=None, sps=None, replicate=0):
    return {"group_id": group, "variant": variant, "base_template_id": family or group,
            "replicate_id": replicate,
            "official_metrics": {"SR": sr, "PS": sr if ps is None else ps, "SPS": sr if sps is None else sps}}


def test_missing_official_metric_is_not_substituted():
    with pytest.raises(MissingMetricError, match="SPS"):
        require_metrics({"group_id": "g", "SR": 1, "PS": .5, "sPS": .9})


def test_explicit_upstream_percentage_mapping_records_version_and_names():
    result = normalize_official_metrics({"sr": 50, "ps": 75, "sps": 80},
        source_names={"SR": "sr", "PS": "ps", "SPS": "sps"},
        evaluator_version="reviewed-commit", scale="percent")
    assert result["official_metrics"] == {"SR": .5, "PS": .75, "SPS": .8}
    assert result["official_metric_metadata"]["SPS_is_published_sPS_equivalent"] is False
    with pytest.raises(MissingMetricError):
        normalize_official_metrics({"sPS": .9}, source_names={"SPS": "sps"},
            evaluator_version="reviewed-commit", requested=("SPS",))


@pytest.mark.parametrize("value", [25, -1, float("nan"), float("inf"), "0.5"])
def test_reject_non_ratio_scores(value):
    with pytest.raises(ValueError):
        require_metrics({"SR": value}, ("SR",))


def test_cluster_members_are_never_resampled_individually():
    # There is one base family, so every resample must contain both related
    # variants. An observation-level bootstrap would incorrectly widen this CI.
    result = clustered_bootstrap([[0], [1]], ["same-family", "same-family"], samples=500)
    assert result[0]["mean"] == .5
    assert result[0]["ci_low"] == result[0]["ci_high"] == .5
    assert result[0]["cluster_count"] == 1


def test_cluster_bootstrap_keeps_unequal_family_sizes_and_is_reproducible():
    values, families = [[0], [0], [1]], ["a", "a", "b"]
    first = clustered_bootstrap(values, families, samples=500, seed=71)
    assert first == clustered_bootstrap(values, families, samples=500, seed=71)
    assert first[0]["mean"] == pytest.approx(1 / 3)
    assert first[0]["ci_low"] == 0
    assert first[0]["ci_high"] == 1


def test_paired_effect_cancels_difficulty_and_records_pp_relative_separately():
    rows = [observation("a", "E0", 0, ps=.2, sps=.4),
            observation("b", "E0", 1, ps=.6, sps=.7),
            observation("a", "E2", 1, ps=.3, sps=.5),
            observation("b", "E2", 1, ps=.7, sps=.8)]
    result = compare_runs(rows, bootstrap_samples=400, primary_contrast=("E0", "E2"), primary_endpoint="SR")
    assert result["metrics"]["SR"]["absolute_improvement_pp"] == 50
    assert result["metrics"]["SR"]["relative_improvement_percent"] == 100
    assert result["metrics"]["SR"]["status"] == "primary"
    assert result["metrics"]["PS"]["status"] == "secondary_unadjusted"
    delta = result["metrics"]["PS"]["paired_difference"]
    assert delta["ci_low"] == pytest.approx(.1)
    assert delta["ci_high"] == pytest.approx(.1)
    assert len(result["per_group"]) == 2


def test_missing_and_duplicate_pairs_fail_instead_of_dropping_observations():
    row = observation("a", "E0", 0)
    with pytest.raises(IncompletePairsError):
        compare_runs([row, observation("b", "E2", 1)], bootstrap_samples=10)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_runs([row, row, observation("a", "E2", 1)], bootstrap_samples=10)


def test_replicates_are_paired_and_clustered_under_original_family():
    rows = [observation("a", variant, sr, family="base", replicate=replicate)
            for replicate in (1, 2) for variant, sr in (("E0", 0), ("E2", 1))]
    result = compare_runs(rows, bootstrap_samples=30)
    assert result["paired_groups"] == 1
    assert result["paired_observations"] == 2
    assert result["cluster_count"] == 1
    assert result["metrics"]["SR"]["paired_difference"]["ci_low"] == 1
    with pytest.raises(IncompletePairsError):
        compare_runs(rows[:-1], bootstrap_samples=30)


def test_cluster_metadata_must_match_both_variants():
    with pytest.raises(ValueError, match="Mismatched"):
        compare_runs([observation("a", "E0", 0, family="x"), observation("a", "E2", 1, family="y")], bootstrap_samples=10)


def test_zero_baseline_relative_gain_is_undefined():
    result = compare_runs([observation("a", "E0", 0), observation("a", "E2", 1)], bootstrap_samples=10)
    assert result["metrics"]["SR"]["relative_improvement_percent"] is None


def test_exact_mcnemar_uses_discordances_only():
    result = exact_mcnemar([0] * 5 + [1, 0], [1] * 5 + [1, 0])
    assert result["p_value_two_sided_exact"] == .0625
    assert result["contingency"] == {"both_fail": 1, "both_success": 1,
        "baseline_only_success": 0, "treatment_only_success": 5}
    assert exact_mcnemar([1, 0], [1, 0])["p_value_two_sided_exact"] == 1
    with pytest.raises(ValueError, match="binary"):
        exact_mcnemar([.5], [1])
