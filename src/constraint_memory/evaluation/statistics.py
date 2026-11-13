"""Paired, base-family clustered inference for official benchmark ratios."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import OFFICIAL_METRICS, require_metrics


class IncompletePairsError(ValueError):
    """The two conditions do not have the same group/replicate grid."""


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of no observations")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    if row.get("group_id") is None or str(row["group_id"]) == "":
        raise ValueError("Each observation requires group_id")
    return str(row["group_id"]), str(row.get("replicate_id", row.get("seed", 0)))


def _cluster(row: Mapping[str, Any]) -> str:
    family = row.get("base_template_id")
    if family is None or str(family) == "":
        raise ValueError("base_template_id is required for clustered inference")
    return str(family)


def clustered_bootstrap(
    values: Sequence[Sequence[float]],
    clusters: Sequence[str],
    *,
    samples: int = 10_000,
    seed: int = 20261013,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """Resample original base families, retaining every member/replicate.

    Point estimates are observation means. A sampled family contributes all its
    observations, including repeated run seeds, so correlated generated groups
    and replicates are never treated as independent clusters.
    """
    if not values or len(values) != len(clusters):
        raise ValueError("Values and clusters must be nonempty and aligned")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    width = len(values[0])
    if not width or any(len(row) != width for row in values):
        raise ValueError("Bootstrap observations must have equal nonzero width")
    if any(not math.isfinite(v) for row in values for v in row):
        raise ValueError("Bootstrap values must be finite")
    grouped: dict[str, list[Sequence[float]]] = defaultdict(list)
    for family, observation in zip(clusters, values, strict=True):
        grouped[str(family)].append(observation)
    sizes = [len(rows) for rows in grouped.values()]
    sums = [[sum(row[j] for row in rows) for j in range(width)] for rows in grouped.values()]
    draws: list[list[float]] = [[] for _ in range(width)]
    rng = random.Random(seed)
    families = len(sizes)
    if families == 1:
        # Every requested resample is identical; retain its exact interval
        # without spending 10,000 iterations on single-family smoke reports.
        draws = [[sums[0][j] / sizes[0]] for j in range(width)]
    else:
        for _ in range(samples):
            indices = rng.choices(range(families), k=families)
            denominator = sum(sizes[i] for i in indices)
            for j in range(width):
                draws[j].append(sum(sums[i][j] for i in indices) / denominator)
    alpha = (1 - confidence) / 2
    return [
        {
            "mean": sum(row[j] for row in values) / len(values),
            "ci_low": percentile(draws[j], alpha),
            "ci_high": percentile(draws[j], 1 - alpha),
            "confidence": confidence,
            "bootstrap_samples": samples,
            "cluster_count": families,
            "observation_count": len(values),
        }
        for j in range(width)
    ]


def exact_mcnemar(baseline: Sequence[float], treatment: Sequence[float]) -> dict[str, Any]:
    """Two-sided exact binomial McNemar test on paired binary group SR."""
    if not baseline or len(baseline) != len(treatment):
        raise ValueError("McNemar inputs must be nonempty aligned pairs")
    table = {"both_fail": 0, "baseline_only_success": 0, "treatment_only_success": 0, "both_success": 0}
    for before, after in zip(baseline, treatment, strict=True):
        if before not in (0, 1) or after not in (0, 1):
            raise ValueError("McNemar requires official binary group SR, not a fractional score")
        label = (
            "both_success" if before == after == 1 else
            "both_fail" if before == after == 0 else
            "baseline_only_success" if before == 1 else "treatment_only_success"
        )
        table[label] += 1
    b = table["baseline_only_success"]
    c = table["treatment_only_success"]
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    return {"contingency": table, "discordant_pairs": n, "p_value_two_sided_exact": p,
            "caveat": "McNemar does not account for correlated base families; clustered bootstrap is primary."}


def compare_runs(
    rows: Sequence[Mapping[str, Any]],
    baseline: str = "E0",
    treatment: str = "E2",
    *,
    requested_metrics: Sequence[str] = OFFICIAL_METRICS,
    bootstrap_samples: int = 10_000,
    seed: int = 20261013,
    primary_contrast: tuple[str, str] | None = None,
    primary_endpoint: str | None = None,
) -> dict[str, Any]:
    """Compare identical instances; only an explicit declaration is primary.

    The contrast alone does not declare an endpoint. Without both declaration
    fields, every measured endpoint remains secondary and unadjusted.
    """
    if baseline == treatment:
        raise ValueError("Baseline and treatment must differ")
    if (primary_contrast is None) != (primary_endpoint is None):
        raise ValueError("Primary contrast and primary endpoint must be declared together")
    if primary_contrast is not None:
        if isinstance(primary_contrast, str) or len(primary_contrast) != 2 or any(not isinstance(value, str) or not value for value in primary_contrast) or primary_contrast[0] == primary_contrast[1]:
            raise ValueError("Primary contrast must contain two distinct named conditions")
        primary_contrast = tuple(primary_contrast)
        if primary_endpoint not in OFFICIAL_METRICS:
            raise ValueError("Primary endpoint must be an explicitly supported official metric")
    by_variant: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {baseline: {}, treatment: {}}
    for row in rows:
        variant = row.get("variant")
        if variant not in by_variant:
            continue
        key = _identity(row)
        if key in by_variant[variant]:
            raise ValueError(f"Duplicate observation for {variant}: {key}")
        by_variant[variant][key] = row
    before, after = by_variant[baseline], by_variant[treatment]
    if not before or set(before) != set(after):
        missing_before = sorted(set(after) - set(before))
        missing_after = sorted(set(before) - set(after))
        raise IncompletePairsError(
            f"Paired comparison requires identical nonempty group/replicate grid; "
            f"missing {baseline}: {missing_before}; missing {treatment}: {missing_after}"
        )
    keys = sorted(before)
    clusters: list[str] = []
    observations: list[list[float]] = []
    pairs: list[dict[str, Any]] = []
    for key in keys:
        cluster = _cluster(before[key])
        if _cluster(after[key]) != cluster:
            raise ValueError(f"Mismatched base_template_id for pair {key}")
        left = require_metrics(before[key], requested_metrics)
        right = require_metrics(after[key], requested_metrics)
        clusters.append(cluster)
        observations.append([v for metric in requested_metrics for v in (left[metric], right[metric], right[metric] - left[metric])])
        pairs.append({"group_id": key[0], "replicate_id": key[1], "base_template_id": cluster,
                      "baseline": left, "treatment": right,
                      "differences": {metric: right[metric] - left[metric] for metric in requested_metrics}})
    summaries = clustered_bootstrap(observations, clusters, samples=bootstrap_samples, seed=seed)
    metrics: dict[str, Any] = {}
    for i, metric in enumerate(requested_metrics):
        left, right, delta = summaries[i * 3:i * 3 + 3]
        metrics[metric] = {
            "baseline": left, "treatment": right, "paired_difference": delta,
            "absolute_improvement_pp": delta["mean"] * 100,
            "relative_improvement_percent": delta["mean"] / left["mean"] * 100 if left["mean"] else None,
            "status": "primary" if (baseline, treatment) == primary_contrast and metric == primary_endpoint else "secondary_unadjusted",
            "aggregation": "unweighted mean of paired group ratios",
        }
    return {
        "baseline": baseline, "treatment": treatment,
        "primary_contrast": {"baseline": primary_contrast[0], "treatment": primary_contrast[1]} if primary_contrast else None,
        "primary_endpoint": primary_endpoint,
        "paired_groups": len({key[0] for key in keys}), "paired_observations": len(keys),
        "cluster_count": len(set(clusters)), "bootstrap_seed": seed,
        "bootstrap_method": "paired percentile bootstrap of original base-template families",
        "metrics": metrics, "per_group": pairs,
        "mcnemar": exact_mcnemar([p["baseline"]["SR"] for p in pairs], [p["treatment"]["SR"] for p in pairs]) if "SR" in requested_metrics else None,
        "secondary_comparisons": "All endpoints/contrasts except predeclared primary are labeled secondary_unadjusted.",
        "pilot_caveat": (
            "Only one base-template family is observed. Bootstrap resampling cannot estimate between-family uncertainty; "
            "zero-width intervals are descriptive and provide no evidence of superiority."
            if len(set(clusters)) == 1 else
            "Small pilots cannot establish conclusive superiority."
        ),
    }
