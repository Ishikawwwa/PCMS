"""Strict boundary for normalized scores from the official offline evaluator.

This module deliberately does not calculate travel-plan success. In particular,
the pinned MemoryArena SPS metric is not interchangeable with a custom slot
score (or a metric merely named ``sPS`` in a paper).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

OFFICIAL_METRICS = ("SR", "PS", "SPS")


class MissingMetricError(ValueError):
    """A requested official metric was not supplied by the evaluator."""


def require_metrics(
    record: Mapping[str, Any], requested: Sequence[str] = OFFICIAL_METRICS
) -> dict[str, float]:
    """Validate normalized official ratios without inventing missing scores.

    ``official_metrics`` or ``metrics`` may contain canonical names; otherwise
    the names are read directly from the record. Values must be ratios, not
    percentages. Provenance supplied by the adapter is retained by reports.
    """
    values = record.get("official_metrics", record.get("metrics", record))
    if not isinstance(values, Mapping):
        raise MissingMetricError("Official metrics must be a mapping")
    normalized: dict[str, float] = {}
    for metric in requested:
        if metric not in values or values[metric] is None:
            raise MissingMetricError(
                f"Missing requested official metric {metric!r} for group "
                f"{record.get('group_id', '<unknown>')!r}; a proxy is not substituted"
            )
        value = values[metric]
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{metric} must be a finite numeric ratio")
        if not 0 <= value <= 1:
            raise ValueError(f"{metric} must be a ratio in [0, 1], got {value!r}")
        normalized[metric] = float(value)
    return normalized


def normalize_official_metrics(
    raw: Mapping[str, Any],
    *,
    source_names: Mapping[str, str],
    evaluator_version: str,
    scale: str = "ratio",
    requested: Sequence[str] = OFFICIAL_METRICS,
) -> dict[str, Any]:
    """Explicitly map upstream names/units and preserve their definitions.

    Callers must declare source names; there is no case-folding or sPS alias.
    The upstream pinned scorer returns percentages and therefore uses
    ``scale='percent'``. This wrapper never runs an evaluator itself.
    """
    if not evaluator_version:
        raise ValueError("An exact evaluator version/commit must be recorded")
    if scale not in {"ratio", "percent"}:
        raise ValueError("scale must be 'ratio' or 'percent'")
    result: dict[str, Any] = {}
    for metric in requested:
        source = source_names.get(metric)
        if source is None or source not in raw or raw[source] is None:
            raise MissingMetricError(f"Missing declared upstream source for {metric}")
        value = raw[source]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Upstream {source} must be numeric")
        result[metric] = value / 100 if scale == "percent" else value
    result = require_metrics(result, requested)
    return {
        "official_metrics": result,
        "official_metric_metadata": {
            "evaluator_version": evaluator_version,
            "source_names": dict(source_names),
            "source_scale": scale,
            "SPS_is_published_sPS_equivalent": False,
        },
    }
