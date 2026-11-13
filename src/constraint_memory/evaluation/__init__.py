"""Offline official-score reporting and memory-fidelity diagnostics."""

from .accounting import aggregate_ledger, normalize_usage, summarize_resources
from .diagnostics import (
    ERROR_TAXONOMY, compression_metrics, extraction_metrics, recovery_metrics,
    reliability_metrics, retrieval_metrics, validate_failure_report,
)
from .metrics import MissingMetricError, normalize_official_metrics, require_metrics
from .reports import difficulty_breakdown, load_rows, markdown_report, report_run, summarize_variants
from .statistics import IncompletePairsError, clustered_bootstrap, compare_runs, exact_mcnemar

__all__ = [
    "ERROR_TAXONOMY", "IncompletePairsError", "MissingMetricError",
    "aggregate_ledger", "clustered_bootstrap", "compare_runs", "compression_metrics",
    "difficulty_breakdown", "exact_mcnemar", "extraction_metrics", "load_rows",
    "markdown_report", "normalize_official_metrics", "normalize_usage", "recovery_metrics",
    "reliability_metrics", "report_run", "require_metrics", "retrieval_metrics",
    "summarize_resources", "summarize_variants", "validate_failure_report",
]
