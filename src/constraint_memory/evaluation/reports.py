"""JSON/CSV/Markdown research reports with explicit official score provenance."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .accounting import summarize_resources
from .metrics import OFFICIAL_METRICS, require_metrics
from .statistics import _cluster, _identity, clustered_bootstrap, compare_runs


def load_rows(run_dir_or_rows: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(run_dir_or_rows, (str, Path)):
        return [dict(row) for row in run_dir_or_rows]
    source = Path(run_dir_or_rows)
    if source.is_dir():
        candidates = [source / "results.jsonl", source / "results.json"]
        source = next((path for path in candidates if path.exists()), candidates[0])
    if not source.exists():
        raise FileNotFoundError(f"No result rows found at {source}")
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        loaded = json.loads(source.read_text(encoding="utf-8"))
        rows = loaded["results"] if isinstance(loaded, dict) and "results" in loaded else loaded
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Result file must contain a list/JSONL of group observations")
    return rows


def summarize_variants(
    rows: Sequence[Mapping[str, Any]],
    *,
    requested_metrics: Sequence[str] = OFFICIAL_METRICS,
    bootstrap_samples: int = 10_000,
    seed: int = 20261013,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, tuple[str, str]]] = set()
    for row in rows:
        if not row.get("variant"):
            raise ValueError("Each result observation requires variant")
        variant = str(row["variant"])
        identity = (variant, _identity(row))
        if identity in seen:
            raise ValueError(f"Duplicate result observation: {identity}")
        seen.add(identity)
        groups[variant].append(row)
    summaries: dict[str, Any] = {}
    for variant, items in sorted(groups.items()):
        metrics = [require_metrics(row, requested_metrics) for row in items]
        clusters = [_cluster(row) for row in items]
        bootstrap = clustered_bootstrap(
            [[values[metric] for metric in requested_metrics] for values in metrics],
            clusters, samples=bootstrap_samples, seed=seed,
        ) if requested_metrics else []
        metric_summary = {
            name: {**interval, "aggregation": "unweighted mean of group ratios"}
            for name, interval in zip(requested_metrics, bootstrap, strict=True)
        }
        if "PS" in metric_summary:
            metric_summary["PS"]["aggregation"] = "group-mean PS; official pooled PS is person weighted and reported separately"
        counts = [row.get("person_count", row.get("traveler_count")) for row in items]
        pooled_ps = None
        if "PS" in requested_metrics and all(isinstance(count, int) and not isinstance(count, bool) and count > 0 for count in counts):
            pooled_ps = sum(values["PS"] * count for values, count in zip(metrics, counts, strict=True)) / sum(counts)
        provenance = sorted({json.dumps(row.get("official_metric_metadata", row.get("score_metadata", {})), sort_keys=True) for row in items})
        summaries[variant] = {
            "observations": len(items), "distinct_groups": len({str(row["group_id"]) for row in items}),
            "base_template_families": len(set(clusters)), "metrics": metric_summary,
            "official_pooled_PS": pooled_ps,
            "official_pooled_PS_note": "person-weighted upstream PS; null when expected person counts are missing",
            "official_metric_provenance": [json.loads(value) for value in provenance],
            "resources": summarize_resources(items),
        }
    return summaries


def difficulty_breakdown(rows: Sequence[Mapping[str, Any]], requested_metrics: Sequence[str] = OFFICIAL_METRICS) -> dict[str, Any]:
    """Report observed group difficulty and traveler-position scores separately.

    Missing annotation dimensions are counted explicitly. Traveler scores must
    come from the official adapter; no group SR is copied into traveler rows.
    Traveler indices, when used to derive position, are zero based.
    """
    dimensions = ("dependency_depth", "constraint_type", "distinct_previous_travelers",
                  "recovered_slot_count", "compaction_events", "revised_dependency")
    buckets: dict[str, dict[str, list[Mapping[str, Any]]]] = {name: defaultdict(list) for name in dimensions}
    positions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    missing = {name: 0 for name in dimensions}
    for row in rows:
        metadata = row.get("metadata", {})
        for dimension in dimensions:
            value = row.get(dimension, metadata.get(dimension))
            if dimension == "dependency_depth" and value is None:
                value = row.get("max_dependency_depth", metadata.get("max_dependency_depth"))
            if value is None:
                missing[dimension] += 1
            else:
                buckets[dimension][f"{row['variant']}:{value}"].append(row)
        travelers = row.get("travelers", row.get("rounds", row.get("traveler_metrics", [])))
        for ordinal, traveler in enumerate(travelers):
            position = traveler.get("traveler_position")
            if position is None:
                index = traveler.get("traveler_index", ordinal)
                # Adapter's upstream-facing person indices are one based.
                # Runner-normalized traveler rows and rounds are zero based.
                if "traveler_metrics" in row and "travelers" not in row and "rounds" not in row:
                    index -= 1
                count = row.get("person_count", row.get("traveler_count", len(travelers)))
                if not isinstance(index, int) or not isinstance(count, int) or count <= 0 or not 0 <= index < count:
                    raise ValueError("Traveler position requires valid zero-based index and expected traveler count")
                position = ("early", "middle", "late")[min(2, int(3 * (index + 0.5) / count))]
            if position not in {"early", "middle", "late"}:
                raise ValueError(f"Invalid traveler_position {position!r}")
            positions[f"{row['variant']}:{position}"].append(traveler)

    def summarize(items: Sequence[Mapping[str, Any]], *, optional: bool = False) -> dict[str, Any]:
        means: dict[str, Any] = {}
        for metric in requested_metrics:
            observed = []
            for item in items:
                values = item.get("official_metrics", item.get("metrics", item))
                if optional and (metric not in values or values[metric] is None):
                    continue
                observed.append(require_metrics(item, (metric,))[metric])
            means[metric] = {"mean": sum(observed) / len(observed) if observed else None,
                             "observations": len(observed), "missing_observations": len(items) - len(observed)}
        return {"observations": len(items), "metrics": means}

    return {"group_difficulty": {name: {key: summarize(items) for key, items in sorted(groups.items())} for name, groups in buckets.items()},
            "missing_annotations": missing,
            "traveler_position": {key: summarize(items, optional=True) for key, items in sorted(positions.items())},
            "note": "Descriptive subgroup means; primary uncertainty uses paired base-family bootstrap. Missing traveler metrics stay null."}


def _csv_value(value: Any) -> Any:
    return json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "unavailable"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = ["# Constraint-memory experiment report", "",
             "Scores below are normalized official evaluator ratios. SPS preserves the pinned upstream definition; equivalence to published sPS has not been assumed.", "",
             "PS in the paired analysis and metric table is the unweighted mean of group PS. Official pooled person-weighted PS is recorded separately when expected traveler counts are available.", "",
             "| Variant | Groups / observations | Base families | SR (95% CI) | Group-mean PS (95% CI) | Official SPS (95% CI) | Pooled PS |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for variant, summary in report["variants"].items():
        scores = summary["metrics"]
        def score_text(name: str) -> str:
            metric = scores.get(name, {})
            text = _format(metric.get("mean"), percent=True)
            if metric:
                text += f" [{_format(metric['ci_low'], percent=True)}, {_format(metric['ci_high'], percent=True)}]"
            return text
        lines.append(f"| {variant} | {summary['distinct_groups']} / {summary['observations']} | {summary['base_template_families']} | {score_text('SR')} | {score_text('PS')} | {score_text('SPS')} | {_format(summary['official_pooled_PS'], percent=True)} |")
    comparison = report.get("comparison")
    if comparison:
        declaration = comparison.get("primary_contrast")
        if declaration and (comparison["baseline"], comparison["treatment"]) == (declaration["baseline"], declaration["treatment"]):
            description = f"Declared primary contrast: {comparison['treatment']} versus {comparison['baseline']}; primary endpoint: {comparison['primary_endpoint']}."
        elif declaration:
            description = f"Secondary comparison: {comparison['treatment']} versus {comparison['baseline']}. Declared primary contrast: {declaration['treatment']} versus {declaration['baseline']}; primary endpoint: {comparison['primary_endpoint']}."
        else:
            description = f"Exploratory comparison: {comparison['treatment']} versus {comparison['baseline']}; no primary contrast or endpoint was declared."
        lines.extend(["", description, "",
                      "| Metric | Treatment minus baseline (pp) | 95% paired cluster CI (pp) | Relative improvement | Status |",
                      "| --- | --- | --- | --- | --- |"])
        for metric, value in comparison["metrics"].items():
            delta = value["paired_difference"]
            relative = value["relative_improvement_percent"]
            lines.append(f"| {metric} | {value['absolute_improvement_pp']:+.2f} | [{delta['ci_low'] * 100:+.2f}, {delta['ci_high'] * 100:+.2f}] | {relative:+.2f}% | {value['status']} |" if relative is not None else f"| {metric} | {value['absolute_improvement_pp']:+.2f} | [{delta['ci_low'] * 100:+.2f}, {delta['ci_high'] * 100:+.2f}] | undefined (zero baseline) | {value['status']} |")
        if comparison["mcnemar"]:
            mcnemar = comparison["mcnemar"]
            lines.extend(["", f"Exact two-sided McNemar p = {mcnemar['p_value_two_sided_exact']:.6g}; discordant pairs = {mcnemar['discordant_pairs']}.",
                          f"Paired contingency: `{json.dumps(mcnemar['contingency'], sort_keys=True)}`.", mcnemar["caveat"]])
        lines.extend(["", comparison["secondary_comparisons"], comparison["pilot_caveat"]])
    else:
        lines.extend(["", "A paired comparison was not available in this result set."])
    lines.extend(["", "## Resources", "", "| Variant | Agent tokens/group | Maintenance tokens/group | Total tokens/group | Provider RUB/group | Wall clock/group (s) |", "| --- | --- | --- | --- | --- | --- |"])
    for variant, summary in report["variants"].items():
        resources = summary["resources"]
        total, categories = resources["per_group"]["total"], resources["per_group"]
        show = lambda value: "unavailable" if value is None else str(value)
        lines.append(f"| {variant} | {show(categories['agent']['total_tokens'])} | {show(categories['maintenance']['total_tokens'])} | {show(total['total_tokens'])} | {show(total['cost_rub'])} | {show(resources['wall_clock_seconds']['mean_per_group'])} |")
    lines.extend(["", "Unknown token counts and provider charges remain unavailable; known partial totals and completeness counts are retained in JSON. Request latency and external tool latency are separated from end-to-end wall-clock time.", "",
                  "Depth, constraint-type, compaction, revised-state and early/middle/late traveler breakdowns are in the JSON report. Their subgroup means are descriptive, with missing annotations counted explicitly.", "",
                  "No claim of reproducing or beating MAGE follows from this report unless an actual matched MAGE run is included. No run infers benchmark success from a memory fixture.", ""])
    return "\n".join(lines)


def _write_svg_plots(report: Mapping[str, Any], output_dir: Path) -> None:
    """Dependency-free exportable plots; no interpolation of missing metrics."""
    points = []
    for variant, summary in report["variants"].items():
        sr = summary["metrics"].get("SR", {}).get("mean")
        tokens = summary["resources"]["total"]["total_tokens"]
        if sr is not None and tokens is not None:
            points.append((variant, tokens / summary["observations"], sr))
    if points:
        max_tokens = max(point[1] for point in points) or 1
        body = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">', '<rect width="760" height="460" fill="white"/>', '<path d="M80 35V380H710" fill="none" stroke="#334155"/>', '<text x="270" y="435" font-family="sans-serif">Total model tokens / group (agent + maintenance)</text>', '<text x="14" y="22" font-family="sans-serif">Official group SR (%)</text>']
        for fraction in (0, .25, .5, .75, 1):
            y = 380 - fraction * 330
            body.append(f'<text x="35" y="{y + 5}" font-family="sans-serif">{fraction * 100:g}</text><path d="M80 {y}H710" stroke="#e2e8f0"/>')
        for name, tokens, sr in points:
            x, y = 80 + tokens / max_tokens * 580, 380 - sr * 330
            body.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#2563eb"/><text x="{x + 9}" y="{y - 8}" font-family="sans-serif">{html.escape(name)} ({tokens:.0f})</text>')
        body.append('</svg>')
        (output_dir / "cost_performance.svg").write_text("\n".join(body), encoding="utf-8")
    depths = report["breakdowns"]["group_difficulty"]["dependency_depth"]
    if depths:
        body = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">', '<rect width="760" height="460" fill="white"/>', '<path d="M80 35V380H710" fill="none" stroke="#334155"/>', '<text x="230" y="435" font-family="sans-serif">Official SR by maximum dependency depth</text>']
        width = 590 / max(1, len(depths))
        for i, (name, summary) in enumerate(sorted(depths.items())):
            sr = summary["metrics"].get("SR", {}).get("mean")
            if sr is None:
                continue
            x, height = 90 + i * width, sr * 330
            body.append(f'<rect x="{x}" y="{380 - height}" width="{max(4, width - 10)}" height="{height}" fill="#2563eb"/><text x="{x}" y="405" font-size="11" font-family="sans-serif">{html.escape(name)}</text><text x="{x}" y="{370 - height}" font-size="11" font-family="sans-serif">{sr * 100:.1f}%</text>')
        body.append('</svg>')
        (output_dir / "sr_by_depth.svg").write_text("\n".join(body), encoding="utf-8")


def report_run(
    run_dir_or_rows: str | Path | Sequence[Mapping[str, Any]],
    output_dir: str | Path | None = None,
    *,
    requested_metrics: Sequence[str] = OFFICIAL_METRICS,
    baseline: str | None = None,
    treatment: str | None = None,
    primary_contrast: tuple[str, str] | None = None,
    primary_endpoint: str | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 20261013,
    write_parquet: bool = False,
    write_plots: bool = True,
    ledger: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    declaration_source = "explicit_arguments" if primary_contrast is not None or primary_endpoint is not None else None
    if declaration_source is None and isinstance(run_dir_or_rows, (str, Path)):
        source = Path(run_dir_or_rows)
        manifest_path = (source if source.is_dir() else source.parent) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            experiment = manifest.get("config", {}).get("experiment", {})
            primary_contrast = experiment.get("primary_contrast")
            primary_endpoint = experiment.get("primary_endpoint")
            if primary_contrast is not None or primary_endpoint is not None:
                declaration_source = "run_manifest"
    if (primary_contrast is None) != (primary_endpoint is None):
        raise ValueError("Primary contrast and primary endpoint must be declared together")
    if primary_contrast is not None:
        if isinstance(primary_contrast, str) or len(primary_contrast) != 2 or any(not isinstance(value, str) or not value for value in primary_contrast) or primary_contrast[0] == primary_contrast[1]:
            raise ValueError("Primary contrast must contain two distinct named conditions")
        primary_contrast = tuple(primary_contrast)
        if primary_endpoint not in OFFICIAL_METRICS:
            raise ValueError("Primary endpoint must be an explicitly supported official metric")
    baseline = baseline or (primary_contrast[0] if primary_contrast else "E0")
    treatment = treatment or (primary_contrast[1] if primary_contrast else "E2")
    rows = load_rows(run_dir_or_rows)
    if not rows:
        raise ValueError("Cannot report an empty experiment")
    variants = summarize_variants(rows, requested_metrics=requested_metrics, bootstrap_samples=bootstrap_samples, seed=seed)
    comparison = compare_runs(rows, baseline, treatment, requested_metrics=requested_metrics, bootstrap_samples=bootstrap_samples, seed=seed,
                              primary_contrast=primary_contrast, primary_endpoint=primary_endpoint) if baseline in variants and treatment in variants and requested_metrics else None
    report = {"schema_version": 1, "official_metric_names": list(requested_metrics),
              "SPS_is_published_sPS_equivalent": False, "bootstrap_samples": bootstrap_samples,
              "bootstrap_seed": seed, "confidence": .95,
              "primary_declaration_source": declaration_source,
              "variants": variants, "comparison": comparison,
              "breakdowns": difficulty_breakdown(rows, requested_metrics),
              "resources": summarize_resources(rows, ledger=ledger)}
    if output_dir is None and isinstance(run_dir_or_rows, (str, Path)):
        source = Path(run_dir_or_rows)
        output_dir = (source if source.is_dir() else source.parent) / "reports"
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (target / "report.md").write_text(markdown_report(report), encoding="utf-8")
        _write_csv(target / "per_group.csv", rows)
        if comparison:
            _write_csv(target / "paired_differences.csv", comparison["per_group"])
        if write_plots:
            _write_svg_plots(report, target)
        if write_parquet:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as error:
                raise RuntimeError("Parquet export was requested but pyarrow is unavailable; install the parquet extra") from error
            pq.write_table(pa.Table.from_pylist(rows), target / "per_group.parquet")
    elif write_parquet:
        raise ValueError("Parquet export requires output_dir")
    return report
