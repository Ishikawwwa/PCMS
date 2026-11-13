"""Token, provider billing, latency, and storage accounting without fake zeros."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .statistics import percentile


def _number(value: Any, field: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number or null")
    if field.endswith("tokens") and int(value) != value:
        raise ValueError(f"{field} must be an integer or null")
    return value


def normalize_usage(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Accept the explicit Polza token names and internal ledger names.

    Billing is in RUB. Unknown provider cost remains null; no price estimate is
    silently substituted. Tool calls contribute latency separately from model
    maintenance operations and can explicitly declare zero model tokens.
    """
    operation = str(entry.get("operation", "unknown"))
    category = entry.get("category")
    if category is None:
        category = "tool" if entry.get("external_tool") else "agent" if operation == "agent" else "maintenance"
    if category not in {"agent", "maintenance", "tool"}:
        raise ValueError(f"Unknown accounting category {category!r}")
    input_tokens = _number(entry.get("input_tokens", entry.get("prompt_tokens")), "input_tokens")
    output_tokens = _number(entry.get("output_tokens", entry.get("completion_tokens")), "output_tokens")
    total_tokens = _number(entry.get("total_tokens"), "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    currency = entry.get("currency", "RUB")
    if entry.get("cost_rub") is not None and currency != "RUB":
        raise ValueError("cost_rub requires currency RUB")
    return {**dict(entry), "operation": operation, "category": category,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens,
            "cost_rub": _number(entry.get("cost_rub"), "cost_rub"), "currency": "RUB",
            "latency_seconds": _number(entry.get("latency_seconds"), "latency_seconds"),
            "external_tool": bool(entry.get("external_tool", category == "tool"))}


def _totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"calls": len(records), "currency": "RUB"}
    for field in ("input_tokens", "output_tokens", "total_tokens", "cost_rub", "latency_seconds"):
        known = [entry[field] for entry in records if entry[field] is not None]
        missing = len(records) - len(known)
        result[field] = sum(known) if records and missing == 0 else None
        result[f"known_{field}"] = sum(known) if known else None
        result[f"missing_{field}_calls"] = missing
    latencies = [entry["latency_seconds"] for entry in records if entry["latency_seconds"] is not None]
    result["latency_p50_seconds"] = percentile(latencies, 0.5) if latencies else None
    result["latency_p95_seconds"] = percentile(latencies, 0.95) if latencies else None
    return result


def aggregate_ledger(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = [normalize_usage(entry) for entry in entries]
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    operations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        categories[record["category"]].append(record)
        operations[record["operation"]].append(record)
    external_tools = [record for record in records if record["external_tool"]]
    return {
        "total": _totals(records),
        "categories": {name: _totals(categories.get(name, [])) for name in ("agent", "maintenance", "tool")},
        "operations": {name: _totals(items) for name, items in sorted(operations.items())},
        "external_tools": _totals(external_tools),
        "latency_note": "Summed request latency is not end-to-end wall-clock time; parallel calls may overlap.",
        "cost_note": "Provider-reported RUB charges only; incomplete billing is null, with known partial charges separate.",
    }


def summarize_resources(rows: Iterable[Mapping[str, Any]], *, ledger: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    records = list(rows)
    entries = list(ledger) if ledger is not None else [entry for row in records for entry in row.get("usage", [])]
    result = aggregate_ledger(entries)
    result["groups"] = len(records)
    # Token and billing totals retain their completeness rules when expressed
    # per observation; unavailable totals cannot become misleading averages.
    result["per_group"] = {
        category: {field: totals[field] / len(records) if records and totals[field] is not None else None
                   for field in ("input_tokens", "output_tokens", "total_tokens", "cost_rub")}
        for category, totals in {"total": result["total"], **result["categories"]}.items()
    }
    for field in ("wall_clock_seconds", "storage_bytes", "index_bytes", "record_count", "indexed_tokens"):
        values = [_number(row.get(field), field) for row in records]
        known = [value for value in values if value is not None]
        result[field] = {
            "total": sum(known) if values and len(known) == len(values) else None,
            "mean_per_group": sum(known) / len(known) if values and len(known) == len(values) else None,
            "known_total": sum(known) if known else None,
            "observed_groups": len(known), "missing_groups": len(values) - len(known),
        }
    wall = [row["wall_clock_seconds"] for row in records if row.get("wall_clock_seconds") is not None]
    result["wall_clock_seconds"]["p50"] = percentile(wall, 0.5) if wall else None
    result["wall_clock_seconds"]["p95"] = percentile(wall, 0.95) if wall else None
    return result
