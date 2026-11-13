"""Offline lossless context-encoding experiment on development traces.

This script does not alter the running backend or measure planner performance.
Each table declares its columns, and a reversible symbol dictionary abbreviates
stable record/evidence references. Missing fields and nulls remain distinct.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from constraint_memory.config import sha256, verify_tokenizer
from constraint_memory.memory.ledger import LedgerMemory
from tokenizers import Tokenizer

PREFIXES = ("decision:", "constraint:", "definition:", "event:")


def encode(sections):
    symbols = {}

    def string(value):
        if value.startswith(PREFIXES):
            if value not in symbols:
                symbols[value] = f"~{len(symbols)}"
            return symbols[value]
        return "~" + value if value.startswith("~") else value

    def nested(value):
        if isinstance(value, str):
            return string(value)
        if isinstance(value, list):
            return [nested(item) for item in value]
        if isinstance(value, dict):
            return {string(key): nested(item) for key, item in value.items()}
        return value

    tables = {}
    for section, rows in sections.items():
        groups = {}
        for row in rows:
            columns = tuple(sorted(row))
            groups.setdefault(columns, []).append([nested(row[key]) for key in columns])
        tables[section] = [{"columns": list(columns), "rows": values} for columns, values in groups.items()]
    return {"symbols": {symbol: value for value, symbol in symbols.items()}, "tables": tables}


def decode(packed):
    symbols = packed["symbols"]

    def string(value):
        if value.startswith("~~"):
            return value[1:]
        if value.startswith("~"):
            return symbols[value]
        return value

    def nested(value):
        if isinstance(value, str):
            return string(value)
        if isinstance(value, list):
            return [nested(item) for item in value]
        if isinstance(value, dict):
            return {string(key): nested(item) for key, item in value.items()}
        return value

    return {section: [dict(zip(table["columns"], [nested(item) for item in row], strict=True))
                     for table in groups for row in table["rows"]]
            for section, groups in packed["tables"].items()}


def equivalent(first, second):
    # Table grouping may reorder rows, never their exact content or multiplicity.
    return {key: sorted(json.dumps(row, sort_keys=True) for row in rows) for key, rows in first.items()} == {
        key: sorted(json.dumps(row, sort_keys=True) for row in rows) for key, rows in second.items()}


class CaptureLedger(LedgerMemory):
    def _sections(self, records, definitions, outcomes, compact=False):
        sections = super()._sections(records, definitions, outcomes, compact)
        if compact:
            self.compact_sections = copy.deepcopy(sections)
        return sections


def audit(run_dir, source_variant):
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_relative_to(ROOT / "runs"):
        raise ValueError("A workspace run is required")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"]["experiment"]["partition"] not in {"smoke", "dev"}:
        raise ValueError("Only development traces are allowed")
    body = (run_dir / "memory_ops.jsonl").read_bytes()
    if not body.endswith(b"\n"):
        body = body.rsplit(b"\n", 1)[0] + b"\n"
    operations = [json.loads(line) for line in body.splitlines() if line]
    tokenizer_path, _ = verify_tokenizer(ROOT)
    if sha256(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from run")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    contexts = []
    memories = {}
    for operation_index, op in enumerate(operations):
        if op["variant"] != source_variant:
            continue
        if op["op"] == "reset":
            memory = CaptureLedger("E3", op["config"])
            memory.set_token_counter(lambda text: len(tokenizer.encode(text).ids), name="pinned Qwen tokenizer")
            memories[op["group_id"]] = memory
        memory = memories[op["group_id"]]
        memory.compact_sections = None
        result = getattr(memory, op["op"])(*op["args"]) if "args" in op else getattr(memory, op["op"])(op["arg"])
        if op["op"] != "begin_subtask" or memory.compact_sections is None:
            continue
        sections = memory.compact_sections
        packed = encode(sections)
        if not equivalent(sections, decode(packed)):
            raise ValueError("Binding table failed lossless decoding")
        text = "Exact binding tables. For each table, match row positions to columns. Strings ~N expand through symbols everywhere, including object keys. ~~ escapes a literal ~. Preserve all values, units, versions, edges and unresolved questions.\n" + json.dumps(packed, separators=(",", ":"), ensure_ascii=False)
        ordinary = memory._serialize(sections)
        contexts.append({"operation_index": operation_index, "traveler": op["arg"]["traveler"],
            "original_status": result["status"], "ordinary_compact_tokens": len(tokenizer.encode(ordinary).ids),
            "table_tokens": len(tokenizer.encode(text).ids), "budget": memory.budget,
            "required_records": len(result["required_closure_ids"]), "exact_roundtrip": True,
            "source_sections_sha256": hashlib.sha256(json.dumps(sections, sort_keys=True).encode()).hexdigest()})
    return {"diagnostic_only": True, "model_calls": 0, "planner_performance_measured": False,
        "run_id": manifest["run_id"], "source_variant": source_variant,
        "trace_prefix_sha256": hashlib.sha256(body).hexdigest(), "contexts": contexts,
        "limitation": "Encoding retains supplied rows exactly; it does not fix reference resolution or demonstrate that Qwen interprets the tables correctly."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-variant", default="E0")
    args = parser.parse_args()
    result = audit(args.run_dir, args.source_variant)
    output = args.run_dir / ("binding_table_diagnostic_" + args.source_variant + ".json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
