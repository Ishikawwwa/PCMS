"""Development-only retrieval prototype; never a live experimental treatment.

Past actual submissions are materialized facts, distinct from requirements that
led to them. Their historical justification stays in the full ledger archive.
Only current requirements, consumed facts, and their exact entity proofs enter
the planner frontier. The script also tests the unambiguous :value alias fix.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from constraint_memory.config import sha256, verify_tokenizer
from constraint_memory.memory.ledger import LedgerMemory
from tokenizers import Tokenizer
from prototype_binding_tables import encode, decode, equivalent


class FrontierPrototype(LedgerMemory):
    walking_frontier = False

    def _named_travelers(self, query):
        # Ignore only a complete known-name companion-list line. Schema success
        # never proves semantic coverage: all other literal mentions retain the
        # existing lexical fallback, even with a successful writer response.
        names = {row["traveler"] for history in self.decisions.values() for row in history}
        names.update(task.get("scope", {}).get("traveler", "") for task in self.tasks.values())
        alternatives = "(?:" + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True) if name) + ")"
        listing = alternatives + r"(?:(?:,\s*(?:and\s+)?|\s+and\s+)" + alternatives + r")*\.?"
        lines = []
        for line in query.splitlines():
            match = re.fullmatch(r"I['’]m (?:going on this trip|traveling) with (.+)", line.strip())
            if match and re.fullmatch(listing, match[1]):
                continue
            lines.append(line)
        return super()._named_travelers("\n".join(lines))

    def _resolve(self, ref, step, diagnostics=True, allow_invalid=False):
        # Apply only the terminal value attribute, preserving explicit versions.
        ref = re.sub(r"^(decision:[^:]+:day[^:]+:[^:]+):value(@v\d+)?$", r"\1\2", ref)
        return super()._resolve(ref, step, diagnostics, allow_invalid)

    def _deps(self, record):
        if self.walking_frontier and "decision_id" in record:
            return [record["entity_ref"]] if record.get("entity_ref") else []
        return super()._deps(record)

    def _closure(self, seeds, step, *, retrieval=False):
        self.walking_frontier = retrieval
        try:
            return super()._closure(seeds, step, retrieval=retrieval)
        finally:
            self.walking_frontier = False

    def _sections(self, records, definitions, outcomes, compact=False):
        sections = super()._sections(records, definitions, outcomes, True)
        # Exact past facts remain visible; historical justifications are archived
        # edges, not live unresolved requirements in this planning frontier.
        for row in sections["KNOWN CONTEXT"]:
            row.pop("depends_on", None)
            if row.get("entity_ref"):
                row["depends_on"] = [row["entity_ref"]]
        self.frontier_sections = copy.deepcopy(sections)
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
    memories = {}
    contexts = []
    for operation_index, op in enumerate(operations):
        if op["variant"] != source_variant:
            continue
        if op["op"] == "reset":
            memory = FrontierPrototype("E3", op["config"])
            memory.set_token_counter(lambda text: len(tokenizer.encode(text).ids), name="pinned Qwen tokenizer")
            memories[op["group_id"]] = memory
        memory = memories[op["group_id"]]
        memory.frontier_sections = None
        result = getattr(memory, op["op"])(*op["args"]) if "args" in op else getattr(memory, op["op"])(op["arg"])
        if op["op"] != "begin_subtask":
            continue
        sections = memory.frontier_sections
        packed = encode(sections)
        if not equivalent(sections, decode(packed)):
            raise ValueError("Frontier table failed lossless decoding")
        header = "CURRENT BINDING FRONTIER. KNOWN CONTEXT contains actual submitted choices and properties grounded in observed tool rows, not proof that older requirements were satisfied. Older justification edges remain in the audit archive. Current ACTIVE CONSTRAINTS still apply. Each table maps row positions to columns; ~N expands through symbols everywhere, including object keys; ~~ escapes a literal ~.\n"
        text = header + json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
        contexts.append({"operation_index": operation_index, "traveler": op["arg"]["traveler"],
            "status": result["status"], "frontier_records": len(result["required_closure_ids"]),
            "row_tokens": len(tokenizer.encode(memory._serialize(sections)).ids), "table_tokens": len(tokenizer.encode(text).ids),
            "budget": memory.budget, "missing_refs": result["missing_refs"], "exact_frontier_roundtrip": True})
    return {"diagnostic_only": True, "model_calls": 0, "planner_performance_measured": False,
        "run_id": manifest["run_id"], "source_variant": source_variant,
        "trace_prefix_sha256": hashlib.sha256(body).hexdigest(), "contexts": contexts,
        "limitations": ["Frontier sufficiency has not been tested with live Qwen planning.",
            "Historical requirement ancestry is archived rather than included in this context.",
            "Only complete known-name companion-list lines are excluded from lexical seeds; other named mentions retain fallback.",
            "This combines the value-alias correction, companion-list filtering, and a retrieval change and is not a clean ablation."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-variant", default="E0")
    args = parser.parse_args()
    result = audit(args.run_dir, args.source_variant)
    output = args.run_dir / ("materialized_frontier_diagnostic_" + args.source_variant + ".json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
