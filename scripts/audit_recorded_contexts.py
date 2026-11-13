"""Compare memory retrieval on one recorded causal trace, without model calls.

The alternate memories receive the same actual submissions and observations.
This measures context feasibility, not their counterfactual task performance.
No answer dataset, scoring annotations, or future requests are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constraint_memory.config import sha256, verify_tokenizer
from constraint_memory.memory import create_backend
from tokenizers import Tokenizer


def audit(run_dir: Path, source_variant: str, variants: list[str]) -> dict:
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_relative_to(ROOT / "runs"):
        raise ValueError("A workspace run is required")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"]["experiment"]["partition"] not in {"smoke", "dev"}:
        raise ValueError("Trace comparison is development-only")
    # A single read freezes the live log prefix. An incomplete final record is
    # excluded explicitly, rather than silently incorporating later operations.
    captured = (run_dir / "memory_ops.jsonl").read_bytes()
    complete = captured if captured.endswith(b"\n") else captured.rsplit(b"\n", 1)[0] + b"\n"
    operations = [json.loads(line) for line in complete.splitlines() if line]
    operations = [op for op in operations if op["variant"] == source_variant]
    if not operations:
        raise ValueError("Requested source variant has no recorded operations")
    tokenizer_path, _ = verify_tokenizer(ROOT)
    if sha256(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the run")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    contexts = []
    instances = {}
    for operation_index, op in enumerate(operations):
        for variant in variants:
            key = (variant, op["group_id"])
            if op["op"] == "reset":
                memory = create_backend(variant, {**op["config"], "backend": variant})
                memory.set_token_counter(lambda text: len(tokenizer.encode(text).ids), name="Qwen3.6-27B pinned tokenizer")
                instances[key] = memory
            memory = instances[key]
            result = getattr(memory, op["op"])(*op["args"]) if "args" in op else getattr(memory, op["op"])(op["arg"])
            if op["op"] == "begin_subtask":
                if result["token_count"] > result["token_budget"]:
                    raise ValueError("Alternate context violates the common visible-memory cap")
                contexts.append({"operation_index": operation_index, "variant": variant,
                    "group_id": op["group_id"], "traveler": op["arg"]["traveler"], "step": op["arg"]["step"],
                    "status": result["status"], "tokens": result["token_count"], "budget": result["token_budget"],
                    "coverage": result.get("coverage"), "required_records": len(result.get("required_closure_ids", [])),
                    "returned_records": len(result.get("closure_ids", [])), "missing_refs": result.get("missing_refs", []),
                    "diagnostic_codes": [item["code"] for item in result.get("diagnostics", [])]})
    return {"diagnostic_only": True, "memory_effect_measured": False, "model_calls": 0,
        "comparison": "alternate retrieval on the same actual causal trace; alternate planner behavior is not observed",
        "run_id": manifest["run_id"], "source_variant": source_variant,
        "source_log_prefix_sha256": hashlib.sha256(complete).hexdigest(), "captured_complete_bytes": len(complete),
        "excluded_incomplete_tail_bytes": len(captured) - len(complete), "source_operations": len(operations),
        "original_source_pin_verification": "partial" if "[REDACTED]" in manifest.get("source_sha256", {}).values() else "not_checked",
        "contexts": contexts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-variant", default="E0")
    parser.add_argument("--variants", nargs="+", default=["E0", "E2", "E3"])
    args = parser.parse_args()
    result = audit(args.run_dir, args.source_variant, args.variants)
    destination = args.run_dir / ("context_trace_diagnostic_" + args.source_variant + ".json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
