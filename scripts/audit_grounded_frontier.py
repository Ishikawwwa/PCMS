"""Development-only causal context audit on fixed, recorded actions.

Compare archived E3, alias/city-fixed E3, and the actual E3F development candidate.
Current and earlier/base choices are grounded from tool rows available at each
commit, with the same replacement and deduplication rules as GroupSession.
No model calls, hidden answers, database reads, or counterfactual scores occur.
Input run logs are never modified. Private module names keep imports isolated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from archive_run_source import verify_archive, run_scope
from constraint_memory.config import verify_tokenizer
from tokenizers import Tokenizer

FILES = {
    "memory.backends": "src/constraint_memory/memory/backends.py",
    "memory.grounding": "src/constraint_memory/memory/grounding.py",
    "memory.ledger": "src/constraint_memory/memory/ledger.py",
    "extraction": "src/constraint_memory/extraction.py",
}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def load_private(sources, prefix):
    """Load known pure local modules under unique names, never replace globals."""
    for suffix in ("", ".memory"):
        module = types.ModuleType(prefix + suffix)
        module.__path__ = []
        sys.modules[module.__name__] = module
    loaded = {}
    for suffix, rel in FILES.items():
        name = prefix + "." + suffix
        module = types.ModuleType(name)
        module.__package__ = name.rsplit(".", 1)[0]
        module.__file__ = "private-source:" + rel
        sys.modules[name] = module
        exec(compile(sources[rel].decode("utf-8"), module.__file__, "exec"), module.__dict__)
        loaded[suffix] = module
    return loaded


def capture_sections(cls):
    class Captured(cls):
        def _sections(self, records, definitions, outcomes, compact=False):
            sections = super()._sections(records, definitions, outcomes, compact)
            self.section_observations.append({"compact": compact,
                "tokens": self._token_count(self._serialize(sections)),
                "row_count": len(records), "sections": copy.deepcopy(sections)})
            return sections
    return Captured


def load_log(path):
    body = path.read_bytes()
    if body and not body.endswith(b"\n"):
        raise ValueError("A completed log must end with a full JSONL record")
    return [json.loads(line) for line in body.splitlines() if line], {"sha256": digest(body), "bytes": len(body)}


def causal_row_index(operations, traces, source_variant):
    """Bind rows to genuine observed events, activate them only at that event."""
    commits = {op["result"]["action_event_id"]: op["arg"]["step"] for op in operations
               if op["op"] == "commit_subtask" and op.get("result", {}).get("action_event_id")}
    observed = [op["arg"] for op in operations if op["op"] == "observe"]
    call_steps = {}
    for event in observed:
        if event["type"] == "tool_call":
            call = json.loads(event["content"])
            call_steps[(event["group_id"], call["id"])] = event["step"]
    results = [event for event in observed if event["type"] == "tool_result"]
    indexed = {}
    last_result_step = {}
    for trace in traces:
        if trace["variant"] != source_variant:
            continue
        group = trace["group_id"]
        result = trace["result"]
        boundary = commits[trace["commit"]["action_event_id"]]
        scratch = {output["tool_call_id"]: output["result"] for step in result["scratchpad"]
                   for output in step.get("tool_results", [])}
        for output in result.get("structured_tool_results", []):
            content = output["formatted_result"]
            if scratch[output["tool_call_id"]] != content or digest(content.encode()) != output["formatted_result_sha256"]:
                raise ValueError("Structured rows do not match the literal planner-visible output")
            call_step = call_steps[(group, output["tool_call_id"])]
            matches = [event for event in results if event["group_id"] == group and event["content"] == content
                       and max(last_result_step.get(group, -1), call_step) < event["step"] <= boundary]
            if not matches:
                raise ValueError("Structured rows lack a matching causal observation")
            event = min(matches, key=lambda row: row["step"])
            last_result_step[group] = event["step"]
            key = (group, event["event_id"])
            if key in indexed:
                raise ValueError("One observation cannot be rebound to multiple outputs")
            indexed[key] = [{"event_id": event["event_id"], "tool_name": output["tool_name"],
                             "row": row, "source_content": content, "observed_step": event["step"]}
                            for row in output.get("displayed_rows", [])]
    return indexed


def augmented_commit(memory, arg, observed_rows, extractor):
    """Mirror GroupSession.previous+new grounding and same-value dedup exactly."""
    result = copy.deepcopy(arg)
    choices = extractor.parse_submission(arg.get("action", ""))
    recorded = [row for row in arg.get("decisions", []) if row.get("attribute", "value") == "value"]
    choice_key = lambda row: (row["traveler"], row["day"], row["slot"], row["value"])
    if [choice_key(row) for row in choices] != [choice_key(row) for row in recorded]:
        raise ValueError("Deterministic choices differ from the recorded accepted action")
    step = arg["step"]
    if any(row["observed_step"] >= step for row in observed_rows):
        raise ValueError("Current or future tool observations crossed the commit boundary")
    replaced = set(arg.get("replace_travelers", []))
    previous = memory.committed_decisions(step)
    selected = [row for row in previous if row.get("attribute", "value") == "value" and row["traveler"] not in replaced] + choices
    attributes = extractor.grounded_attributes(selected, observed_rows)
    previous_facts = {(row["traveler"], row["day"], row["slot"], row.get("attribute", "value")): row for row in previous}
    fresh = [row for row in attributes["decisions"] if row["traveler"] in replaced or
             previous_facts.get((row["traveler"], row["day"], row["slot"], row["attribute"]), {}).get("value") != row["value"]]
    result["decisions"] = choices + fresh
    # The actual public-base initialization precedes every tool observation and
    # has no replace_travelers key. Keep its recorded action/decisions untouched.
    details = {"traveler": arg["traveler"], "step": step, "choices_submitted": len(choices),
        "previous_choices_considered": sum(row.get("attribute", "value") == "value" for row in selected)-len(choices),
        "causal_rows_available": len(observed_rows), "grounded_properties_before_dedup": len(attributes["decisions"]),
        "properties_inserted": len(fresh), "prior_or_base_properties_inserted": sum(row["traveler"] not in replaced for row in fresh),
        "city_alias_properties_inserted": sum("state_annotation" in row.get("grounding_method", "") for row in fresh),
        "unresolved_grounding_count": len(attributes["unresolved"])}
    return result, details


def audit(run_dir, source_variant="E0"):
    _, run_dir = run_scope(ROOT, run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest["config"]["experiment"]["partition"] not in {"dev", "smoke"}:
        raise ValueError("Only completed development/smoke runs are accepted")
    archive_evidence = verify_archive(ROOT, run_dir)
    with zipfile.ZipFile(run_dir / "source_archive.zip") as archive:
        archived = {rel: archive.read(rel) for rel in FILES.values()}
    current = {rel: (ROOT / rel).read_bytes() for rel in FILES.values()}
    old = load_private(archived, "_cm_frontier_archived_" + digest(b"".join(archived.values()))[:12])
    fixed = load_private(current, "_cm_frontier_current_" + digest(b"".join(current.values()))[:12])
    frontier_rel = "src/constraint_memory/memory/frontier.py"
    current[frontier_rel] = (ROOT / frontier_rel).read_bytes()
    frontier_module = types.ModuleType(fixed["memory.ledger"].__package__ + ".frontier")
    frontier_module.__package__ = fixed["memory.ledger"].__package__
    frontier_module.__file__ = "private-source:" + frontier_rel
    exec(compile(current[frontier_rel].decode("utf-8"), frontier_module.__file__, "exec"), frontier_module.__dict__)
    classes = {"old_E3": old["memory.ledger"].LedgerMemory,
               "alias_city_fixed_E3": fixed["memory.ledger"].LedgerMemory, "frontier_E3F": frontier_module.FrontierMemory}
    extractors = {"old_E3": old["extraction"], "alias_city_fixed_E3": fixed["extraction"], "frontier_E3F": fixed["extraction"]}
    classes = {name: capture_sections(cls) for name, cls in classes.items()}
    all_ops, ops_evidence = load_log(run_dir / "memory_ops.jsonl")
    operations = [op for op in all_ops if op["variant"] == source_variant]
    traces, traces_evidence = load_log(run_dir / "traces.jsonl")
    index = causal_row_index(operations, traces, source_variant)
    tokenizer_path, _ = verify_tokenizer(ROOT)
    if digest(tokenizer_path.read_bytes()) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the original run")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    count_tokens = lambda text: len(tokenizer.encode(text).ids)
    states, observed_rows = {}, {}
    contexts, commits = [], []
    for operation_index, op in enumerate(operations):
        group = op["group_id"]
        if op["op"] == "reset":
            observed_rows[group] = []
            for name, cls in classes.items():
                options = {**op["config"], "backend": "E3F" if name == "frontier_E3F" else "E3", "token_budget": 4096}
                state = cls(options) if name == "frontier_E3F" else cls("E3", options)
                state.reset(group)
                state.set_token_counter(count_tokens, name="pinned Qwen3.6-27B tokenizer")
                state.section_observations = []
                states[(name, group)] = state
            continue
        if op["op"] == "observe":
            event = op["arg"]
            observed_rows[group].extend(index.get((group, event["event_id"]), []))
        for name in classes:
            state = states[(name, group)]
            state.section_observations = []
            if op["op"] == "commit_subtask":
                arg, properties = augmented_commit(state, op["arg"], observed_rows[group], extractors[name])
                result = state.commit_subtask(arg)
                commits.append({"operation_index": operation_index, "condition": name, "group_id": group,
                                **properties, "commit_status": result["status"],
                                "diagnostic_codes": [row["code"] for row in result["diagnostics"]]})
            else:
                result = getattr(state, op["op"])(*op["args"]) if "args" in op else getattr(state, op["op"])(op["arg"])
            if op["op"] != "begin_subtask":
                continue
            if result["token_count"] > 4096:
                raise ValueError("A condition exceeded the common visible memory cap")
            available = [row for row in state.committed_decisions(op["arg"]["step"]) if row.get("attribute", "value") != "value"]
            overflow = next((row for row in result["diagnostics"] if row["code"] == "BUDGET_OVERFLOW"), None)
            compact = [row for row in state.section_observations if row["compact"]]
            contexts.append({"operation_index": operation_index, "condition": name, "group_id": group,
                "traveler": op["arg"]["traveler"], "round_idx": op["arg"]["round_idx"], "step": op["arg"]["step"],
                "status": result["status"], "visible_tokens": result["token_count"], "budget": 4096,
                "required_context_tokens": overflow["required_token_estimate"] if overflow else result["token_count"],
                "compact_section_tokens": compact[-1]["tokens"] if compact else None,
                "required_records": len(result["required_closure_ids"]), "returned_records": len(result["closure_ids"]),
                "available_observed_properties": len(available),
                "available_properties_by_traveler": dict(Counter(row["traveler"] for row in available)),
                "missing_refs": result["missing_refs"], "coverage": result["coverage"],
                "diagnostic_codes": [row["code"] for row in result["diagnostics"]]})
    if any((ROOT / rel).read_bytes() != body for rel, body in current.items()):
        raise ValueError("Source changed during audit; rerun from a stable snapshot")
    summary = {name: {"arrivals": sum(row["condition"] == name and row["round_idx"] > 0 for row in contexts),
        "overflows": sum(row["condition"] == name and row["status"] == "BUDGET_OVERFLOW" for row in contexts),
        "max_required_tokens": max(row["required_context_tokens"] for row in contexts if row["condition"] == name),
        "max_visible_tokens": max(row["visible_tokens"] for row in contexts if row["condition"] == name),
        "properties_inserted": sum(row["properties_inserted"] for row in commits if row["condition"] == name),
        "prior_or_base_properties_inserted": sum(row["prior_or_base_properties_inserted"] for row in commits if row["condition"] == name)} for name in classes}
    return {"diagnostic_only": True, "model_calls": 0, "database_reads": 0, "hidden_answer_reads": 0,
        "planner_performance_measured": False, "counterfactual_score_measured": False,
        "run_id": manifest["run_id"], "source_variant": source_variant, "memory_cap": 4096,
        "trace_sha256": traces_evidence["sha256"], "source_logs": {"traces": traces_evidence, "operations": ops_evidence},
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "archive_evidence": {key: archive_evidence.get(key) for key in ("verified", "archive_sha256", "original_source_pin_verification", "original_source_verified")},
        "archived_module_sha256": {rel: digest(body) for rel, body in archived.items()},
        "current_module_sha256": {rel: digest(body) for rel, body in current.items()},
        "script_sha256": digest(Path(__file__).read_bytes()),
        "summary": summary, "contexts": contexts, "commits": commits,
        "limitations": ["All conditions receive the same observed E0 actions and writer extractions; their alternate planner behavior is unobserved.",
            "Old E3 retains original city/reference behavior; corrected conditions receive additional causally recoverable facts, so this is not a one-mechanism ablation.",
            "Fixed and frontier conditions mirror current GroupSession previous+new/dedup grounding; properties can be added to public-base and prior choices after later observations.",
            "Overflow does not stop the diagnostic feeding the recorded actions: that is intentionally a fixed-trace feasibility comparison, not runtime policy execution.",
            "The E3F candidate archives full historical dependencies but excludes past choices' requirement ancestry from planner retrieval; sufficiency remains untested.",
            "Original run provenance remains partial for the redacted secrets.py pin; these four loaded archived module pins were verified."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-variant", default="E0")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/r9_grounded_frontier_diagnostic.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "reports"):
        raise ValueError("Diagnostic output must be contained in workspace reports/")
    result = audit(args.run_dir, args.source_variant)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
