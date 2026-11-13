"""Deterministic replay of recorded backend calls without models or answer keys."""
import hashlib
import json
from pathlib import Path
from constraint_memory.memory import create_backend
from .logging import read_jsonl, write_json


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",", ":")).encode()).hexdigest()


def replay_run(run_dir):
    run_dir=Path(run_dir)
    manifest=json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    tokenizer=None
    if manifest.get("tokenizer_sha256"):
        tokenizer_path=run_dir.resolve().parent.parent/"external/qwen-tokenizer/tokenizer.json"
        from constraint_memory.config import sha256
        if sha256(tokenizer_path)!=manifest["tokenizer_sha256"]:
            raise ValueError("Replay tokenizer differs from the original run")
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    instances={}
    for operation in read_jsonl(run_dir/"memory_ops.jsonl"):
        key=(operation["variant"],operation["group_id"])
        if operation["op"]=="reset":
            instances[key]=create_backend(key[0],operation["config"])
            if tokenizer:
                instances[key].set_token_counter(lambda text:len(tokenizer.encode(text).ids),name="Qwen3.6-27B pinned tokenizer")
        if key not in instances:
            raise ValueError("Memory event appears before namespace reset")
        if "args" in operation:
            getattr(instances[key],operation["op"])(*operation["args"])
        else:
            getattr(instances[key],operation["op"])(operation["arg"])
    results=[]
    for (variant,group_id),memory in instances.items():
        snapshot=json.loads((run_dir/f"snapshots/{variant}/{group_id}.json").read_text(encoding="utf-8"))
        actual=canonical_hash(memory.snapshot())
        expected=canonical_hash(snapshot)
        results.append({"variant":variant,"group_id":group_id,"match":actual==expected,"actual_sha256":actual,"expected_sha256":expected})
    write_json(run_dir/"replay.json",{"groups":results,"all_match":all(r["match"] for r in results)})
    if not all(r["match"] for r in results):
        raise ValueError("Replay diverges from recorded memory snapshot")
    return results
