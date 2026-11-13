"""Import public chronological MemoryArena inputs and observed execution events.

Hidden answers/scorer inputs are never opened. Corpus hashes identify exactly
which public source was imported, and input order survives deduplication.
"""
import hashlib
import json
from .traces import group_records
from pathlib import Path

ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "data" / "manifests" / "splits.json"
RECORDED = "runs/cross-family-continuation-20261016T234548697206Z/events.jsonl"
EVENT_TYPES = {"request", "task_contract", "property_schema", "tool_call", "tool_result", "agent_action"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def descriptors():
    if not MANIFEST.exists():
        return []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    result = []
    for gid, group in manifest["groups"].items():
        family = group["base_template_id"]
        partition = next((p for p in ("dev", "validation", "holdout") if family in manifest["families"][p]), "unknown")
        result.append({"id": f"memoryarena-{gid}", "name": f"MemoryArena {gid} · public inputs · {partition}", "source": "memoryarena", "group_id": gid, "partition": partition, "family_id": family, "record_count": len(group["turns"]) + 2, "description": f"Original base request, public base itinerary and {len(group['turns'])} chronological traveler requests. {partition} family. No gold answers, official scores, or invented actions.", "records": [], "corpus_kind": "public_inputs", "dataset_revision": "da1a37c8b19280e18627ca01cf368195a5e1d92e", "manifest_sha256": digest(MANIFEST.read_bytes())})
    if (ROOT / RECORDED).exists():
        result.insert(0, {"id": "memoryarena-recorded-120-E1", "name": "MemoryArena 120 · recorded E1 execution · dev", "source": "memoryarena", "group_id": "120", "partition": "dev", "description": "Actual chronological requests, agent actions and tool interactions from an existing E1 run. Historical execution; this does not execute a new travel planner or measure Mnemos task success.", "records": [], "corpus_kind": "recorded_execution", "source_path": RECORDED, "variant": "E1"})
    return result


def checked_file(relative, expected=None):
    path = (ROOT / relative).resolve()
    visible = (ROOT / "data" / "visible").resolve()
    if not path.is_relative_to(visible):
        raise ValueError("MemoryArena input must remain inside the public visible corpus")
    data = path.read_bytes()
    sha = digest(data)
    if expected and sha != expected:
        raise ValueError("Public MemoryArena source checksum changed: " + relative)
    return json.loads(data), {"source_path": relative, "source_file_sha256": sha}


def load(dataset):
    gid = dataset["group_id"]
    if dataset["corpus_kind"] == "recorded_execution":
        # Only the predeclared observed event archive is readable through this adapter.
        raw = (ROOT / RECORDED).read_bytes()
        records = []
        batch = 0
        for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            event = json.loads(line)
            if event.get("group_id") != gid or event.get("variant") != "E1" or event.get("type") not in EVENT_TYPES:
                continue
            # Record content and kind as ordinary observations, never as engine instructions.
            content = event["content"] if isinstance(event["content"], str) else json.dumps(event["content"], ensure_ascii=False, sort_keys=True)
            records.append({"content": content, "session": f"ma-{gid}-E1-segment-{batch:03d}", "sequence": len(records), "event_kind": event["type"], "source_event_id": event["event_id"], "original_step": event["step"], "group_id": gid, "variant": "E1", "provenance": "Actual recorded MemoryArena tool/planner event; synthetic benchmark environment.", "source_path": RECORDED, "source_file_sha256": digest(raw), "source_line": line_no, "source_line_sha256": digest(line.encode())})
            if event["type"] == "agent_action":
                batch += 1
        return group_records(records)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    group = manifest["groups"].get(gid)
    if not group:
        raise ValueError("Unknown MemoryArena group")
    base, origin = checked_file(group["base_path"], group["base_sha256"])
    shared = {"group_id": gid, "partition": dataset["partition"], "family_id": group["base_template_id"], "provenance": "Original public MemoryArena benchmark input; not an observed agent execution."}
    person = base["base_person"]
    records = [{**shared, **origin, "content": person["query"], "session": f"ma-{gid}-base", "sequence": 0, "event_kind": "base_request"}, {**shared, **origin, "content": "Public base itinerary (provided benchmark context):\n" + json.dumps(person["daily_plans"], ensure_ascii=False, indent=2), "session": f"ma-{gid}-base", "sequence": 1, "event_kind": "public_base_itinerary"}]
    for turn in sorted(group["turns"], key=lambda t: t["traveler_index"]):
        value, origin = checked_file(turn["path"], turn["sha256"])
        records.append({**shared, **origin, "content": value["query"], "session": f"ma-{gid}-traveler-{turn['traveler_index']:02d}", "sequence": len(records), "traveler_index": turn["traveler_index"], "event_kind": "traveler_request", "name": value["name"]})
    return records
