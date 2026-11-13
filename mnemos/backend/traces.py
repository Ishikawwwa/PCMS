"""Execution events are evidence inside a trace, never independent memories."""
import hashlib
import json


EXECUTION_KINDS = {"tool_call", "tool_result", "agent_action", "agent_output", "submission"}


def trace_record(records):
    events = [{"kind": r.get("event_kind", "observation"), "content": r["content"],
               "metadata": {k: v for k, v in r.items() if k not in {"content", "events"}}} for r in records]
    content = "\n\n".join(f"[{i + 1} · {e['kind']}]\n{e['content']}" for i, e in enumerate(events))
    occurrence = hashlib.sha256(json.dumps(events, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"content": content, "events": events, "event_kind": "execution_trace",
            "session": records[0].get("session", "execution"), "source_event_id": "trace:" + occurrence,
            "provenance": "Ordered execution trace; original event identities and tool arguments/results retained inside the trace.",
            "trace_schema": "mnemos/execution-trace-v1"}


def group_records(records):
    """Group recorded action segments without crossing their session boundaries."""
    sessions = {r.get("session", "execution") for r in records if r.get("event_kind") in EXECUTION_KINDS}
    grouped, order = {}, []
    for record in records:
        session = record.get("session", "execution")
        if session not in sessions:
            order.append(record)
        else:
            if session not in grouped:
                grouped[session] = []
                order.append(session)
            grouped[session].append(record)
    return [trace_record(grouped[item]) if isinstance(item, str) else item for item in order]


def legacy_records(graph):
    """Build a new trace workspace from an old event-per-node run, without editing it."""
    records = []
    for node in graph["nodes"]:
        if node["type"] != "trace":
            continue
        meta = node.get("source_metadata", {})
        kind = meta.get("kind", meta.get("event_kind", "observation"))
        session = node.get("session", "imported")
        if kind in EXECUTION_KINDS:
            session += ":decision-" + str(meta.get("step", meta.get("original_step", 0)))
        records.append({"content": node["content"], "event_kind": kind, "session": session,
                        "original_node_id": node["id"], "original_run_id": node["run_id"],
                        "source_event_id": node.get("source_identity", node["id"]), "original_metadata": meta})
    return group_records(records)
