"""Entity-grounded proposal engine. Preview never changes accepted graph state."""
import copy
import hashlib
import itertools
import json
import time
from collections import defaultdict

import numpy as np

from .algorithms import normalized, projection
from .grounded_schema import DESIGN, PROMPT, SCHEMAS, TYPES, INSTRUCTIONS
from .models import Settings
from .store import Conflict, encoded, identity, now

ELIGIBLE = {"accepted", "promoted", "recorded", "resolved"}


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def config_id(run):
    return digest(run["settings"])


def source_spans(graph, spans):
    nodes = {n["id"]: n for n in graph["nodes"]}
    if not spans:
        raise ValueError("Resolvable source provenance is required")
    for span in spans:
        trace = nodes.get(span["trace_id"], {})
        event = next((e for e in trace.get("events", []) if e["id"] == span["event_id"]), None)
        if trace.get("type") != "trace" or event is None:
            raise ValueError("Unknown trace/event provenance")
        if not 0 <= span["start"] < span["end"] <= len(event["content"]):
            raise ValueError("Invalid provenance offsets")
        if event["content"][span["start"]:span["end"]] != span["quote"]:
            raise ValueError("Provenance quote does not match exact source bytes")
    return spans


def descendants(graph, ids):
    found = set(ids)
    while True:
        extra = {r["source"] for r in graph["links"] if r["type"] == "DERIVED_FROM" and r["target"] in found}
        if extra <= found:
            break
        found |= extra
    return found


def roots(graph, ids):
    nodes = {n["id"]: n for n in graph["nodes"]}
    result, seen, pending = set(), set(), list(ids)
    while pending:
        nid = pending.pop()
        if nid in seen:
            continue
        seen.add(nid)
        n = nodes.get(nid, {})
        if n.get("type") == "episode":
            result.add(nid)
        pending.extend(r["target"] for r in graph["links"] if r["source"] == nid and r["type"] == "DERIVED_FROM")
    return sorted(result)


def envelope(kind, title, command, version, **fields):
    return {"id": identity(kind), "type": kind, "title": title, "created_at": now(),
            "created_operation_id": command.operation_id, "created_graph_version": version + 1,
            "created_by_event": command.operation_id, "status": "candidate", "version": 1,
            "level": {"trace": 0, "entity": 1, "fact": 1, "state": 1, "episode": 1, "concept": 2, "rule": 3}.get(kind, 2),
            "metadata": {"design": DESIGN}, **fields}


def relation(source, kind, target, scope, operation, provenance=()):
    return {"id": identity("relation"), "type": kind, "source": source, "target": target,
            "scope": scope, "status": "accepted", "valid_from": None, "valid_to": None,
            "provenance": list(provenance), "created_operation_id": operation,
            "created_at": now(), "metadata": {"vocabulary": DESIGN}}


def compatible_embeddings(graph, nodes):
    embeddings = {e["id"]: e for e in graph.get("embeddings", []) if e["status"] == "accepted"}
    pairs = [(n, embeddings.get(n.get("embedding_ref"))) for n in nodes]
    pairs = [(n, e) for n, e in pairs if e]
    for n, e in pairs:
        vector = np.asarray(e["vector"], dtype=float)
        if vector.ndim != 1 or len(vector) != e["dimension"] or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Invalid embedding vector/dimension")
        if e.get("subject_id", n["id"]) != n["id"]:
            raise ValueError("Embedding subject mismatch")
    identities = {(e["model"], e["revision"], e["dimension"]) for n, e in pairs}
    if len(identities) > 1:
        raise ValueError("Incompatible embedding models/revisions/dimensions; re-embed or isolate")
    return pairs


def discover_grounded(graph, settings):
    nodes = [n for n in graph["nodes"] if n["type"] in {"concept", "rule"} and n["status"] in ELIGIBLE and n["level"] < settings.max_depth]
    pairs = compatible_embeddings(graph, nodes)
    if len(pairs) < 3:
        return [], {"reason": "At least three compatible accepted concepts/rules are required", "near_misses": []}
    ns, es = zip(*pairs)
    matrix = normalized([e["vector"] for e in es]) @ normalized([e["vector"] for e in es]).T
    groups, misses, seen = [], [], set()
    examined = 0
    for i, n in enumerate(ns):
        neighbors = sorted((j for j in range(len(ns)) if j != i and matrix[i, j] >= settings.threshold), key=lambda j: (-matrix[i, j], ns[j]["id"]))[:settings.top_k]
        for size in range(max(3, settings.min_size), settings.max_size + 1):
            for other in itertools.combinations(neighbors, size - 1):
                ids = tuple(sorted((i, *other), key=lambda j: ns[j]["id"]))
                key = tuple(ns[j]["id"] for j in ids)
                if key in seen:
                    continue
                seen.add(key)
                examined += 1
                if examined > settings.candidate_cap * 100:
                    break
                values = [float(matrix[a, b]) for a, b in itertools.combinations(ids, 2)]
                minimum, mean = min(values), sum(values) / len(values)
                if minimum < settings.min_pairwise or mean < settings.mean_pairwise:
                    if len(misses) < 50:
                        misses.append({"member_ids": key, "minimum": minimum, "mean": mean, "reason": "all-pairs thresholds"})
                    continue
                groups.append({"member_ids": list(key), "pairwise_similarity": matrix[np.ix_(ids, ids)].tolist(),
                               "min_pairwise_cosine": minimum, "mean_pairwise_cosine": mean, "cohesion": mean,
                               "member_embedding_ids": [es[j]["id"] for j in ids], "grouping_method": "top-k/all-pairs-v1",
                               "threshold_snapshot": {**{k: getattr(settings, k) for k in ("threshold", "min_pairwise", "mean_pairwise", "top_k", "max_size")}, "min_size": max(3, settings.min_size)},
                               "root_episode_ids": roots(graph, key), "level": max(ns[j]["level"] for j in ids),
                               "scope": sorted({ns[j].get("scope", "") for j in ids})})
            if examined > settings.candidate_cap * 100:
                break
        if examined > settings.candidate_cap * 100:
            break
    existing = {tuple(h["member_ids"]) for h in graph["hyperedges"] if h.get("status") in ELIGIBLE}
    groups = sorted((g for g in groups if tuple(g["member_ids"]) not in existing), key=lambda g: (-g["cohesion"], g["member_ids"]))[:settings.candidate_cap]
    return groups, {"near_misses": misses, "examined_groups": examined, "search_capped": examined > settings.candidate_cap * 100, "overlap_allowed": True}


class GroundedEngine:
    def __init__(self, store, models):
        self.store, self.models = store, models
        with store.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS grounded_operations(id TEXT,run_id TEXT,data TEXT,PRIMARY KEY(run_id,id))")
            db.execute("CREATE TABLE IF NOT EXISTS grounded_retrievals(id TEXT PRIMARY KEY,run_id TEXT,data TEXT)")

    def get(self, run_id, operation_id):
        with self.store.connection() as db:
            row = db.execute("SELECT data FROM grounded_operations WHERE run_id=? AND id=?", (run_id, operation_id)).fetchone()
        if not row:
            raise KeyError("Operation not found")
        return json.loads(row["data"])

    def save(self, run_id, value):
        with self.store.connection() as db:
            db.execute("INSERT OR REPLACE INTO grounded_operations VALUES(?,?,?)", (value["operation_id"], run_id, encoded(value)))

    def generate(self, run, command, job_id, kind, data, record_schema=None):
        settings = Settings(**run["settings"])
        if settings.prompt_version != PROMPT:
            raise ValueError("Frozen prompt version is unavailable; explicitly branch before recomputation")
        if settings.mode != "polza" and not getattr(self.models, "test_only", False):
            raise ValueError("Research extraction requires configured real generation; synthetic adapters are test-only")
        operations = set(SCHEMAS) | {"grounded_validation", "planner", "agent"}
        used = [r for r in self.store.receipts(run["id"]) if r["data"].get("operation") in operations and not r["data"].get("cache_hit")]
        if any(r["data"].get("cost_unit") == "RUB" and r["data"].get("cost") is None for r in used):
            raise ValueError("Previous provider billing is unresolved; reconcile before another paid operation")
        if len(used) >= settings.max_model_calls:
            raise ValueError("Run model-call budget exhausted")
        return self.models.grounded_generate(kind, data, settings,
                  lambda value: self.store.receipt(run["id"], job_id, value), record_schema)

    def preview(self, run_id, command, job_id):
        if not self.store.transition(run_id, job_id, "running"):
            return
        run = self.store.run(run_id)
        value = {"operation_id": command.operation_id, "job_id": job_id, "command": command.model_dump(), "base_graph_version": run["version"],
                 "config_id": config_id(run), "started_at": now(), "prompt_version": PROMPT, "status": "running", "proposals": [], "diagnostics": {}, "usage": {}}
        self.save(run_id, value)
        started = time.perf_counter()
        try:
            if command.graph_version != run["version"] or command.config_id != config_id(run):
                raise Conflict("Stale graph/config snapshot")
            proposals, diagnostics = self.propose(run, command, job_id)
            if self.store.job(run_id, job_id)["status"] == "cancelled":
                value.update(status="cancelled", proposals=[])
            else:
                value.update(status="success" if proposals else "no_op", proposals=proposals, diagnostics=diagnostics)
            with self.store.connection() as db:
                db.execute("UPDATE jobs SET status='succeeded',result=? WHERE id=? AND status='running'", (encoded({"action": command.type, "preview_operation_id": command.operation_id}), job_id))
                db.execute("UPDATE runs SET status='ready' WHERE id=?", (run_id,))
                self.store.event(db, run_id, "job.succeeded", {"job_id": job_id, "preview_operation_id": command.operation_id})
        except Exception as error:
            value.update(status="failed", error=str(error), proposals=[])
            self.store.transition(run_id, job_id, "failed", str(error))
        finally:
            value.update(ended_at=now(), wall_seconds=time.perf_counter() - started)
            calls = [r for r in self.store.receipts(run_id) if r["job_id"] == job_id]
            value["usage"] = {"call_ids": [r["id"] for r in calls], "input_tokens": sum(r["data"].get("usage", {}).get("input_tokens") or 0 for r in calls),
                              "output_tokens": sum(r["data"].get("usage", {}).get("output_tokens") or 0 for r in calls),
                              "cost_rub": sum(r["data"].get("cost") or 0 for r in calls if r["data"].get("cost_unit") == "RUB"),
                              "unknown_billing_calls": sum(r["data"].get("cost_unit") == "RUB" and r["data"].get("cost") is None for r in calls)}
            self.save(run_id, value)

    def propose(self, run, command, job_id):
        graph, p = run["graph"], command.parameters
        settings = Settings(**run["settings"])
        nodes = {n["id"]: n for n in graph["nodes"]}
        if any(i not in nodes and not any(h["id"] == i for h in graph["hyperedges"]) for i in command.input_ids):
            raise ValueError("Unknown input object")
        selected = [nodes[i] for i in command.input_ids if i in nodes] if command.input_ids else list(nodes.values())
        proposals, diagnostics = [], {}
        def add(kind, obj, **extra):
            proposals.append({"id": obj.get("id", identity(kind)), "kind": kind, "object": obj, **extra})
        if command.type == "import_traces":
            records = p.get("records", [])
            if p.get("source_run_id"):
                original = self.store.run(p["source_run_id"])
                records = [{"content": n.get("content", ""), "events": n.get("events") or n.get("source_metadata", {}).get("events"),
                            "session": n.get("session_id", n.get("session", "imported")), "metadata": {"original_run_id": original["id"], "original_node_id": n["id"]},
                            "scope": n.get("scope", "case:" + str(n.get("source_metadata", {}).get("group_id", original["id"])))} for n in original["graph"]["nodes"] if n["type"] == "trace"]
                if p.get("scope"):
                    records = [record for record in records if record["scope"] == p["scope"]]
            if not isinstance(records, list):
                raise ValueError("records must be an array")
            seen = {n.get("content_hash") for n in nodes.values() if n["type"] == "trace"}
            duplicates = []
            for index, record in enumerate(records):
                content = record.get("content")
                if not isinstance(content, str) or not content:
                    raise ValueError(f"Record {index}: exact content is required")
                session = record.get("session_id", record.get("session", "imported"))
                raw_events = record.get("events") or [{"kind": record.get("event_kind", "observation"), "content": content}]
                events = []
                for i, e in enumerate(raw_events):
                    text = e.get("content")
                    if not isinstance(text, str):
                        raise ValueError("Event content must be exact text")
                    events.append({"id": e.get("id", "event:" + str(i)), "kind": e.get("kind", e.get("type", "observation")), "role": e.get("role"), "content": text,
                                   "observed_at": e.get("observed_at"), "source_ref": e.get("source_ref", record.get("source_event_id")), "content_hash": hashlib.sha256(text.encode()).hexdigest(), "start": 0, "end": len(text)})
                checksum = digest({"session": session, "content": content, "events": events, "scope": record.get("scope", session)})
                if checksum in seen:
                    duplicates.append(index)
                    continue
                seen.add(checksum)
                obj = envelope("trace", content[:80], command, run["version"], content=content, events=events, session_id=session, session=session,
                               scope=record.get("scope", session), content_hash=checksum, status="recorded", metadata={"design": DESIGN, **record.get("metadata", {})})
                add("node", obj)
            diagnostics = {"duplicates": duplicates, "parser_errors": [], "input_count": len(records), "source_text_in_preview": True}
        elif command.type == "revise_proposal":
            original = nodes.get(command.input_ids[0]) if command.input_ids else None
            if original is None:
                prior = self.get(run["id"], p["source_operation_id"])
                original = next(v["object"] for v in prior["proposals"] if v["id"] == p["proposal_id"])
            if original["type"] == "trace":
                raise ValueError("Trace content is immutable; import a corrected source instead")
            schema = next(schema for op, schema in SCHEMAS.items() if TYPES[op] == original["type"])
            changes = p.get("changes", {})
            if not changes or not set(changes) <= set(schema.model_fields):
                raise ValueError("Supply explicit changes to valid object schema fields")
            fields = schema.model_validate({**{k: original.get(k) for k in schema.model_fields if k in original}, **changes}).model_dump()
            source_spans(graph, fields["provenance"])
            title = fields.pop("title")
            obj = envelope(original["type"], title, command, run["version"], **fields)
            obj.update(content=fields.get("summary", fields.get("content", fields.get("proposition", fields.get("instruction", encoded(fields))))),
                       evidence_ids=sorted({s["trace_id"] for s in fields["provenance"]}), metadata={"design": DESIGN, "previous_proposal_id": original["id"]})
            if original["id"] in nodes:
                obj["supersedes_id"] = original["id"]
            if original["type"] == "rule":
                obj.update(origin_hyperedge_id=original.get("origin_hyperedge_id"), origin_method=original.get("origin_method"), hypothesis_status="hypothesis", level=original["level"])
            report = self.generate(run, command, job_id, "grounded_validation", {"candidates": [obj], "sources": [n for n in nodes.values() if n["type"] == "trace"]})
            decision = next(v for v in report["decisions"] if v["id"] == obj["id"])
            if decision.get("unsupported") or decision.get("contradictions"):
                decision["passed"] = False
            obj["validation_record"] = decision
            if not decision["passed"]:
                obj["status"] = "needs_review"
            add("node", obj, validation=decision)
        elif command.type in SCHEMAS:
            sources = [n for n in nodes.values() if n["type"] == "trace" and n["status"] in ELIGIBLE and n.get("events")]
            if command.type == "propose_episodes":
                sources = [n for n in selected if n["type"] == "trace" and n["status"] in ELIGIBLE]
            scope = p.get("scope")
            if scope:
                sources = [n for n in sources if n.get("scope") == scope]
            if p.get("source_trace_ids"):
                sources = [n for n in sources if n["id"] in p["source_trace_ids"]]
            if not sources:
                return [], {"reason": "No accepted traces with event provenance in selected scope"}
            accepted = [n for n in selected if n["status"] in ELIGIBLE and (not scope or n.get("scope") == scope)]
            data = {"sources": [{k: n.get(k) for k in ("id", "events", "scope", "session_id")} for n in sources],
                    "selected_objects": [{k: v for k, v in n.items() if k not in {"vector", "embedding_input"}} for n in accepted],
                    "entities": [n for n in nodes.values() if n["type"] == "entity" and n["status"] in ELIGIBLE and not n.get("same_as") and (not scope or n.get("scope") == scope)], "scope": scope,
                    "branch_id": p.get("branch_id", "main"), "task_id": p.get("task_id")}
            if command.type == "propose_rule":
                edges = [h for h in graph["hyperedges"] if h["id"] in command.input_ids and h["status"] in ELIGIBLE]
                if len(edges) != 1 or len(edges[0]["member_ids"]) < 3:
                    raise ValueError("Select one accepted hyperedge with at least three distinct members")
                data["hyperedge"] = edges[0]
                data["members"] = [nodes[i] for i in edges[0]["member_ids"]]
                if max(n["level"] for n in data["members"]) + 1 > settings.max_depth:
                    raise ValueError("Rule depth cap exceeded")
            schema = SCHEMAS[command.type]
            output_schema = schema.model_json_schema()
            # The constrained decoder chooses exact native spans, including
            # whitespace, rather than asking a model to reproduce byte offsets.
            native_spans = []
            for trace in data["sources"]:
                for event in trace["events"]:
                    content = event["content"]
                    fragments = [(0, content)] if len(content) <= 600 else [(start, content[start:start+400]) for start in range(0, len(content), 400)]
                    if len(fragments) > 6:
                        fragments = [fragments[round(i*(len(fragments)-1)/5)] for i in range(6)]
                    for start, quote in fragments:
                        if quote:
                            native_spans.append({"trace_id": trace["id"], "event_id": event["id"], "start": start, "end": start + len(quote), "quote": quote})
            if len(native_spans) > 128:
                diagnostics["span_choices_capped"] = {"available": len(native_spans), "supplied": 128, "policy": "Explicit bounded span-choice pool; abstain on unavailable support"}
                native_spans = [native_spans[round(i*(len(native_spans)-1)/127)] for i in range(128)]
            output_schema.setdefault("$defs", {})["Span"] = {"anyOf": [{"type": "object", "properties": {k: {"type": "integer" if isinstance(v, int) else "string", "enum": [v]} for k, v in span.items()}, "required": list(span), "additionalProperties": False} for span in native_spans]}
            data["allowed_provenance_spans"] = native_spans
            existing_semantics = [n for n in nodes.values() if n["type"] in {"concept", "rule"} and n["status"] in ELIGIBLE and (not scope or n.get("scope") == scope)]
            data["existing_semantic_objects"] = [{k: n.get(k) for k in ("id", "type", "title", "proposition", "instruction", "scope")} for n in existing_semantics]
            output = self.generate(run, command, job_id, command.type, data, output_schema)
            if set(output) != {"records", "abstention"} or not isinstance(output["records"], list) or len(output["records"]) > 32:
                raise ValueError("Malformed extraction envelope")
            diagnostics["abstention"] = output["abstention"]
            for raw in output["records"]:
                fields = schema.model_validate(raw).model_dump()
                source_spans(graph, fields["provenance"])
                if scope and fields["scope"] != scope:
                    raise ValueError("Extraction escaped selected scope")
                kind = TYPES[command.type]
                if kind in {"entity", "fact", "state", "episode"} and any(nodes[span["trace_id"]].get("scope") != fields["scope"] for span in fields["provenance"]):
                    raise ValueError("Concrete record scope must match its cited trace scope")
                if kind == "fact":
                    entity = nodes.get(fields["subject_id"], {})
                    if entity.get("type") != "entity" or entity.get("status") not in ELIGIBLE or entity.get("scope") != fields["scope"]:
                        raise ValueError("Fact needs a resolved, accepted subject in the same scope")
                    if fields["target_id"] and (nodes.get(fields["target_id"], {}).get("type") != "entity" or nodes[fields["target_id"]].get("scope") != fields["scope"]):
                        raise ValueError("Invalid or out-of-scope fact target")
                    if (fields["value_type"] == "unknown") != (fields["value"] is None and fields["target_id"] is None):
                        raise ValueError("Missing values must explicitly use unknown type")
                    if fields["value_type"] == "number" and type(fields["value"]) not in (float, int):
                        raise ValueError("Numeric facts need literal numeric values")
                    if fields["value_type"] == "boolean" and type(fields["value"]) is not bool:
                        raise ValueError("Boolean facts require a boolean value")
                    if fields["value_type"] == "string" and not isinstance(fields["value"], str):
                        raise ValueError("Text facts require a string value")
                    if fields["value_type"] == "entity" and not fields["target_id"]:
                        raise ValueError("Entity value needs a target")
                    if fields.get("supersedes_id"):
                        old = nodes.get(fields["supersedes_id"], {})
                        if any(old.get(k) != fields[k] for k in ("subject_id", "predicate", "scope")) or not fields.get("valid_from"):
                            raise ValueError("Supersession needs same subject/predicate/scope and an evidence-backed effective start")
                if kind == "state":
                    if any(i not in nodes for i in fields["dependencies"] + fields["entity_ids"]):
                        raise ValueError("State references unknown dependencies/entities")
                    if any(nodes[i].get("scope") != fields["scope"] for i in fields["dependencies"] + fields["entity_ids"]):
                        raise ValueError("State dependencies/entities cannot escape task scope")
                    if p.get("branch_id", "main") != fields["branch_id"]:
                        raise ValueError("State escaped selected branch")
                    if p.get("task_id") and p["task_id"] != fields["task_id"]:
                        raise ValueError("State escaped selected task")
                parents = fields.get("supporting_ids", [])
                if parents and any(i not in nodes or nodes[i]["status"] not in ELIGIBLE for i in parents):
                    raise ValueError("Supporting inputs must be accepted graph objects")
                if kind == "concept" and not parents:
                    raise ValueError("Concept needs supporting episodes, facts or states")
                if kind == "concept" and any(nodes[i]["type"] not in {"episode", "fact", "state"} for i in parents):
                    raise ValueError("Concept parents must be chosen episodes, facts or states")
                if kind == "entity":
                    candidates = [n["id"] for n in nodes.values() if n["type"] == "entity" and n.get("canonical_name", "").casefold() == fields["canonical_name"].casefold() and n.get("scope") == fields["scope"]]
                    fields["identity_candidates"] = sorted(set(fields["identity_candidates"] + candidates))
                    if fields["identity_candidates"]:
                        fields["resolution"] = "ambiguous"
                title = fields.pop("title")
                obj = envelope(kind, title, command, run["version"], **fields)
                obj["content"] = fields.get("summary", fields.get("content", fields.get("proposition", fields.get("instruction", encoded(fields)))))
                obj["evidence_ids"] = sorted({s["trace_id"] for s in fields["provenance"]})
                obj["root_episode_ids"] = roots(graph, parents)
                obj["independent_root_count"] = len({nodes[i].get("scope") for i in obj["root_episode_ids"]})
                if kind == "rule":
                    members = data.get("members", [nodes[i] for i in parents])
                    obj.update(origin_hyperedge_id=data.get("hyperedge", {}).get("id"), level=max((n["level"] for n in members), default=1) + 1,
                               hypothesis_status="hypothesis", independence_status="unestablished", origin_method="hypergraph" if command.type == "propose_rule" else "direct")
                    if obj["level"] > settings.max_depth:
                        raise ValueError("Rule depth cap exceeded")
                    if command.type == "propose_rule" and not set(data["hyperedge"]["member_ids"]) <= set(parents):
                        raise ValueError("Hypergraph rule must explain and cite all member concepts")
                    if any(n["id"] in descendants(graph, parents) for n in members if n["type"] == "rule"):
                        diagnostics["ancestry_checked"] = True
                add("node", obj)
            if proposals:
                report = self.generate(run, command, job_id, "grounded_validation", {"candidates": [v["object"] for v in proposals], "sources": data["sources"], "existing": existing_semantics})
                reports = report.get("decisions", [])
                if len(reports) != len(proposals) or {v.get("id") for v in reports} != {v["id"] for v in proposals}:
                    raise ValueError("Independent validator must examine every proposal exactly once")
                for proposal in proposals:
                    decision = next(v for v in reports if v["id"] == proposal["id"])
                    if type(decision.get("passed")) is not bool or not isinstance(decision.get("reason"), str):
                        raise ValueError("Malformed independent validation decision")
                    if decision.get("unsupported") or decision.get("contradictions") or proposal["object"]["type"] in {"concept", "rule"} and decision.get("novelty") is not True:
                        decision["passed"] = False
                    proposal["validation"] = decision
                    proposal["object"]["validation_record"] = decision
                    if not decision["passed"]:
                        proposal["object"]["status"] = "needs_review"
        elif command.type == "decide_identity":
            if len(command.input_ids) != 1 or nodes[command.input_ids[0]]["type"] != "entity" or not p.get("reason"):
                raise ValueError("Select one entity and supply the identity decision reason")
            n = nodes[command.input_ids[0]]
            match = p.get("same_as")
            if match:
                other = nodes.get(match, {})
                if other.get("type") != "entity" or other.get("scope") != n.get("scope") or other.get("entity_type") != n.get("entity_type"):
                    raise ValueError("Explicit identity target must match type and scope")
            add("update", {**n, "resolution": "resolved", "status": "accepted", "same_as": match, "identity_decision": {"reason": p["reason"], "same_as": match, "distinct": not bool(match)}})
        elif command.type == "embed":
            eligible = [n for n in selected if n["type"] in {"concept", "rule"} and n["status"] in ELIGIBLE]
            texts = [encoded({k: n.get(k) for k in ("type", "title", "proposition", "instruction", "scope", "applicability", "required_inputs", "exceptions")}) for n in eligible]
            if texts:
                vectors, model = self.models.embed(texts, settings)
                vectors = normalized(vectors).tolist()
                for n, text, vector in zip(eligible, texts, vectors):
                    e = {"id": identity("embedding"), "subject_id": n["id"], "input_text": text, "input_hash": hashlib.sha256(text.encode()).hexdigest(), "provider": "sentence-transformers", "model": settings.embedding_model, "revision": settings.embedding_revision, "dimension": len(vector), "normalization": "l2", "vector": vector, "status": "accepted", "operation_id": command.operation_id, "type": "embedding", "created_at": now(), "created_operation_id": command.operation_id, "created_graph_version": run["version"] + 1, "metadata": {"design": DESIGN}}
                    add("embedding", e)
                    add("update", {**n, "embedding_ref": e["id"], "embedding_model_id": model, "vector": vector, "embedding_input": text})
        elif command.type == "project":
            pairs = compatible_embeddings(graph, [n for n in selected if n["type"] in {"concept", "rule"} and n["status"] in ELIGIBLE])
            if pairs:
                fitted = projection([n["id"] for n, e in pairs], [e["vector"] for n, e in pairs])
                add("projection", {**fitted, "id": identity("projection"), "method": "PCA/SVD", "model": pairs[0][1]["model"], "revision": pairs[0][1]["revision"], "created_operation_id": command.operation_id, "input_embedding_ids": [e["id"] for n, e in pairs]})
        elif command.type == "propose_hyperedges":
            search_graph = graph
            if p.get("scope"):
                search_graph = {**graph, "nodes": [n for n in graph["nodes"] if n.get("scope") == p["scope"]]}
            groups, diagnostics = discover_grounded(search_graph, settings)
            for g in groups:
                add("hyperedge", envelope("hyperedge", "Semantic association", command, run["version"], **g))
        elif command.type in {"challenge_source", "revalidate", "validate_rule"}:
            if not command.input_ids:
                raise ValueError("Choose input objects")
            if command.type == "challenge_source":
                if not p.get("reason"):
                    raise ValueError("A source challenge requires a reason")
                affected = descendants(graph, command.input_ids)
                diagnostics["affected_ids"] = sorted(affected)
                for n in nodes.values():
                    if n["id"] in affected:
                        add("update", {**n, "status": "disputed" if n["id"] in command.input_ids else "needs_revalidation", "challenge_reason": p["reason"]})
                for h in graph["hyperedges"]:
                    if set(h["member_ids"]) & affected:
                        add("hyperedge_update", {**h, "status": "needs_revalidation"})
            else:
                sources = [n for n in nodes.values() if n["type"] == "trace"]
                report = self.generate(run, command, job_id, "grounded_validation", {"candidates": selected, "sources": sources})
                for n in selected:
                    r = next((v for v in report.get("decisions", []) if v.get("id") == n["id"]), None)
                    if not r:
                        raise ValueError("Missing revalidation decision")
                    if type(r.get("passed")) is not bool:
                        raise ValueError("Malformed revalidation decision")
                    parents = [r["target"] for r in graph["links"] if r["type"] == "DERIVED_FROM" and r["source"] == n["id"]]
                    healthy = all(nodes[i]["status"] in ELIGIBLE for i in parents)
                    supported = r["passed"] and not r.get("unsupported") and not r.get("contradictions")
                    add("update", {**n, "status": "accepted" if supported and healthy else "needs_revalidation", "validation_record": r})
        return proposals, diagnostics

    def commit(self, run_id, operation_id, body):
        value = self.get(run_id, operation_id)
        decision = body.model_dump()
        if value.get("commit"):
            if value["commit"] != decision:
                raise Conflict("Operation already committed with different decisions")
            return value
        if value["status"] not in {"success", "no_op"}:
            raise Conflict("Only completed previews can commit")
        run = self.store.run(run_id)
        if run["version"] != body.graph_version or run["version"] != value["base_graph_version"]:
            raise Conflict("Stale proposal; explicitly rebase or branch")
        if not set(body.accepted_ids) <= {p["id"] for p in value["proposals"]}:
            raise ValueError("Unknown accepted proposal")
        graph = copy.deepcopy(run["graph"])
        chosen = [p for p in value["proposals"] if p["id"] in body.accepted_ids]
        # Validate the whole selection before any write.
        for p in chosen:
            obj = p["object"]
            if p.get("validation") and not p["validation"]["passed"]:
                raise ValueError("Unsupported proposal cannot become accepted memory")
            if obj.get("provenance"):
                source_spans(graph, obj["provenance"])
            if p["kind"] == "node" and obj["type"] == "fact":
                subject = next((n for n in graph["nodes"] if n["id"] == obj["subject_id"]), {})
                if subject.get("type") != "entity" or subject.get("status") not in ELIGIBLE or subject.get("scope") != obj.get("scope"):
                    raise ValueError("Committed facts need a resolved same-scope entity subject")
            if p["kind"] == "node" and obj["type"] in {"fact", "state", "entity", "episode"}:
                known_sources = {n["id"]: n for n in graph["nodes"] if n["type"] == "trace"}
                if any(known_sources[s["trace_id"]].get("scope") != obj.get("scope") for s in obj.get("provenance", [])):
                    raise ValueError("Concrete record cannot change its source scope")
            if obj.get("type") == "entity" and obj.get("resolution") == "ambiguous":
                if obj["id"] in body.identity_decisions:
                    match = body.identity_decisions[obj["id"]]
                    existing = next((n for n in graph["nodes"] if n["id"] == match), None) if match else None
                    if match and (not existing or existing["type"] != "entity" or existing.get("scope") != obj.get("scope") or existing.get("entity_type") != obj.get("entity_type")):
                        raise ValueError("Identity decision requires a same-scope/type existing entity")
                    obj.update(resolution="resolved", same_as=match, identity_decision={"actor": body.reviewer, "reason": body.reason, "same_as": match})
                else:
                    obj["status"] = "ambiguous"  # Persist unresolved, never eligible as a fact subject.
        if value["command"]["type"] == "challenge_source" and len(chosen) != len(value["proposals"]):
            raise ValueError("Source challenge must invalidate its full dependency closure atomically")
        embedding_ids = {p["object"]["id"] for p in chosen if p["kind"] == "embedding"} | {e["id"] for e in graph.get("embeddings", [])}
        if any(p["kind"] == "update" and p["object"].get("embedding_ref") and p["object"]["embedding_ref"] not in embedding_ids for p in chosen):
            raise ValueError("Embedding update requires its embedding proposal")
        from .models import Action
        request = Action(action="attribute", targets=[], parameters={"grounded_commit": operation_id, "decision": decision}, expected_graph_version=run["version"], idempotency_key="grounded-commit:" + operation_id)
        active = next((j for j in self.store.jobs(run_id) if j["status"] == "running" and j["request"]["action"] == "benchmark"), None)
        if active:
            from .benchmark import session
            state = session(self.store, run_id)
            if state.get("pending_operation_id") != operation_id:
                raise Conflict("Only the benchmark's pending review operation may commit during execution")
            job = active
        else:
            job, fresh = self.store.reserve(run_id, request)
            self.store.transition(run_id, job["id"], "running")
        try:
            for p in chosen:
                obj = copy.deepcopy(p["object"])
                kind = p["kind"]
                if kind in {"node", "hyperedge"}:
                    if obj["status"] == "candidate":
                        obj["status"] = "accepted"
                    obj["review"] = {"actor": body.reviewer, "reason": body.reason, "at": now()}
                    graph["nodes" if kind == "node" else "hyperedges"].append(obj)
                    if kind == "node":
                        if obj["type"] == "entity" and obj.get("same_as"):
                            graph["links"].append(relation(obj["id"], "IDENTICAL_TO", obj["same_as"], obj["scope"], operation_id, obj["provenance"]))
                        if obj.get("supersedes_id"):
                            previous = next(n for n in graph["nodes"] if n["id"] == obj["supersedes_id"])
                            if obj["type"] == "fact" and obj.get("valid_from"):
                                previous["valid_to"] = obj["valid_from"]
                            else:
                                previous["status"] = "retired"
                            previous["version"] = previous.get("version", 1) + 1
                            graph["links"].append(relation(obj["id"], "SUPERSEDES", previous["id"], obj["scope"], operation_id, obj["provenance"]))
                            affected = descendants(graph, [previous["id"]]) - {previous["id"]}
                            for dependent in graph["nodes"]:
                                if dependent["id"] in affected:
                                    dependent["status"] = "needs_revalidation"
                        parents = set(obj.get("supporting_ids", [])) | set(obj.get("evidence_ids", [])) | {s["trace_id"] for s in obj.get("provenance", [])}
                        if obj.get("type") == "rule" and obj.get("origin_hyperedge_id"):
                            edge = next(h for h in graph["hyperedges"] if h["id"] == obj["origin_hyperedge_id"])
                            parents |= set(edge["member_ids"])
                        for target in sorted(parents):
                            graph["links"].append(relation(obj["id"], "DERIVED_FROM", target, obj.get("scope"), operation_id, obj.get("provenance", [])))
                        if obj["type"] == "fact":
                            graph["links"].append(relation(obj["subject_id"], "HAS_FACT", obj["id"], obj["scope"], operation_id, obj["provenance"]))
                            if obj.get("target_id"):
                                graph["links"].append(relation(obj["subject_id"], "PARTICIPATES_IN" if obj["predicate"] == "participates_in" else "APPLIES_TO", obj["target_id"], obj["scope"], operation_id, obj["provenance"]))
                        if obj["type"] == "state":
                            for eid in obj["entity_ids"]:
                                graph["links"].append(relation(eid, "HAS_STATE", obj["id"], obj["scope"], operation_id, obj["provenance"]))
                            for target in obj["dependencies"]:
                                graph["links"].append(relation(obj["id"], "DEPENDS_ON", target, obj["scope"], operation_id, obj["provenance"]))
                    else:
                        for member in obj["member_ids"]:
                            graph["links"].append(relation(member, "MEMBER_OF", obj["id"], obj.get("scope"), operation_id))
                elif kind in {"update", "hyperedge_update"}:
                    table = "nodes" if kind == "update" else "hyperedges"
                    previous = next(n for n in graph[table] if n["id"] == obj["id"])
                    if previous["type"] == "trace" and obj.get("content") != previous.get("content"):
                        raise ValueError("Source content is immutable; import a correction with supersession")
                    obj["version"] = previous.get("version", 1) + 1
                    obj["last_operation_id"] = operation_id
                    graph[table] = [obj if n["id"] == obj["id"] else n for n in graph[table]]
                elif kind == "embedding":
                    graph.setdefault("embeddings", []).append(obj)
                elif kind == "projection":
                    graph["projection"] = obj
            graph["schema_version"] = DESIGN
            result = {"action": value["command"]["type"], "status": "success" if chosen else "no_op", "created_ids": [p["id"] for p in chosen if p["kind"] in {"node", "hyperedge"}], "updated_ids": [p["id"] for p in chosen if p["kind"] == "update"], "operation_id": operation_id}
            self.store.commit(run_id, job["id"], run["version"], graph, result, identity("event"), final=not bool(active))
            value.update(commit=decision, committed_graph_version=run["version"] + 1, result=result)
            self.save(run_id, value)
            return value
        except Exception as error:
            if not active:
                self.store.transition(run_id, job["id"], "failed", str(error))
            raise
