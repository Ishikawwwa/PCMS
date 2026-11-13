import copy
import hashlib
import json
import time
from collections import Counter

import numpy as np

from .adapters import Models, PROMPT_VERSION, canonical
from .algorithms import discover, normalized, projection
from .fixtures import datasets
from .models import Action, Settings
from .memoryarena import load as load_memoryarena
from .store import encoded, identity, now
from .traces import group_records


def source_ids(graph, object_id):
    """Follow derivation edges only, never semantic associations."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    visited, sources, pending = set(), set(), [object_id]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if nodes.get(current, {}).get("type") == "trace":
            sources.add(current)
        pending.extend(link["target"] for link in graph["links"] if link["source"] == current and link["type"] == "DERIVED_FROM")
    return sources


def impact(graph, target):
    affected = {target}
    while True:
        more = {link["source"] for link in graph["links"] if link["target"] in affected and link["type"] == "DERIVED_FROM"}
        if more <= affected:
            break
        affected |= more
    return {"node_ids": sorted(affected), "hyperedge_ids": [h["id"] for h in graph["hyperedges"] if set(h["member_ids"]) & affected]}


def validate_evidence(output, sources, episode=False):
    required = {"goal", "context", "events", "outcome", "source_ids", "evidence_quotes"} if episode else {"title", "proposition", "conditions", "exceptions", "evidence_ids", "evidence_quotes"}
    if set(output) != required:
        raise ValueError("Invalid model output fields: " + ", ".join(sorted(set(output) ^ required)))
    strings = ("goal", "context", "outcome") if episode else ("title", "proposition")
    if any(not isinstance(output[k], str) or not output[k].strip() for k in strings):
        raise ValueError("Model text fields must be nonempty strings")
    arrays = ("events",) if episode else ("conditions", "exceptions")
    if any(not isinstance(output[k], list) or not all(isinstance(v, str) and v.strip() for v in output[k]) for k in arrays):
        raise ValueError("Model scope and event fields must be arrays of text")
    ids = output["source_ids" if episode else "evidence_ids"]
    known = {s["id"]: s for s in sources}
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids) or len(ids) != len(set(ids)) or not set(ids) <= set(known):
        raise ValueError("Evidence must cite unique supplied original trace IDs")
    quotes = output["evidence_quotes"]
    if not isinstance(quotes, list) or not quotes:
        raise ValueError("Exact evidence quotes are required")
    cited = set()
    for item in quotes:
        if not isinstance(item, dict) or set(item) != {"source_id", "quote"}:
            raise ValueError("Malformed evidence quote")
        sid, quote = item["source_id"], item["quote"]
        if sid not in ids or not isinstance(quote, str) or not quote.strip() or quote not in known[sid]["content"]:
            raise ValueError("Evidence quote is not present in its cited original source")
        cited.add(sid)
    if cited != set(ids):
        raise ValueError("Each evidence source requires an exact quote")
    return output


def validate_report(report):
    if set(report) != {"passed", "reason", "unsupported", "contradictions", "novelty"} or type(report["passed"]) is not bool or type(report["novelty"]) is not bool or not isinstance(report["reason"], str):
        raise ValueError("Invalid independent validation output")
    if any(not isinstance(report[k], list) or not all(isinstance(v, str) for v in report[k]) for k in ("unsupported", "contradictions")):
        raise ValueError("Invalid validation diagnostics")
    return report


def validate_facts(output, sources):
    if set(output) != {"facts"} or not isinstance(output["facts"], list) or not 1 <= len(output["facts"]) <= 32:
        raise ValueError("Episode extraction requires 1–32 granular facts")
    seen = set()
    for fact in output["facts"]:
        if not isinstance(fact, dict) or set(fact) != {"title", "fact", "context", "source_ids", "evidence_quotes"}:
            raise ValueError("Each episode must be one contextualized fact with original evidence")
        if any(not isinstance(fact[k], str) or not fact[k].strip() for k in ("title", "fact", "context")):
            raise ValueError("Fact text and context must be nonempty")
        validate_evidence({"title": fact["title"], "proposition": fact["fact"], "conditions": [fact["context"]], "exceptions": [], "evidence_ids": fact["source_ids"], "evidence_quotes": fact["evidence_quotes"]}, sources)
        fingerprint = fact["fact"].strip().casefold()
        if fingerprint in seen:
            raise ValueError("Duplicate facts in one extraction")
        seen.add(fingerprint)
    return output


def attributed_fact(fact, sources):
    origins = set()
    for quote in fact["evidence_quotes"]:
        source = next(s for s in sources if s["id"] == quote["source_id"])
        meta = source.get("source_metadata", {})
        events = meta.get("events", [])
        if events:
            origins.update(e["kind"] for e in events if quote["quote"] in e["content"])
        else:
            origins.add(meta.get("kind", meta.get("event_kind", "source_observation")))
    roles = sorted(origins)
    if origins and origins <= {"agent_output", "agent_action", "submission"}:
        scope = "Agent-reported claim (not independently verified)"
    elif origins and origins <= {"tool_call"}:
        scope = "Attempted tool action"
    elif origins and origins <= {"tool_result"}:
        scope = "Observed tool result"
    else:
        scope = "Source-attributed observation"
    return {"content": scope + ": " + fact["fact"], "evidence_roles": roles, "evidence_scope": scope}


class Engine:
    def __init__(self, store, models=None):
        self.store = store
        self.models = models or Models()

    def execute(self, run_id, job_id):
        if not self.store.transition(run_id, job_id, "running"):
            return
        run = self.store.run(run_id)
        action = Action(**self.store.job(run_id, job_id)["request"])
        graph = copy.deepcopy(run["graph"])
        started = time.perf_counter()
        event_id = identity("event")
        try:
            result = self.operation(run, graph, action, job_id, event_id)
            result["wall_seconds"] = time.perf_counter() - started
            result["action"] = action.action
            self.store.commit(run_id, job_id, run["version"], graph, result, event_id)
        except Exception as error:
            self.store.transition(run_id, job_id, "failed", str(error))

    def operation(self, run, graph, action, job_id, event_id):
        settings = Settings(**run["settings"])
        nodes = {n["id"]: n for n in graph["nodes"]}
        edges = {h["id"]: h for h in graph["hyperedges"]}
        if any(t not in nodes and t not in edges for t in action.targets):
            raise ValueError("Unknown target object")
        p = action.parameters
        created, updated = [], []

        def model(op, data):
            key = hashlib.sha256(encoded({"operation": op, "input": data, "settings": settings.model_dump(), "prompt_version": PROMPT_VERSION}).encode()).hexdigest()
            with self.store.connection() as db:
                cached = db.execute("SELECT data FROM model_cache WHERE key=?", (key,)).fetchone()
            if cached:
                value = json.loads(cached["data"])
                self.store.receipt(run["id"], job_id, {"operation": op, "cache_hit": True, "cache_key": key, "original_audit": value["audit"], "output": value["output"], "usage": {"input_tokens": 0, "output_tokens": 0}, "wall_seconds": 0, "cost": 0})
                return value["output"]
            calls = [r for r in self.store.receipts(run["id"]) if r["data"].get("operation") in {"episode", "concept", "abstraction", "validation"} and not r["data"].get("cache_hit")]
            if len(calls) >= settings.max_model_calls:
                raise ValueError("Run model-call budget exhausted")
            audits = []
            def receipt(value):
                audits.append(copy.deepcopy(value))
                self.store.receipt(run["id"], job_id, value)
            output = self.models.generate(op, data, settings, receipt)
            if op == "episode":
                validate_facts(output, data["sources"])
            elif op == "concept":
                parents = output.get("episode_ids")
                if not isinstance(parents, list) or not all(isinstance(i, str) for i in parents) or len(parents) != len(set(parents)) or len(parents) < 2 or not set(parents) <= {e["id"] for e in data["episodes"]}:
                    raise ValueError("A concept must cite at least two unique supplied granular episodes")
                allowed = set().union(*(source_ids(graph, i) for i in parents))
                validate_evidence({k:v for k,v in output.items() if k != "episode_ids"}, [s for s in data["sources"] if s["id"] in allowed])
            elif op == "abstraction" and output.get("no_valid_abstraction") is not True:
                validate_evidence(output, data["sources"])
            elif op == "validation":
                validate_report(output)
            # Cache complete outputs; downstream schema/semantic gates still run on every use.
            with self.store.connection() as db:
                db.execute("INSERT OR IGNORE INTO model_cache VALUES(?,?)", (key, encoded({"output": output, "audit": audits[-1] if audits else None})))
            return output

        def add(kind, level, fields, parents=()):
            nid = identity(kind)
            node = {"id": nid, "run_id": run["id"], "type": kind, "level": level, "status": "candidate", "validation_state": "unverified", "version": 1, "source_version": run["version"], "created_at": now(), "created_by_event": event_id, "explanation": "Explicit " + action.action + " operation", "model_prompt_version": PROMPT_VERSION, **fields}
            graph["nodes"].append(node)
            nodes[nid] = node
            created.append(nid)
            for parent in parents:
                graph["links"].append({"id": identity("link"), "run_id": run["id"], "source": nid, "target": parent, "type": "DERIVED_FROM", "created_by_event": event_id})
            return node

        def update(obj, **fields):
            obj.update(fields, version=obj["version"] + 1, updated_by_event=event_id)
            updated.append(obj["id"])

        def sources_for(ids):
            lineage = set().union(*(source_ids(graph, nid) for nid in ids))
            return [nodes[i] for i in sorted(lineage)]

        def selected(kind=None, status=None):
            return [n for n in graph["nodes"] if (not action.targets or n["id"] in action.targets) and (kind is None or n["type"] in kind) and (status is None or n["status"] in status)]

        if action.action == "import":
            records = p.get("records")
            if "dataset_id" in p:
                with self.store.connection() as db:
                    row = db.execute("SELECT data FROM datasets WHERE id=?", (p["dataset_id"],)).fetchone()
                if not row:
                    raise ValueError("Unknown dataset")
                dataset = json.loads(row["data"])
                records = load_memoryarena(dataset) if dataset.get("source") == "memoryarena" else dataset["records"]
            if not isinstance(records, list) or not 1 <= len(records) <= 5000:
                raise ValueError("Import requires 1–5000 trace records")
            records = group_records(records)
            hashes = {n.get("source_identity", n["content_hash"]) for n in graph["nodes"] if n["type"] == "trace"}
            limit = p.get("count", len(records))
            if type(limit) is not int or not 1 <= limit <= 5000:
                raise ValueError("Source import count must be 1–5000")
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get("content"), str) or not record["content"].strip() or len(record["content"]) > 100000:
                    raise ValueError("Each trace needs nonempty content, at most 100000 characters")
                digest = hashlib.sha256(record["content"].strip().encode()).hexdigest()
                occurrence = hashlib.sha256(encoded([record.get("source_path"), record.get("source_file_sha256"), record["source_event_id"]]).encode()).hexdigest() if record.get("source_event_id") else digest
                if occurrence in hashes:
                    continue
                add("trace", 0, {"content": record["content"].strip(), "content_hash": digest, "source_identity": occurrence, "title": (str(record.get("event_kind", "")) + " · " if record.get("event_kind") else "") + record["content"][:65], "session": str(record.get("session", "imported")), "provenance": record.get("provenance", "user-imported observation; execution authenticity unverified"), "source_metadata": {k: v for k, v in record.items() if k != "content"}, "status": "recorded", "validation_state": "source_record"})
                hashes.add(occurrence)
                if len(created) >= limit:
                    break
        elif action.action == "episode":
            used = {i for n in graph["nodes"] if n["type"] == "episode" for i in n.get("extraction_trace_ids", n.get("evidence_ids", []))}
            traces = [n for n in selected({"trace"}, {"recorded"}) if n["id"] not in used]
            groups = {}
            for trace in traces:
                groups.setdefault(trace["session"], []).append(trace)
            count = p.get("count", 1)
            if type(count) is not int or not 1 <= count <= 20:
                raise ValueError("Episode batch count must be 1–20")
            batches = [group[i:i+6] for group in groups.values() for i in range(0, len(group), 6)]
            for group in batches[:count]:
                output = validate_facts(model("episode", {"sources": group}), group)
                for fact in output["facts"]:
                    add("episode", 1, {"title": fact["title"], **attributed_fact(fact, group), "episode": fact,
                        "evidence_ids": fact["source_ids"], "evidence_quotes": fact["evidence_quotes"],
                        "extraction_trace_ids": [n["id"] for n in group], "session": group[0]["session"],
                        "episode_schema": "mnemos/granular-fact-v1"}, fact["source_ids"])
        elif action.action == "attribute":
            for node in selected({"episode"}):
                if node.get("episode_schema") == "mnemos/granular-fact-v1" and "evidence_scope" not in node:
                    update(node, **attributed_fact(node["episode"], sources_for([node["id"]])))
        elif action.action == "concept":
            used = {link["target"] for link in graph["links"] if nodes.get(link["source"], {}).get("type") == "concept"}
            groups = {}
            for episode in selected({"episode"}, {"validated"}):
                if episode["id"] not in used:
                    groups.setdefault(episode.get("session", episode["id"]), []).append(episode)
            for episodes in groups.values():
                if len(episodes) < 2:
                    continue
                sources = sources_for([e["id"] for e in episodes])
                output = model("concept", {"episodes": episodes, "sources": sources})
                parents = output.get("episode_ids")
                if not isinstance(parents, list) or len(set(parents)) < 2 or not set(parents) <= {e["id"] for e in episodes}:
                    raise ValueError("A concept must cite at least two supplied granular episodes")
                fields = {k:v for k,v in output.items() if k != "episode_ids"}
                validate_evidence(fields, sources_for(parents))
                add("concept", 2, {**fields, "episode_ids": parents}, parents)
        elif action.action == "edit":
            if len(action.targets) != 1 or action.targets[0] not in nodes:
                raise ValueError("Select exactly one concept proposal to edit")
            old = nodes[action.targets[0]]
            if old["type"] not in {"concept", "meta_concept"} or old["status"] not in {"candidate", "needs_review"}:
                raise ValueError("Only unaccepted concept proposals can be edited")
            changes = p.get("changes", {})
            if not isinstance(changes, dict) or not changes or set(changes) - {"title", "proposition", "conditions", "exceptions"}:
                raise ValueError("Edit title, proposition, conditions or exceptions")
            sources = sources_for([old["id"]])
            output = {key: old[key] for key in ("title", "proposition", "conditions", "exceptions", "evidence_ids", "evidence_quotes")}
            output.update(changes)
            validate_evidence(output, sources)
            report = None
            if old["type"] == "meta_concept":
                report = validate_report(model("validation", {"candidate": output, "sources": sources, "existing_concepts": graph["nodes"]}))
                report["passed"] = report["passed"] and report["novelty"] and not report["unsupported"] and not report["contradictions"]
            fields = {**output, **({"episode_ids": old["episode_ids"]} if "episode_ids" in old else {}), "supersedes_id": old["id"], "status": "needs_review" if report is None or report["passed"] else "rejected", "validation_report": report}
            new = add(old["type"], old["level"], fields, [old["id"]])
            update(old, status="superseded", superseded_by_id=new["id"])
            for edge in graph["hyperedges"]:
                if old["id"] in edge.get("crystallized_concept_ids", []):
                    update(edge, crystallized_concept_ids=[new["id"]], status="accepted" if new["status"] == "needs_review" else "rejected")
        elif action.action == "validate":
            for node in selected({"episode", "concept", "meta_concept"}, {"candidate", "needs_review"}):
                sources = sources_for([node["id"]])
                report = node.get("validation_report") if node["type"] == "meta_concept" else None
                if not report:
                    report = validate_report(model("validation", {"candidate": node, "sources": sources,
                        "supporting_episodes": [nodes[i] for i in node.get("episode_ids", []) if i in nodes],
                        "existing_concepts": [n for n in graph["nodes"] if n["type"] in {"concept", "meta_concept"} and n["id"] != node["id"]]}))
                passed = report["passed"] and not report["unsupported"] and not report["contradictions"] and (node["type"] == "episode" or report["novelty"])
                if any(s["status"] == "disputed" for s in sources):
                    passed = False
                update(node, status="validated" if passed else "needs_review", validation_state="machine_checked" if passed else "machine_check_failed", validation_report=report)
                for edge in graph["hyperedges"]:
                    if node["id"] in edge.get("crystallized_concept_ids", []) and passed:
                        update(edge, status="crystallized")
        elif action.action == "review":
            decision = p.get("decision")
            if decision not in {"accept", "reject"} or not isinstance(p.get("reason"), str) or not p["reason"].strip() or not action.targets:
                raise ValueError("Select candidates and supply a review decision and reason")
            for node in selected({"episode", "concept", "meta_concept"}, {"candidate", "needs_review"}):
                if decision == "accept":
                    if any(nodes[s]["status"] == "disputed" for s in source_ids(graph, node["id"])):
                        raise ValueError("Disputed evidence prevents acceptance")
                    if node["type"] == "meta_concept" and not node.get("validation_report", {}).get("passed"):
                        raise ValueError("Independent validation must pass before accepting an abstraction")
                update(node, status="validated" if decision == "accept" else "rejected", validation_state="human_reviewed", review={"decision": decision, "reason": p["reason"], "at": now()})
                for edge in graph["hyperedges"]:
                    if node["id"] in edge.get("crystallized_concept_ids", []):
                        update(edge, status="crystallized" if decision == "accept" else "rejected")
        elif action.action == "embed":
            eligible = [n for n in selected() if "vector" not in n and n["status"] in {"recorded", "validated"}]
            if eligible:
                embedding_started = time.perf_counter()
                texts = [canonical(n) for n in eligible]
                vectors, model_id = self.models.embed(texts, settings)
                metadata = self.models.embedding_metadata(settings)
                self.store.receipt(run["id"], job_id, {"operation": "embedding", "model": model_id, "inputs": texts, "dimensions": len(vectors[0]) if vectors else 0, "count": len(vectors), "encoder_metadata": metadata, "wall_seconds": time.perf_counter() - embedding_started, "cost": None, "cost_unit": "local-compute-unmeasured"})
                if len(vectors) != len(eligible):
                    raise ValueError("Embedding count differs from node count")
                vectors = normalized(vectors).tolist()
                existing = {n["embedding_model_id"] for n in graph["nodes"] if "vector" in n}
                if existing and existing != {model_id}:
                    raise ValueError("Incompatible embedding model revision")
                if existing and len(next(n["vector"] for n in graph["nodes"] if "vector" in n)) != len(vectors[0]):
                    raise ValueError("Incompatible embedding dimensions")
                for node, vector in zip(eligible, vectors):
                    update(node, vector=vector, embedding_ref=identity("emb"), embedding_model_id=model_id, embedding_input=canonical(node), encoder_metadata=metadata)
        elif action.action == "project":
            embedded = [n for n in graph["nodes"] if "vector" in n]
            if not embedded:
                raise ValueError("Embed nodes before projecting")
            fitted = projection([n["id"] for n in embedded], [n["vector"] for n in embedded])
            fitted.update(id=identity("projection"), version=run["version"] + 1, created_by_event=event_id)
            graph["projection"] = fitted
        elif action.action == "associate":
            overrides = p.get("settings", {})
            allowed = {"top_k", "threshold", "min_pairwise", "min_size", "max_size", "jaccard"}
            if not isinstance(overrides, dict) or set(overrides) - allowed:
                raise ValueError("Only discovery thresholds and cardinality can change in a discovery action")
            settings = Settings(**{**settings.model_dump(), **overrides})
            pool = [n for n in selected({"concept", "meta_concept"}, {"validated"}) if "vector" in n and ("levels" not in p or n["level"] in p["levels"])]
            if len(pool) < 2:
                raise ValueError("Discovery needs at least two accepted embedded concepts")
            proposals = discover([n["id"] for n in pool], [n["vector"] for n in pool], settings, {n["id"]: source_ids(graph, n["id"]) for n in pool}) if pool else []
            for proposal in proposals:
                hid = identity("hyper")
                graph["hyperedges"].append({"id": hid, "run_id": run["id"], "type": "hyperedge", "version": 1, "source_version": run["version"], "created_at": now(), "source_event_id": event_id, "created_by_event": event_id, "status": "candidate", "validation_state": "association_only", "level": max(nodes[i]["level"] for i in proposal["member_ids"]), "formation_method": "knn_cohesive_neighborhood_v1", "explanation": "Original-space cosine association; not a supported inference", "settings": settings.model_dump(), "crystallized_concept_ids": [], **proposal})
                created.append(hid)
        elif action.action == "crystallize":
            candidates = [edges[t] for t in action.targets if t in edges and edges[t]["status"] in {"candidate", "accepted"}]
            if len(candidates) != 1:
                raise ValueError("Select exactly one candidate hyperedge")
            edge = candidates[0]
            members = [nodes[i] for i in edge["member_ids"]]
            if any(n["status"] != "validated" for n in members):
                raise ValueError("All members must be validated")
            level = max(n["level"] for n in members) + 1
            if level > settings.max_depth:
                raise ValueError("Maximum recursive depth reached")
            sources = sources_for(edge["member_ids"])
            if len(sources) < 2:
                update(edge, status="rejected", rejection_reason="insufficient evidence: at least two original sources required")
            else:
                data = {"members": members, "sources": sources, "existing_concepts": [n for n in graph["nodes"] if n["type"] in {"concept", "meta_concept"}], "contradictions": [n for n in graph["nodes"] if n["status"] == "disputed"]}
                output = model("abstraction", data)
                if output.get("no_valid_abstraction") is True:
                    if set(output) != {"no_valid_abstraction", "reason"} or not isinstance(output["reason"], str):
                        raise ValueError("Invalid NO_VALID_ABSTRACTION response")
                    update(edge, status="rejected", rejection_reason=output["reason"])
                else:
                    validate_evidence(output, sources)
                    if any(output["proposition"].strip().lower() == n.get("proposition", "").strip().lower() for n in data["existing_concepts"]):
                        update(edge, status="rejected", rejection_reason="low novelty: exact duplicate proposition")
                    elif any(not set(output["evidence_ids"]) & source_ids(graph, m["id"]) for m in members):
                        update(edge, status="rejected", rejection_reason="insufficient evidence: not all members supported")
                    else:
                        report = validate_report(model("validation", {**data, "candidate": output}))
                        passed = report["passed"] and report["novelty"] and not report["unsupported"] and not report["contradictions"]
                        report["passed"] = passed
                        if not passed:
                            update(edge, status="rejected", rejection_reason=report["reason"], validation_report=report, rejected_proposal=output)
                        else:
                            meta = add("meta_concept", level, {**output, "status": "needs_review", "validation_report": report, "hyperedge_id": edge["id"]}, [*edge["member_ids"], *output["evidence_ids"]])
                            update(edge, status="accepted", crystallized_concept_ids=[meta["id"]])
        elif action.action == "challenge":
            if len(action.targets) != 1 or action.targets[0] not in nodes or nodes[action.targets[0]]["type"] != "trace" or not isinstance(p.get("reason"), str) or not p["reason"].strip():
                raise ValueError("Select one source trace and supply a dispute reason")
            target = nodes[action.targets[0]]
            affected = impact(graph, target["id"])
            update(target, status="disputed", dispute={"reason": p["reason"], "event_id": event_id})
            for nid in affected["node_ids"]:
                if nid != target["id"] and nodes[nid]["status"] != "rejected":
                    update(nodes[nid], status="needs_revalidation", validation_state="source_disputed")
            for hid in affected["hyperedge_ids"]:
                update(edges[hid], status="stale")
        elif action.action == "revalidate":
            for node in selected({"concept", "meta_concept", "episode"}, {"needs_revalidation"}):
                sources = sources_for([node["id"]])
                report = validate_report(model("validation", {"candidate": node, "sources": sources, "existing_concepts": graph["nodes"]}))
                disputed = any(s["status"] == "disputed" for s in sources)
                if disputed:
                    report["passed"] = False
                    report["reason"] = "Supporting trace is still disputed; retain challenge and import corrective evidence as a new source."
                update(node, status="needs_review" if report["passed"] else "needs_revalidation", validation_report=report)
        if not created and not updated and action.action not in {"project", "associate"}:
            raise ValueError("No eligible objects for this operation. Check selection, approval and preceding stages.")
        return {"created_ids": created, "updated_ids": sorted(set(updated)), "settings": settings.model_dump() if action.action == "associate" else None}

    def retrieve(self, run_id, query, variants, top_k):
        run = self.store.run(run_id)
        settings = Settings(**run["settings"])
        started = time.perf_counter()
        vectors, model_id = self.models.embed([query], settings)
        q = normalized(vectors)[0]
        graph = run["graph"]
        results = []
        types = {"A": {"trace"}, "B": {"episode"}, "C": {"concept"}, "D": {"concept"}, "E": {"concept", "meta_concept"}}
        for variant in variants:
            scored = []
            for node in graph["nodes"]:
                if node["type"] not in types[variant] or node["status"] not in {"validated", "recorded"} or "vector" not in node:
                    continue
                if node["embedding_model_id"] != model_id:
                    raise ValueError("Query encoder does not match stored vectors")
                scored.append({"id": node["id"], "score": float(np.dot(q, node["vector"])), "level": node["level"], "text": canonical(node), "source_ids": sorted(source_ids(graph, node["id"]))})
            hits = sorted(scored, key=lambda h: (-h["score"], h["id"]))[:top_k]
            associations = [h for h in graph["hyperedges"] if h["status"] not in {"stale", "rejected"} and set(h["member_ids"]) & {hit["id"] for hit in hits}] if variant in {"D", "E"} else []
            by_id = {n["id"]: n for n in graph["nodes"]}
            edge_hits = []
            if variant in {"D", "E"}:
                for edge in graph["hyperedges"]:
                    members = [by_id[i] for i in edge["member_ids"]]
                    if edge["status"] in {"stale", "rejected"} or any(n["status"] != "validated" or "vector" not in n for n in members):
                        continue
                    center = np.mean([n["vector"] for n in members], axis=0)
                    if np.linalg.norm(center) == 0:
                        continue
                    edge_hits.append({"id": edge["id"], "score": float(np.dot(q, center / np.linalg.norm(center))), "member_ids": edge["member_ids"], "method": "normalized_member_centroid", "status": "association_only"})
                edge_hits = sorted(edge_hits, key=lambda h: (-h["score"], h["id"]))[:top_k]
            expanded_ids = sorted(set().union(*(set(h["member_ids"]) for h in [*associations, *edge_hits]))) if associations or edge_hits else []
            expanded_evidence = [{"id": sid, "content": by_id[sid]["content"]} for sid in sorted(set().union(*(source_ids(graph, nid) for nid in {*expanded_ids, *(h["id"] for h in hits)})))]
            results.append({"variant": variant, "hits": hits, "edge_hits": edge_hits, "hyperedge_ids": [h["id"] for h in associations], "expanded_member_ids": expanded_ids, "supporting_evidence": expanded_evidence, "provenance_coverage": sum(bool(h["source_ids"]) for h in hits) / len(hits) if hits else None, "groundedness_review": "unrated", "answer_correctness": None, "answer": "\n".join(f'[{h["id"]}] {h["text"]}' for h in hits) or "No eligible embedded evidence.", "retrieval_tokens": None, "retrieval_characters": sum(len(h["text"]) for h in hits) + sum(len(s["content"]) for s in expanded_evidence), "generation_tokens": 0, "cost": 0, "cost_note": "Extractive retrieval; local compute cost unmeasured", "wall_seconds": time.perf_counter() - started})
        corpus_hash = hashlib.sha256(encoded(sorted(n["content_hash"] for n in graph["nodes"] if n["type"] == "trace")).encode()).hexdigest()
        result = {"id": identity("evaluation"), "run_id": run_id, "graph_version": run["version"], "source_corpus_sha256": corpus_hash, "query": query, "query_set": "frozen_held_out" if query in settings.held_out_queries else "ad_hoc", "embedding_model_id": model_id, "results": results, "created_at": now(), "settings": run["settings"]}
        with self.store.connection() as db:
            db.execute("INSERT INTO evaluations VALUES(?,?,?)", (result["id"], run_id, encoded(result)))
            self.store.event(db, run_id, "retrieval.completed", result)
        return result


def health(graph):
    nodes, edges = graph["nodes"], graph["hyperedges"]
    concepts = [n for n in nodes if n["type"] in {"concept", "meta_concept"}]
    counts = Counter(n.get("proposition", "").strip().lower() for n in concepts)
    memberships = Counter(i for h in edges for i in h["member_ids"])
    return {"count_by_depth": dict(Counter(n["level"] for n in nodes)), "max_depth": max((n["level"] for n in nodes), default=0), "hyperedge_count": len(edges), "average_cardinality": sum(len(h["member_ids"]) for h in edges) / len(edges) if edges else 0, "max_cardinality": max((len(h["member_ids"]) for h in edges), default=0), "overlapping_nodes": sum(n > 1 for n in memberships.values()), "duplicate_concept_rate": sum(n - 1 for n in counts.values()) / len(concepts) if concepts else 0, "provenance_coverage": sum(bool(source_ids(graph, n["id"])) for n in concepts) / len(concepts) if concepts else None, "human_reviewed_concepts": sum(n.get("validation_state") == "human_reviewed" for n in concepts), "disputed_sources": sum(n["status"] == "disputed" for n in nodes), "needs_revalidation": sum(n["status"] == "needs_revalidation" for n in nodes), "rejected_crystallizations": sum(h["status"] == "rejected" for h in edges), "orphaned_concepts": [n["id"] for n in concepts if not source_ids(graph, n["id"])]}
