"""Inspectable entity/state traversal and exact, budgeted context serialization."""
import datetime as dt
import hashlib
import json
import re
import time
from collections import defaultdict, deque

from .grounded import ELIGIBLE, config_id
from .store import encoded, identity, now

TRAVERSE = {"PARTICIPATES_IN", "HAS_FACT", "HAS_STATE", "DEPENDS_ON", "APPLIES_TO", "IDENTICAL_TO"}


def words(text):
    return set(re.findall(r"\w+", text.casefold()))


def current(obj, query):
    if obj.get("status") not in ELIGIBLE:
        return False
    if query.scope and obj.get("scope") != query.scope:
        return False
    if obj.get("branch_id") and obj["branch_id"] != query.branch_id:
        return False
    if query.task_id and obj.get("type") == "state" and obj.get("task_id") != query.task_id:
        return False
    def parse(v):
        value = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
        return value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None else value
    as_of = parse(query.as_of or now())
    return not (obj.get("valid_from") and parse(obj["valid_from"]) > as_of or obj.get("valid_to") and parse(obj["valid_to"]) <= as_of)


def render_object(n):
    if n["type"] == "fact":
        value = n.get("value")
        if n.get("target_id"):
            value = n["target_id"]
        text = "unknown" if value is None else json.dumps(value, ensure_ascii=False)
        return f"{n['subject_id']}: {n['predicate']} = {text}" + (" " + n["units"] if n.get("units") else "") + f"; modality={n['modality']}; scope={n['scope']}"
    if n["type"] == "state":
        return f"{n['state_type']} ({n.get('execution_status', 'unknown')}): {n['content']}; task={n['task_id']}; branch={n['branch_id']}"
    if n["type"] == "rule":
        return f"{n['instruction']} [experimental {n.get('hypothesis_status', 'hypothesis')}]"
    return n.get("summary", n.get("content", n["title"]))


def assemble(run, query, tokenizer, tokenizer_identity, stop_after=None):
    started = time.perf_counter()
    graph = run["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    active = {i: n for i, n in nodes.items() if current(n, query)}
    query_words = words(query.query)
    resolutions = []
    for n in active.values():
        if n["type"] != "entity":
            continue
        names = [n.get("canonical_name", n["title"]), *n.get("aliases", [])]
        if any(words(name) and words(name) <= query_words for name in names):
            canonical = active.get(n.get("same_as"), n)
            resolutions.append({"id": canonical["id"], "name": canonical.get("canonical_name"), "scope": canonical.get("scope"), "resolution": canonical.get("resolution", "resolved"), "method": "typed scoped exact alias tokens + explicit identity decisions"})
    resolutions = list({r["id"]: r for r in resolutions}.values())
    by_name = defaultdict(list)
    for r in resolutions:
        by_name[(r["name"] or "").casefold()].append(r)
    ambiguous = [name for name, matches in by_name.items() if len(matches) > 1 or any(r["resolution"] != "resolved" for r in matches)]
    scopes = {r["scope"] for r in resolutions}
    clarification = bool(ambiguous or len(scopes) > 1)
    scope = query.scope or (next(iter(scopes)) if len(scopes) == 1 else None)
    fallback = None if resolutions else "Explicit lexical fallback; no sufficiently resolved entity"
    eligible = {i: n for i, n in active.items() if scope is not None and n.get("scope") == scope}
    if stop_after == "resolve":
        return {"stage": "resolve", "entity_resolutions": resolutions, "ambiguities": ambiguous,
                "scope": scope, "fallback": fallback, "clarification_required": clarification or scope is None}
    paths = {r["id"]: [] for r in resolutions if r["id"] in eligible}
    adjacency = defaultdict(list)
    for link in graph["links"]:
        if link["type"] not in TRAVERSE or link.get("status", "accepted") not in ELIGIBLE:
            continue
        if link["source"] not in eligible or link["target"] not in eligible:
            continue
        if link.get("scope") not in (None, scope):
            continue
        adjacency[link["source"]].append((link["target"], link))
        adjacency[link["target"]].append((link["source"], link))
    queue = deque(sorted(paths))
    while queue and len(paths) < query.max_nodes and not clarification:
        nid = queue.popleft()
        if len(paths[nid]) >= query.max_hops:
            continue
        for target, link in sorted(adjacency[nid], key=lambda pair: pair[0]):
            if target in paths or len(paths) >= query.max_nodes:
                continue
            paths[target] = paths[nid] + [{"relation_id": link["id"], "type": link["type"], "from": nid, "to": target}]
            queue.append(target)
    selected_entities = {i for i in paths if nodes[i]["type"] == "entity"}
    if stop_after == "traverse":
        return {"stage": "traverse", "scope": scope, "paths": paths, "selected_entities": sorted(selected_entities),
                "limits": {"max_hops": query.max_hops, "max_nodes": query.max_nodes, "visited": len(paths)}}
    # Retrieve all applicable facts/states attached to the resolved participants,
    # including their immediate facts after bounded participant discovery.
    candidates = []
    for n in eligible.values():
        if query.variant == "A":
            allowed = n["type"] in {"episode", "trace"}
        else:
            allowed = n["type"] in {"fact", "state"} and (n["id"] in paths or not resolutions)
        if not allowed or clarification:
            continue
        lexical = len(words(render_object(n)) & query_words) / max(1, len(query_words))
        entity_score = 1 if n["id"] in paths or n.get("subject_id") in selected_entities else 0
        mandatory = n.get("modality") == "required" or n["type"] == "state" and n.get("execution_status") in {"pending", "committed", "failure"}
        candidates.append({"id": n["id"], "type": n["type"], "text": render_object(n), "mandatory": mandatory,
                           "scores": {"entity": entity_score, "lexical": lexical, "hard_constraint": int(mandatory)},
                           "score": entity_score * 3 + lexical + int(mandatory) * 10, "path": paths.get(n["id"], []),
                           "reason": "required fact/current commitment coverage" if mandatory else "scoped entity/state traversal"})
    candidates.sort(key=lambda x: (-x["score"], x["id"]))
    missing, conflicts = [], []
    groups = defaultdict(list)
    for c in candidates:
        n = nodes[c["id"]]
        if n["type"] == "fact":
            groups[(n["subject_id"], n["predicate"], n["scope"], n.get("modality"))].append(n)
            if n.get("value_type") == "unknown":
                missing.append({"id": n["id"], "slot": n["predicate"], "reason": "literal value/units unknown; do not infer"})
    for key, facts in groups.items():
        if len({encoded([n.get("value"), n.get("target_id"), n.get("units")]) for n in facts}) > 1:
            conflicts.append({"key": list(key), "ids": sorted(n["id"] for n in facts), "reason": "Conflicting simultaneously valid values"})
    rules = []
    if query.variant in {"C", "D"} and not clarification:
        for n in eligible.values():
            if n["type"] != "rule":
                continue
            method = "direct" if query.variant == "C" else "hypergraph"
            reasons = []
            if n.get("origin_method") != method:
                reasons.append("wrong experimental variant")
            if n.get("hypothesis_status") != "promoted" and not query.allow_hypotheses:
                reasons.append("experimental hypothesis disabled by policy")
            if n.get("contradicting_ids"):
                reasons.append("declared contradictory evidence")
            if n.get("task_types") and not any(words(t) & query_words for t in n["task_types"]):
                reasons.append("task applicability unbound")
            if n.get("target_entity_types") and not any(nodes[i].get("entity_type") in n["target_entity_types"] for i in selected_entities):
                reasons.append("target entity type unbound")
            conditions = n.get("applicability", [])
            known_conditions = {"same_scope", "branch:" + query.branch_id, "task:" + (query.task_id or "")}
            if any(condition not in known_conditions for condition in conditions):
                reasons.append("natural-language applicability requires explicit review/binding")
            bindings = []
            for slot in n.get("required_inputs", []):
                available = [nodes[c["id"]] for c in candidates] + [nodes[i] for i in selected_entities]
                matches = [obj["id"] for obj in available if obj["type"] == slot["object_type"] and (not slot.get("predicate") or obj.get("predicate") == slot["predicate"]) and slot.get("scope") in {scope, "same_scope"}]
                if slot.get("entity_type"):
                    matches = [i for i in matches if (nodes[i] if nodes[i]["type"] == "entity" else nodes.get(nodes[i].get("subject_id"), {})).get("entity_type") == slot["entity_type"]]
                safe = [i for i in matches if nodes[i].get("value_type") != "unknown" and not any(i in conflict["ids"] for conflict in conflicts)]
                bindings.append({"slot": slot["name"], "ids": safe, "status": "bound" if safe else "missing_or_conflicting"})
            if any(b["status"] != "bound" for b in bindings):
                reasons.append("required inputs unbound")
            rules.append({"id": n["id"], "eligible": not reasons, "bindings": bindings, "excluded_reasons": reasons})
            if not reasons and len([c for c in candidates if c["type"] == "rule"]) < 3:
                candidates.append({"id": n["id"], "type": "rule", "text": render_object(n), "mandatory": False, "score": -1, "scores": {}, "path": [], "reason": "checked rule bindings"})
    if stop_after == "select":
        return {"stage": "select", "candidates": candidates, "rule_checks": rules, "missing_slots": missing, "conflicts": conflicts}
    warnings = []
    if fallback:
        warnings.append(fallback)
    if scope is None:
        warnings.append("No resolved task scope: choose a scope before traversing unrelated data")
    if clarification:
        warnings.append("Ambiguous entity/task: " + ", ".join(ambiguous or sorted(scopes)))
    warnings += [f"Unknown {x['slot']} [{x['id']}]" for x in missing]
    warnings += ["Conflicting values: " + ", ".join(c["ids"]) for c in conflicts]
    if not candidates:
        warnings.append("No eligible memory objects for this variant. Extract and review concrete facts/state; do not infer missing memory from a rule or title")
    system = "Use only cited, scoped facts and actual task state. Required and preferred constraints differ. Do not invent unknown values or claim an unobserved outcome. Rules are experimental guidance, never substitutes for facts."
    def messages(items):
        sections = [("TASK STATE", "state"), ("RELEVANT FACTS", "fact"), ("EPISODIC EVIDENCE", "episode"), ("ORIGINAL EVIDENCE", "trace"), ("CONDITIONAL GUIDANCE", "rule")]
        parts = ["REQUEST\n" + query.query]
        for title, kind in sections:
            content = [f"[{c['id']}] {c['text']}" for c in items if c["type"] == kind]
            if content:
                parts.append(title + "\n" + "\n".join(content))
        if warnings:
            parts.append("UNRESOLVED\n" + "\n".join(warnings))
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]
    def count(items):
        return len(tokenizer.encode(encoded(messages(items))).ids)
    included, excluded = [], []
    for c in candidates:
        if count(included + [c]) <= query.token_budget:
            included.append(c)
        else:
            excluded.append({"id": c["id"], "reason": "context budget", "mandatory": c["mandatory"]})
    omitted_required = [c for c in excluded if c["mandatory"]]
    status = "clarification_required" if clarification or scope is None else "insufficient_context_budget" if omitted_required or count([]) > query.token_budget else "no_relevant_memory" if not candidates else "success"
    actual = messages(included)
    raw = encoded(actual)
    selected_ids = {c["id"] for c in included}
    excluded += [{"id": n["id"], "reason": "status/scope/time/branch/variant or relevance filter"} for n in graph["nodes"] if n["id"] not in selected_ids and not any(e["id"] == n["id"] for e in excluded)]
    return {"id": identity("retrieval"), "created_at": now(), "run_id": run["id"], "graph_version": run["version"], "config_id": config_id(run),
            "query": query.model_dump(), "status": status, "dispatch_status": "preview_only", "entity_resolutions": resolutions, "ambiguities": ambiguous,
            "fallback": fallback, "scope": scope, "paths": paths, "traversal_limits": {"max_hops": query.max_hops, "max_nodes": query.max_nodes, "visited": len(paths)}, "candidates": candidates, "rule_checks": rules, "missing_slots": missing, "conflicts": conflicts,
            "included": included, "excluded": excluded, "omitted_requirements": omitted_required, "messages": actual, "serialized_messages": raw,
            "messages_sha256": hashlib.sha256(raw.encode()).hexdigest(), "token_count": count(included), "token_budget": query.token_budget,
            "tokenizer": tokenizer_identity, "token_count_basis": "exact UTF-8 serialized role/message JSON; external provider framing not included",
            "agent_configured": bool(run["settings"].get("agent_enabled")), "warnings": warnings, "timing": {"total_seconds": time.perf_counter() - started},
            "cost": {"generation_calls": 0, "embedding_calls": 0, "local_compute_cost": "unmeasured"}, "outcome_score": None}
