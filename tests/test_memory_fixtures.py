"""Twenty deterministic causal fixtures from pipeline section 5.2.

Expected IDs and JSON subsets are hand-specified, independent of backend output.
No model calls, reference plans, or benchmark gold enter any operation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from constraint_memory.memory import create_backend


ROOT = Path(__file__).resolve().parents[1]


def event(eid, text, step=0, group="g1"):
    return {"op": "observe", "value": {"event_id": eid, "step": step, "type": "user_request", "content": text, "group_id": group}}


def constraint(cid, traveler="Bob", slot="dinner", day=2, value=None, ref=None, operator="EQUALS", attribute="value", source="request", **extra):
    target = {"ref": ref} if ref else {"value": value}
    return {"constraint_id": cid, "kind": "JOIN" if ref else "LOCAL", "scope": {"group": "g1", "traveler": traveler},
            "subject": {"traveler": traveler, "day": day, "slot": slot, "attribute": attribute}, "operator": operator,
            "target": target, "source_refs": [f"event:{source}"], **extra}


def decision(traveler="Alice", value="Blue Bistro", day=2, slot="dinner", attribute="value", **extra):
    return {"traveler": traveler, "day": day, "slot": slot, "attribute": attribute, "value": value, **extra}


def begin(traveler, query, step, extraction=None, **extra):
    value = {"group_id": "g1", "traveler": traveler, "query": query, "step": step, "round_idx": step, **extra}
    if extraction:
        value["extraction"] = extraction
    return {"op": "begin", "value": value}


def commit(traveler, action, step, decisions=None, extraction=None, **extra):
    value = {"group_id": "g1", "traveler": traveler, "action": action, "step": step, **extra}
    if decisions is not None:
        value["decisions"] = decisions
    if extraction is not None:
        value["extraction"] = extraction
    return {"op": "commit", "value": value}


def definition(term, text, related=None, aliases=None, source="request"):
    return {"term_id": f"definition:{term}", "term": term, "text": text, "related_terms": related or [],
            "entity_aliases": aliases or [], "parent_scope": {"group": "g1"}, "evidence_refs": [f"event:{source}"]}


def expected(status="OK", closure=None, errors=None, records=None, glossary=None, **extra):
    return {"status": status, "closure_ids": closure or [], "error_codes": errors or [],
            "record_json": records or [], "glossary_ids": glossary or [], **extra}


def build_fixture_data():
    fixtures = []

    def add(name, ops, result, variant="E4", config=None):
        fixtures.append({"name": name, "variant": variant, "group_id": "g1", "config": config or {}, "operations": ops, "expected": result})

    alice = [begin("Alice", "Plan Alice", 0), commit("Alice", "day 2 dinner Blue Bistro", 0, [decision()])]
    ref = "decision:Alice:day2:dinner"
    bob = constraint("c_join", ref=ref)

    local = constraint("c_local", traveler="Alice", value="vegan")
    add("01_single_local_restriction_no_history", [event("request", "Alice must have vegan dinner on day 2."), begin("Alice", "vegan dinner", 0, {"constraints": [local]})],
        expected(closure=["c_local@v1"], records=[{"constraint_id": "c_local", "kind": "LOCAL", "operator": "EQUALS", "target": {"value": "vegan"}, "status": "ACTIVE", "source_refs": ["event:request"]}]))

    add("02_join_same_dinner_same_day", alice + [event("request", "Bob day 2 dinner equals Alice day 2 dinner.", 1), begin("Bob", "same dinner as Alice", 1, {"constraints": [bob]})],
        expected(closure=[ref + "@v1", "c_join@v1"], records=[{"decision_id": ref, "value": "Blue Bistro", "status": "COMMITTED", "version": 1}]))

    ccarol = constraint("c_carol", traveler="Carol", ref="decision:Bob:day2:dinner", source="carol_request")
    add("03_join_chain_alice_bob_carol", alice + [event("request", "Bob dinner equals Alice", 1), begin("Bob", "join Alice dinner", 1, {"constraints": [bob]}),
        commit("Bob", "Blue Bistro", 1, [decision("Bob")]), event("carol_request", "Carol dinner equals Bob", 2), begin("Carol", "join Bob dinner", 2, {"constraints": [ccarol]})],
        expected(closure=[ref + "@v1", "c_join@v1", "decision:Bob:day2:dinner@v1", "c_carol@v1"]))

    hotelref = "decision:Alice:day1:hotel:rating"
    rating = constraint("c_rating", day=1, slot="hotel", attribute="rating", ref=hotelref, source="request")
    rating["target"]["offset"] = 1
    rating["kind"] = "RELATION"
    add("04_relation_hotel_rating_offset", [begin("Alice", "hotel", 0), commit("Alice", {"hotel_rating": 4}, 0, [decision(value=4, day=1, slot="hotel", attribute="rating")]),
        event("request", "Bob hotel rating is 1 greater than Alice hotel rating.", 1), begin("Bob", "hotel rating Alice plus 1", 1, {"constraints": [rating]})],
        expected(closure=[hotelref + "@v1", "c_rating@v1"], records=[{"constraint_id": "c_rating", "kind": "RELATION", "target": {"ref": hotelref, "offset": 1}}]))

    priceref = "decision:Alice:day2:dinner:price"
    price = constraint("c_price", ref=priceref, operator="LTE", attribute="price", unit="USD")
    price["target"].update({"offset": -20, "unit": "USD"})
    price["kind"] = "RELATION"
    price_alice = [begin("Alice", "dinner", 0), commit("Alice", {"dinner_price": 80, "unit": "USD"}, 0, [decision(value=80, attribute="price", unit="USD")])]
    add("05_relation_units_and_inequality", price_alice + [event("request", "Bob dinner price USD must be at least 20 less than Alice dinner price USD", 1),
        begin("Bob", "Alice dinner price less 20 USD", 1, {"constraints": [price]})], expected(closure=[priceref + "@v1", "c_price@v1"],
        records=[{"constraint_id": "c_price", "operator": "LTE", "unit": "USD", "target": {"ref": priceref, "offset": -20, "unit": "USD"}}]))

    add("06_mixed_join_relation_sharing_slot", [begin("Alice", "dinner", 0), commit("Alice", {"restaurant": "Blue Bistro", "price": 80, "unit": "USD"}, 0,
        [decision(), decision(value=80, attribute="price", unit="USD")]), event("request", "Bob dinner equals Alice dinner and Bob USD price is 20 less than Alice", 1),
        begin("Bob", "join Alice dinner", 1, {"constraints": [bob, price]})], expected(closure=[ref + "@v1", "c_join@v1", priceref + "@v1", "c_price@v1"]))

    rating2 = constraint("c_entity_rating", ref="decision:Alice:day2:dinner:rating", attribute="rating", operator="GTE")
    add("07_multiple_attributes_same_entity", [begin("Alice", "dinner", 0), commit("Alice", {"restaurant": "Blue Bistro", "price": 80, "rating": 4.5}, 0,
        [decision(), decision(value=80, attribute="price"), decision(value=4.5, attribute="rating")]), event("request", "Bob needs Alice dinner rating and Alice dinner price", 1),
        begin("Bob", "Alice dinner attributes", 1, {"constraints": [price, rating2]})],
        expected(closure=[priceref + "@v1", "c_price@v1", "decision:Alice:day2:dinner:rating@v1", "c_entity_rating@v1", ref + "@v1"],
                 records=[{"decision_id": priceref, "attribute": "price", "value": 80}, {"decision_id": "decision:Alice:day2:dinner:rating", "attribute": "rating", "value": 4.5}]))

    old = constraint("c_old", traveler="Alice", value="vegan")
    new = constraint("c_new", traveler="Alice", value="seafood", source="update", supersedes="c_old")
    add("08_explicit_preference_supersession", [event("request", "Alice dinner vegan", 0), begin("Alice", "vegan", 0, {"constraints": [old]}),
        event("update", "Update Alice dinner to seafood instead of vegan", 1), begin("Alice", "seafood", 1, {"constraints": [new]})],
        expected(closure=["c_new@v1"], records=[{"constraint_id": "c_new", "target": {"value": "seafood"}, "supersedes": "c_old"}],
                 snapshot_records=[{"constraint_id": "c_old", "status": "SUPERSEDED", "version": 2}]))

    add("09_actual_committed_choice_preserved", alice + [event("request", "Bob dinner same as Alice", 1), begin("Bob", "Alice dinner", 1, {"constraints": [bob]})],
        expected(closure=[ref + "@v1", "c_join@v1"], records=[{"decision_id": ref, "value": "Blue Bistro"}], context_contains=["Blue Bistro"], context_excludes=["Red Restaurant"]))

    add("10_failed_branch_excluded", [begin("Alice", "dinner", 0), commit("Alice", "Rejected Bistro", 0, [decision(value="Rejected Bistro")], branch_status="FAILED", status="FAILED"),
        event("request", "Bob dinner equals Alice dinner", 1), begin("Bob", "Alice dinner", 1, {"constraints": [bob]})],
        expected(status="UNRESOLVED", closure=["c_join@v1"], errors=["INVALID_REFERENCE"], context_excludes=["Rejected Bistro"]))

    aliases = [definition("Alex North", "Alex North is one person", aliases=["Alex"]), definition("Alex South", "Alex South is another person", aliases=["Alex"])]
    add("11_ambiguous_entity_alias", [event("request", "Alex North and Alex South are different people"), begin("Bob", "Who is Alex?", 0, {"definitions": aliases})],
        expected(status="UNRESOLVED", errors=["AMBIGUOUS_ALIAS"]), variant="E6")

    chain = [definition("Frodo", "Frodo follows Bilbo", ["Bilbo"]), definition("Bilbo", "Bilbo lives in Shire", ["Shire"]), definition("Shire", "Shire is a named location")]
    add("12_two_hop_glossary_chain", [event("request", "Frodo follows Bilbo; Bilbo lives in Shire; Shire is a named location"), begin("Bob", "Where does Frodo live?", 0, {"definitions": chain})],
        expected(glossary=["definition:Frodo", "definition:Bilbo", "definition:Shire"]), variant="E6")

    cycle = [definition("Frodo", "Frodo refers to Bilbo", ["Bilbo"]), definition("Bilbo", "Bilbo refers to Frodo", ["Frodo"])]
    add("13_cyclic_glossary_aliases", [event("request", "Frodo refers to Bilbo; Bilbo refers to Frodo"), begin("Bob", "Frodo", 0, {"definitions": cycle})],
        expected(status="UNRESOLVED", errors=["GLOSSARY_CYCLE"], glossary=["definition:Frodo", "definition:Bilbo"]), variant="E6")

    contradictory = [constraint("c_x", value="vegan"), constraint("c_y", value="seafood")]
    add("14_contradictory_constraints", [event("request", "Bob dinner must equal vegan and must equal seafood"), begin("Bob", "dinner", 0, {"constraints": contradictory})],
        expected(status="VALIDATION_FAILED", closure=["c_x@v1", "c_y@v1"], errors=["CONTRADICTORY_CONSTRAINTS"]))

    add("15_missing_traveler_slot_reference", [event("request", "Bob dinner equals Alice dinner"), begin("Bob", "join Alice dinner", 0, {"constraints": [bob]})],
        expected(status="UNRESOLVED", closure=["c_join@v1"], errors=["MISSING_REFERENCE", "UNRESOLVED_REFERENCE"], missing_refs=[ref]))

    large = [constraint(f"c_large_{i}", value="venue " + str(i) + " " + "X" * 800) for i in range(3)]
    add("16_out_of_budget_dependency_closure", [event("request", "large explicit requirements"), begin("Bob", "dinner", 0, {"constraints": large})],
        expected(status="BUDGET_OVERFLOW", errors=["BUDGET_OVERFLOW"], required_closure_ids=[f"c_large_{i}@v1" for i in range(3)], batch_min_count=3), variant="E3", config={"token_budget": 220})

    exact = [constraint("c_exact_price", value=49.95, attribute="price", operator="LTE", unit="USD", narrative="repeated explanation " * 400),
             constraint("c_exact_rating", value=4.75, attribute="rating", operator="GTE", narrative="more narrative " * 400)]
    add("17_compression_preserves_exact_comparators", [event("request", "Bob price <= 49.95 USD and rating >= 4.75"), begin("Bob", "dinner", 0, {"constraints": exact})],
        expected(closure=["c_exact_price@v1", "c_exact_rating@v1"], records=[{"constraint_id": "c_exact_price", "operator": "LTE", "target": {"value": 49.95}, "unit": "USD"},
            {"constraint_id": "c_exact_rating", "operator": "GTE", "target": {"value": 4.75}}], min_compression_events=1), variant="E3", config={"token_budget": 600})

    group2 = {"op": "begin", "value": {"group_id": "g2", "traveler": "Bob", "query": "Alice dinner", "step": 0, "dependency_refs": [ref]}}
    add("18_independent_groups_identical_names", alice + [{"op": "reset", "value": "g2"}, group2],
        expected(status="UNRESOLVED", errors=["MISSING_REFERENCE"], context_excludes=["Blue Bistro"], snapshot_group="g2"))

    false = constraint("c_false", traveler="Alice", value=400, attribute="price", operator="LTE")
    add("19_raw_evidence_contradicts_summary_literal", [event("request", "Alice dinner price must be no more than 40"), begin("Alice", "dinner", 0, {"constraints": [false]})],
        expected(status="VALIDATION_FAILED", errors=["EVIDENCE_MISMATCH"], absent_ids=["c_false"]))

    ops = [begin("Alice", "dinner", 0), commit("Alice", "Blue Bistro", 0, [decision()])]
    for index, (traveler, prior) in enumerate((("Bob", "Alice"), ("Carol", "Bob"), ("Dave", "Carol")), start=1):
        cid = f"c_{traveler.lower()}"
        eref = f"r_{traveler.lower()}"
        con = constraint(cid, traveler=traveler, ref=f"decision:{prior}:day2:dinner", source=eref)
        ops += [event(eref, f"{traveler} dinner equals {prior} dinner", index), begin(traveler, f"join {prior}", index, {"constraints": [con]}),
                commit(traveler, "Blue Bistro", index, [decision(traveler)])]
    ops += [begin("Zoe", "independent dinner", 4), commit("Zoe", "Independent Inn", 4, [decision("Zoe", "Independent Inn")]),
            commit("Alice", "new observation invalidates Alice", 5, [], invalidate_decision_refs=[ref], recovery_trigger="explicit_conflict", remaining_action_budget=2),
            begin("Eve", "Dave dinner", 6, dependency_refs=["decision:Dave:day2:dinner"])]
    add("20_targeted_invalidation_depth_four_chain", ops,
        expected(status="UNRESOLVED", errors=["INVALID_REFERENCE"],
                 invalidated_ids=[f"decision:{name}:day2:dinner" for name in ("Alice", "Bob", "Carol", "Dave")],
                 valid_ids=["decision:Zoe:day2:dinner"], recovery_seed=[ref], recovery_earliest_step=0, immutable_value={"decision_id": ref, "version": 1, "status": "COMMITTED", "value": "Blue Bistro"}), variant="E5")
    return {"schema_version": 1, "description": "Section 5.2 causal fixture suite. Fixed parser outputs, actual actions only; no external model or benchmark gold.", "fixtures": fixtures}


def subset_match(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and subset_match(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and actual == expected
    return actual == expected


class MemoryFixtureTests(unittest.TestCase):
    def test_twenty_causal_fixtures(self):
        data = json.loads((ROOT / "data/fixtures/memory_cases.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["fixtures"]), 20)
        for case in data["fixtures"]:
            with self.subTest(fixture=case["name"]):
                memory = create_backend(case["variant"], case["config"])
                memory.reset(case["group_id"])
                response = None
                commit_responses = []
                for operation in case["operations"]:
                    value = copy.deepcopy(operation["value"])
                    if operation["op"] == "observe": memory.observe(value)
                    elif operation["op"] == "begin": response = memory.begin_subtask(value)
                    elif operation["op"] == "commit": commit_responses.append(memory.commit_subtask(value))
                    elif operation["op"] == "reset": memory.reset(value)
                    else: self.fail(f"Unknown fixture operation {operation['op']}")
                wanted = case["expected"]
                self.assertIsNotNone(response)
                self.assertEqual(response["status"], wanted["status"], response)
                self.assertEqual(set(response["closure_ids"]), set(wanted["closure_ids"]), response)
                errors = {item["code"] for item in response["diagnostics"]}
                self.assertTrue(set(wanted["error_codes"]).issubset(errors), response)
                self.assertEqual(set(response.get("glossary_ids", [])), set(wanted["glossary_ids"]), response)
                snapshot = memory.snapshot()
                all_records = snapshot.get("constraints", []) + snapshot.get("decisions", []) + snapshot.get("definitions", [])
                for record in wanted["record_json"]:
                    self.assertTrue(any(subset_match(r, record) for r in response.get("records", [])), record)
                for record in wanted.get("snapshot_records", []):
                    self.assertTrue(any(subset_match(r, record) for r in all_records), record)
                for text in wanted.get("context_contains", []): self.assertIn(text, response["context"])
                for text in wanted.get("context_excludes", []): self.assertNotIn(text, response["context"])
                for ref in wanted.get("absent_ids", []): self.assertFalse(any(ref in r.values() for r in all_records))
                if "missing_refs" in wanted: self.assertEqual(response["missing_refs"], wanted["missing_refs"])
                if "required_closure_ids" in wanted: self.assertEqual(set(response["required_closure_ids"]), set(wanted["required_closure_ids"]))
                if "batch_min_count" in wanted: self.assertGreaterEqual(len(response["batches"]), wanted["batch_min_count"])
                if "min_compression_events" in wanted: self.assertGreaterEqual(memory.stats().get("compression_events", 0), wanted["min_compression_events"])
                if "snapshot_group" in wanted: self.assertEqual(snapshot["group_id"], wanted["snapshot_group"])
                if "invalidated_ids" in wanted:
                    recovery = snapshot["recovery_log"][-1]
                    self.assertEqual(set(recovery["affected_decisions"]), set(wanted["invalidated_ids"]))
                    self.assertEqual(recovery["seed_decisions"], wanted["recovery_seed"])
                    self.assertEqual(recovery["earliest_affected_step"], wanted["recovery_earliest_step"])
                    self.assertFalse(recovery["replanning_performed"])
                    self.assertTrue(any(subset_match(r, wanted["immutable_value"]) for r in all_records))
                    for ref in wanted["valid_ids"]: self.assertIsNotNone(memory._resolve(ref, 6, diagnostics=False))


if __name__ == "__main__":
    unittest.main()
