"""Independent sufficiency tests for E3F's planner-only materialized frontier.

All choices are synthetic actual commits. No models, hidden plans, or holdout
queries are read. Public dev wording below is fixed parser input, not an oracle.
"""
from pathlib import Path
import json
import unittest

from constraint_memory.memory import create_backend


def request(name, step, query="", extraction=None, **extra):
    return {"group_id": "g", "traveler": name, "step": step, "round_idx": step,
            "query": query, "extraction": extraction or {}, **extra}


def observe(memory, eid, content, step, event_type="request"):
    memory.observe({"event_id": eid, "group_id": "g", "step": step, "type": event_type, "content": content})


def choice(name, value="Actual Cafe, Boston", day=1, slot="dinner", **extra):
    return {"traveler": name, "day": day, "slot": slot, "value": value,
            "source_refs": ["event:action"], **extra}


def binding(cid, name, source, prior=None, value=None, operator="EQUALS", **extra):
    return {"constraint_id": cid, "kind": "JOIN" if prior else "LOCAL",
            "scope": {"group": "g", "traveler": name},
            "subject": {"traveler": name, "day": 1, "slot": "dinner", "attribute": "value"},
            "operator": operator, "target": {"ref": prior} if prior else {"value": value},
            "source_refs": ["event:" + source], **extra}


def seed_choice(memory, name="Alice", step=0, value="Actual Cafe, Boston", extra_choices=None):
    memory.begin_subtask(request(name, step))
    rows = [choice(name, value), *(extra_choices or [])]
    memory.commit_subtask({**request(name, step), "action": "; ".join(str(r["value"]) for r in rows), "decisions": rows})


def seed_chain(memory, count=4, local_count=0):
    names = ["Traveler" + str(index) for index in range(count)]
    for index, name in enumerate(names):
        eid = "request" + str(index)
        observe(memory, eid, f"{name} follows a declared dinner requirement.", index)
        rows = []
        if index:
            rows.append(binding("join" + str(index), name, eid, prior=f"decision:{names[index-1]}:day1:dinner"))
        for local_index in range(local_count):
            allowed = ["Actual Cafe, Boston", *[f"Permitted alternative {index} {local_index} {j} with descriptive entity name" for j in range(12)]]
            rows.append(binding(f"local{index}_{local_index}", name, eid, value=allowed, operator="IN"))
        memory.begin_subtask(request(name, index, extraction={"constraints": rows}))
        memory.commit_subtask({**request(name, index), "action": "Actual Cafe, Boston", "decisions": [choice(name)]})
    return names


class MaterializedFrontierTests(unittest.TestCase):
    def memory(self, variant="E3F", **config):
        memory = create_backend(variant, {"token_budget": 100000, **config})
        memory.reset("g")
        return memory

    def consume(self, memory, prior, step, name="Current"):
        eid = "current-request"
        observe(memory, eid, f"{name} joins {prior} for dinner.", step)
        constraint = binding("current-binding", name, eid, prior=f"decision:{prior}:day1:dinner")
        return memory.begin_subtask(request(name, step, f"Join {prior} for dinner.", {"constraints": [constraint]}))

    def test_frontier_stops_at_actual_choice_but_archive_and_invalidation_follow_full_chain(self):
        memory = self.memory()
        names = seed_chain(memory)
        context = self.consume(memory, names[-1], 4)
        self.assertEqual(context["status"], "OK")
        self.assertEqual(set(context["closure_ids"]), {"current-binding@v1", "decision:Traveler3:day1:dinner@v1"})
        self.assertEqual(context["closure_scope"], "materialized_frontier")
        self.assertFalse(context["full_historical_closure_visible"])
        archive, missing = memory._closure(["current-binding"], 4)
        self.assertEqual(missing, [])
        self.assertEqual({r["decision_id"] for r in archive if "decision_id" in r}, {f"decision:{name}:day1:dinner" for name in names})
        original = [r for r in memory.snapshot()["decisions"] if r["status"] == "COMMITTED"]
        # The administrative archive primitive must retain full graph scope,
        # even though E3F does not expose automatic E5 repair as a treatment.
        recovery = memory._invalidate(["decision:Traveler0:day1:dinner"], 5, "test_archive_scope")
        self.assertEqual(set(recovery["affected_decisions"]), {f"decision:{name}:day1:dinner" for name in names})
        for record in original:
            self.assertIn(record, memory.snapshot()["decisions"])

    def test_property_frontier_preserves_actual_entity_version_units_and_sources(self):
        memory = self.memory()
        observe(memory, "tool", {"Name": "Actual Cafe", "City": "Boston", "Average Cost": 80.0}, 0, "tool_result")
        memory.begin_subtask(request("Alice", 0))
        memory.commit_subtask({**request("Alice", 0), "action": "Actual Cafe, Boston", "decisions": [choice("Alice"),
            choice("Alice", 80.0, attribute="price", unit="USD", source_refs=["event:tool", "event:action"])]})
        observe(memory, "current-request", "Current dinner should cost at least 15 less than Alice dinner.", 1)
        c = binding("current-binding", "Current", "current-request", prior="decision:Alice:day1:dinner:price", operator="LTE")
        c["subject"]["attribute"] = "price"
        c["target"].update(scale=1, offset=-15, unit="USD")
        context = memory.begin_subtask(request("Current", 1, "Alice dinner price", {"constraints": [c]}))
        self.assertEqual(set(context["closure_ids"]), {"current-binding@v1", "decision:Alice:day1:dinner@v1", "decision:Alice:day1:dinner:price@v1"})
        price = next(r for r in context["records"] if r.get("attribute") == "price" and "decision_id" in r)
        self.assertEqual(price["entity_ref"], "decision:Alice:day1:dinner@v1")
        self.assertEqual(price["value"], 80.0)
        self.assertEqual(price["unit"], "USD")
        self.assertIn("event:tool", price["source_refs"])
        self.assertIn('"entity_ref":"decision:Alice:day1:dinner@v1"', context["context"])
        self.assertIn('"offset":-15', context["context"])

    def test_live_binding_uses_current_choice_and_preserves_old_action_in_archive(self):
        memory = self.memory()
        seed_choice(memory, value="Old Cafe, Boston")
        memory.commit_subtask({**request("Alice", 1), "action": "New Cafe, Boston", "replace_travelers": ["Alice"], "decisions": [choice("Alice", "New Cafe, Boston")]})
        context = self.consume(memory, "Alice", 2)
        prior = [r for r in context["records"] if r.get("traveler") == "Alice"]
        self.assertEqual([r["value"] for r in prior], ["New Cafe, Boston"])
        self.assertEqual(prior[0]["version"], 2)
        self.assertTrue(any(r["value"] == "Old Cafe, Boston" for r in memory.snapshot()["decisions"]))

    def test_changed_entity_property_is_unresolved_and_new_entity_remains_queryable(self):
        memory = self.memory()
        memory.begin_subtask(request("Alice", 0))
        memory.commit_subtask({**request("Alice", 0), "action": "Old Cafe, Boston; price80", "decisions": [choice("Alice", "Old Cafe, Boston"), choice("Alice", 80, attribute="price")]})
        # Deliberately use a patch without the replacement hint: freshness still
        # cannot assign the old entity's observed fact to the revised entity.
        memory.commit_subtask({**request("Alice", 1), "action": "New Cafe, Boston", "decisions": [choice("Alice", "New Cafe, Boston")]})
        context = memory.begin_subtask(request("Current", 2, dependency_refs=["decision:Alice:day1:dinner:price"]))
        self.assertEqual(context["status"], "UNRESOLVED")
        self.assertIn("decision:Alice:day1:dinner:price", context["missing_refs"])
        self.assertNotIn(80, [r.get("value") for r in context["records"]])
        self.assertIn("New Cafe, Boston", [r.get("value") for r in context["records"]])
        self.assertIn("STALE_PROPERTY_EVIDENCE", {issue["code"] for issue in context["diagnostics"]})

    def test_withdrawn_and_future_choices_never_become_materialized_facts(self):
        memory = self.memory()
        seed_choice(memory, value="Visible Cafe, Boston")
        memory.commit_subtask({**request("Alice", 10), "action": "Future Cafe, Boston", "decisions": [choice("Alice", "Future Cafe, Boston")]})
        earlier = memory.begin_subtask(request("Current", 2, dependency_refs=["decision:Alice:day1:dinner"]))
        self.assertEqual([r["value"] for r in earlier["records"]], ["Visible Cafe, Boston"])
        memory.commit_subtask({**request("Alice", 11), "action": "Malformed full replacement", "replace_travelers": ["Alice"], "decisions": []})
        context = memory.begin_subtask(request("Current", 12, dependency_refs=["decision:Alice:day1:dinner"]))
        self.assertEqual(context["status"], "UNRESOLVED")
        self.assertEqual(context["records"], [])
        self.assertEqual(context["missing_refs"], ["decision:Alice:day1:dinner"])

    def test_companion_filter_suppresses_only_complete_known_name_lists(self):
        memory = self.memory()
        seed_choice(memory, "Alice")
        seed_choice(memory, "Bob", 1)
        for line in ("I'm traveling with Alice and Bob.", "I’m going on this trip with Alice, and Bob."):
            with self.subTest(line=line):
                context = memory.begin_subtask(request("Current", 2, line, {"constraints": [], "parse_success": True, "unresolved": []}))
                self.assertEqual(context["records"], [])
        for line in ("I'm traveling with Alice, whose dinner I want to join.",
                     "I'm traveling with Alice and Unknown.", "I'm traveling with Alice,, Bob.",
                     "I'm joining Alice and Bob for this trip."):
            with self.subTest(line=line):
                context = memory.begin_subtask(request("Current", 2, line, {"constraints": [], "parse_success": True, "unresolved": []}))
                self.assertTrue(any(r.get("traveler") == "Alice" for r in context["records"]))

    def test_successful_empty_parse_does_not_suppress_real_requirement_name_fallback(self):
        memory = self.memory()
        seed_choice(memory, "Alice", extra_choices=[choice("Alice", "Breakfast Cafe, Boston", day=5, slot="breakfast")])
        # Public development wording shape, with synthetic actual selections.
        query = "I'm traveling with Alice.\nOn day 5, I'd like to join Alice for breakfast."
        context = memory.begin_subtask(request("Current", 2, query, {"constraints": [], "parse_success": True, "unresolved": []}))
        self.assertEqual({r["decision_id"] for r in context["records"] if "decision_id" in r},
                         {"decision:Alice:day5:breakfast"})

    def test_missing_required_record_is_not_declared_resolved(self):
        memory = self.memory()
        context = self.consume(memory, "AbsentTraveler", 1)
        self.assertEqual(context["status"], "UNRESOLVED")
        self.assertEqual(context["missing_refs"], ["decision:AbsentTraveler:day1:dinner"])
        self.assertIn("current-binding@v1", context["closure_ids"])

    def test_partial_parse_keeps_other_literal_slots_for_same_person(self):
        memory = self.memory()
        seed_choice(memory, extra_choices=[choice("Alice", "Breakfast Cafe, Boston", slot="breakfast")])
        query = "On day 1, join Alice for dinner.\nOn day 1, also join Alice for breakfast."
        observe(memory, "current-request", query, 1)
        c = binding("current-binding", "Current", "current-request", prior="decision:Alice:day1:dinner")
        context = memory.begin_subtask(request("Current", 1, query, {"constraints": [c], "parse_success": True, "unresolved": []}))
        self.assertEqual({r["decision_id"] for r in context["records"] if "decision_id" in r},
                         {"decision:Alice:day1:dinner", "decision:Alice:day1:breakfast"})

    def test_possessive_reference_scope_fetches_prior_day_and_slot(self):
        memory = self.memory()
        seed_choice(memory, extra_choices=[choice("Alice", "Prior Lunch, Boston", day=4, slot="lunch")])
        query = "On the second day, lunch should cost less than Alice's day 4 lunch."
        context = memory.begin_subtask(request("Current", 1, query, {"constraints": [], "parse_success": True}))
        self.assertEqual({r["decision_id"] for r in context["records"] if "decision_id" in r}, {"decision:Alice:day4:lunch"})

    def test_ambiguous_reference_retains_entities_without_unrelated_property_bulk(self):
        memory = self.memory()
        memory.begin_subtask(request("Alice", 0))
        memory.commit_subtask({**request("Alice", 0), "action": "Actual Cafe, Boston; price80", "decisions":
            [choice("Alice"), choice("Alice", 80, attribute="price")]})
        # Simulate a correctly extracted edge plus an unparsed vague mention.
        query = "Use Alice's selection."
        context = memory.begin_subtask(request("Current", 1, query,
            {"constraints": [binding("current-binding", "Current", "none", prior="decision:Alice:day1:dinner")]}))
        self.assertTrue(any(r.get("decision_id") == "decision:Alice:day1:dinner" for r in context["records"]))
        self.assertFalse(any(r.get("decision_id") == "decision:Alice:day1:dinner:price" for r in context["records"]))

    def test_small_frontier_budget_overflow_stays_explicit(self):
        memory = self.memory(token_budget=30)
        seed_choice(memory)
        context = self.consume(memory, "Alice", 1)
        self.assertEqual(context["status"], "BUDGET_OVERFLOW")
        self.assertEqual(context["closure_ids"], [])
        self.assertEqual(set(context["required_closure_ids"]), {"current-binding@v1", "decision:Alice:day1:dinner@v1"})
        self.assertTrue(context["batches"])
        self.assertLessEqual(context["token_count"], 30)

    def test_candidate_rejects_feature_overrides_that_change_the_named_condition(self):
        for key in ("typed_ledger", "dependency_closure", "protected_compression"):
            with self.subTest(feature=key):
                with self.assertRaises(ValueError): self.memory(**{key: False})
        for key in ("deterministic_validation", "targeted_recovery", "glossary_hops"):
            with self.subTest(feature=key):
                with self.assertRaises(ValueError): self.memory(**{key: True})

    def test_frontier_is_group_local(self):
        memory = self.memory()
        seed_choice(memory)
        memory.reset("other")
        context = memory.begin_subtask({"group_id": "other", "traveler": "Current", "step": 0, "query": "Alice dinner", "dependency_refs": ["decision:Alice:day1:dinner"]})
        self.assertEqual(context["status"], "UNRESOLVED")
        self.assertEqual(context["records"], [])
        self.assertEqual(memory.snapshot()["evidence"], [])

    def test_pinned_qwen_4096_budget_frontier_fits_while_full_chain_overflows(self):
        tokenizer_path = Path(__file__).resolve().parents[1] / "external/qwen-tokenizer/tokenizer.json"
        if not tokenizer_path.exists():
            self.skipTest("Pinned Qwen tokenizer is unavailable in this checkout")
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("Optional pinned tokenizer dependency unavailable")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        contexts = {}
        for variant in ("E3", "E3F"):
            memory = self.memory(variant, token_budget=4096)
            memory.set_token_counter(lambda text: len(tokenizer.encode(text).ids), "pinned Qwen tokenizer")
            names = seed_chain(memory, count=8, local_count=12)
            contexts[variant] = self.consume(memory, names[-1], 8)
            self.assertGreater(len(memory.snapshot()["constraints"]), 90)
            self.assertGreater(memory.stats()["storage_bytes"], 100000)
        self.assertEqual(contexts["E3"]["status"], "BUDGET_OVERFLOW")
        self.assertEqual(contexts["E3F"]["status"], "OK")
        self.assertLessEqual(contexts["E3F"]["token_count"], 4096)
        self.assertEqual(len(contexts["E3F"]["required_closure_ids"]), 2)
        self.assertGreater(len(contexts["E3"]["required_closure_ids"]), 90)


if __name__ == "__main__":
    unittest.main()
