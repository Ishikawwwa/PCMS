from __future__ import annotations

import copy
import unittest

from constraint_memory.memory import create_backend
from constraint_memory.memory.backends import token_estimate


def observe(memory, eid, content, step=0, event_type="user_request"):
    memory.observe({"event_id": eid, "step": step, "type": event_type, "content": content, "group_id": memory.group_id})


def request(traveler="Alice", query="dinner", step=0, **extra):
    return {"group_id": "g", "traveler": traveler, "query": query, "step": step, **extra}


def decision(value, traveler="Alice", **extra):
    return {"traveler": traveler, "day": 2, "slot": "dinner", "value": value, **extra}


def constraint(cid="c", value="vegan", traveler="Alice", **extra):
    return {"constraint_id": cid, "kind": "LOCAL", "scope": {"group": "g", "traveler": traveler},
            "subject": {"traveler": traveler, "day": 2, "slot": "dinner"}, "operator": "EQUALS", "target": {"value": value},
            "source_refs": ["event:r"], **extra}


class MemoryInvariantTests(unittest.TestCase):
    def memory(self, variant="E3", **config):
        memory = create_backend(variant, config)
        memory.reset("g")
        return memory

    def test_b0_has_no_persisted_trace(self):
        m = self.memory("B0")
        observe(m, "r", "Private prior action")
        m.commit_subtask({**request(), "action": "Private actual choice"})
        result = m.begin_subtask(request())
        self.assertEqual(result["context"], "")
        self.assertEqual(m.snapshot()["evidence"], [])

    def test_b1_and_b2_never_read_future_evidence(self):
        for variant in ("B1", "B2"):
            with self.subTest(variant=variant):
                m = self.memory(variant)
                observe(m, "r", "Alice dinner Blue Bistro", 0)
                observe(m, "future", "Alice dinner Secret Future Bistro", 9)
                result = m.begin_subtask(request(query="Alice dinner", step=0))
                self.assertIn("Blue Bistro", result["context"])
                self.assertNotIn("Secret Future", result["context"])
                self.assertEqual(result["evidence_ids"], ["r"])

    def test_b1_reports_overflow_without_truncating_history(self):
        m = self.memory("B1", token_budget=20)
        observe(m, "r", "entire trace " * 100)
        result = m.begin_subtask(request())
        self.assertEqual(result["status"], "BUDGET_OVERFLOW")
        self.assertIn("entire trace " * 100, result["context"])

    def test_bm25_ranks_query_terms_not_recent_unrelated_content(self):
        m = self.memory("B2", top_k=1)
        observe(m, "relevant", "Alice dinner venue price price", 0)
        observe(m, "unrelated", "train departure station timetable", 1)
        result = m.begin_subtask(request(query="Alice dinner price", step=1))
        self.assertEqual(result["evidence_ids"], ["relevant"])

    def test_matrix_features_are_independent(self):
        keys = ["typed_ledger", "dependency_closure", "protected_compression", "deterministic_validation", "targeted_recovery", "glossary_hops"]
        for level in range(7):
            m = self.memory(f"E{level}")
            self.assertEqual(m.features, {key: level >= idx + 1 for idx, key in enumerate(keys)})
        m = self.memory("E3", dependency_closure=False)
        self.assertFalse(m.features["dependency_closure"])
        self.assertTrue(m.features["protected_compression"])

    def test_e1_does_not_follow_unmentioned_dependency(self):
        for variant in ("E1", "E2"):
            m = self.memory(variant)
            m.begin_subtask(request())
            m.commit_subtask({**request(), "action": "Blue Bistro", "decisions": [decision("Blue Bistro")]})
            observe(m, "r", "Bob dinner equals Alice", 1)
            c = constraint(traveler="Bob")
            c["target"] = {"ref": "decision:Alice:day2:dinner"}
            result = m.begin_subtask(request("Bob", query="my dinner", step=1, extraction={"constraints": [c]}))
            present = "decision:Alice:day2:dinner@v1" in result["closure_ids"]
            self.assertEqual(present, variant == "E2")

    def test_evidence_is_immutable_and_deep_copied(self):
        m = self.memory()
        event = {"event_id": "r", "step": 0, "type": "tool_result", "content": {"price": 50}}
        m.observe(event)
        event["content"]["price"] = 999
        self.assertEqual(m.retrieve_evidence("event:r")["content"]["price"], 50)
        saved = m.snapshot()
        saved["evidence"][0]["content"]["price"] = 1000
        self.assertEqual(m.retrieve_evidence("event:r")["content"]["price"], 50)
        with self.assertRaises(ValueError): m.observe(event)
        with self.assertRaises(ValueError): m.retrieve_evidence("event:r", step=-1)

    def test_prohibited_fields_and_cross_group_inputs_rejected(self):
        m = self.memory()
        with self.assertRaises(ValueError): m.begin_subtask({**request(), "gold_plan": "secret"})
        with self.assertRaises(ValueError): m.begin_subtask({**request(), "group_id": "other"})
        with self.assertRaises(ValueError): observe(m, "r", {"nested": {"oracle": 1}})

    def test_future_citation_rejected_at_write_boundary(self):
        m = self.memory("E4")
        observe(m, "r", "vegan", 8)
        result = m.begin_subtask(request(step=1, extraction={"constraints": [constraint()]}))
        self.assertEqual(result["status"], "VALIDATION_FAILED")
        self.assertIn("FUTURE_EVIDENCE", {e["code"] for e in result["diagnostics"]})
        self.assertEqual(m.snapshot()["constraints"], [])

    def test_version_visibility_and_stale_dependency(self):
        m = self.memory("E4")
        m.begin_subtask(request())
        m.commit_subtask({**request(), "action": "Blue Bistro", "decisions": [decision("Blue Bistro")]})
        m.commit_subtask({**request(step=2), "action": "Green Bistro", "decisions": [decision("Green Bistro")]})
        result = m.begin_subtask(request("Bob", "Alice dinner", 1))
        self.assertIn("Blue Bistro", result["context"])
        self.assertNotIn("Green Bistro", result["context"])
        result = m.begin_subtask(request("Bob", "same dinner", 3, dependency_refs=["decision:Alice:day2:dinner@v1"]))
        self.assertIn("STALE_DEPENDENCY", {e["code"] for e in result["diagnostics"]})
        self.assertEqual(result["missing_refs"], ["decision:Alice:day2:dinner@v1"])

    def test_duplicate_versions_are_not_silently_overwritten(self):
        m = self.memory("E4")
        observe(m, "r", "vegan seafood")
        c = constraint(version=1)
        m.begin_subtask(request(extraction={"constraints": [c]}))
        changed = copy.deepcopy(c)
        changed["target"]["value"] = "seafood"
        result = m.begin_subtask(request(extraction={"constraints": [changed]}))
        self.assertIn("DUPLICATE_VERSION", {e["code"] for e in result["diagnostics"]})
        self.assertEqual(m.snapshot()["constraints"][0]["target"]["value"], "vegan")
        self.assertEqual(len(m.snapshot()["constraints"]), 1)

    def test_repeated_extraction_is_idempotent(self):
        m = self.memory("E4")
        observe(m, "r", "vegan")
        for _ in range(3): m.begin_subtask(request(extraction={"constraints": [constraint()]}))
        self.assertEqual(len(m.snapshot()["constraints"]), 1)

    def test_claimed_success_is_never_a_satisfaction_proof(self):
        m = self.memory("E4")
        observe(m, "r", "vegan")
        m.begin_subtask(request(extraction={"constraints": [constraint(status="SATISFIED")]}))
        self.assertEqual(m.snapshot()["constraints"][0]["status"], "UNRESOLVED")
        result = m.commit_subtask({**request(), "action": "seafood", "decisions": [decision("seafood")], "success": True})
        self.assertIn("SUBMISSION_VIOLATION", {e["code"] for e in result["diagnostics"]})
        self.assertEqual(m.snapshot()["constraints"][0]["status"], "UNRESOLVED")

    def test_extracted_choice_must_match_actual_action(self):
        m = self.memory("E4")
        observe(m, "r", "Red Restaurant")
        m.begin_subtask(request())
        result = m.commit_subtask({**request(), "action": "Blue Bistro", "decisions": [decision("Red Restaurant", source_refs=["event:r"])]})
        self.assertIn("ACTUAL_ACTION_MISMATCH", {e["code"] for e in result["diagnostics"]})
        self.assertFalse(any(r["status"] == "COMMITTED" for r in m.snapshot()["decisions"]))
        self.assertEqual(m.snapshot()["outcomes"][0]["actual_action"], "Blue Bistro")

    def test_tool_property_is_anchored_to_submitted_entity(self):
        for venue, expected_code in (("Blue Bistro", None), ("Red Restaurant", "PROPERTY_ENTITY_MISMATCH")):
            with self.subTest(venue=venue):
                m = self.memory("E4")
                observe(m, "tool", {"Name": venue, "City": "Paris", "Price": 80}, event_type="tool_result")
                m.begin_subtask(request())
                attribute = decision(80, attribute="price", source_refs=["event:tool"])
                result = m.commit_subtask({**request(), "action": "Dinner: Blue Bistro, Paris", "decisions": [decision("Blue Bistro, Paris"), attribute]})
                errors = {e["code"] for e in result["diagnostics"]}
                if expected_code:
                    self.assertIn(expected_code, errors)
                else:
                    self.assertEqual(result["status"], "OK")
                    price = next(r for r in m.snapshot()["decisions"] if r.get("attribute") == "price")
                    self.assertEqual(price["entity_ref"], "decision:Alice:day2:dinner@v1")

    def test_exact_runtime_token_counter_controls_compression(self):
        m = self.memory("E3", token_budget=150)
        m.set_token_counter(len, "test_exact_character_counter")
        observe(m, "r", "vegan")
        result = m.begin_subtask(request(extraction={"constraints": [constraint()]}))
        self.assertEqual(result["status"], "BUDGET_OVERFLOW")
        self.assertLessEqual(result["token_count"], 150)
        self.assertEqual(result["token_estimator"], "test_exact_character_counter")
        self.assertTrue(any(b["status"] == "RECORD_EXCEEDS_BUDGET" for b in result["batches"]))

    def test_protected_budget_never_disguises_loss(self):
        m = self.memory("E3", token_budget=10)
        observe(m, "r", "vegan")
        result = m.begin_subtask(request(extraction={"constraints": [constraint()]}))
        self.assertEqual(result["status"], "BUDGET_OVERFLOW")
        self.assertEqual(result["required_closure_ids"], ["c@v1"])
        self.assertEqual(result["closure_ids"], [])
        self.assertEqual(result["coverage"]["recall"], 0)
        self.assertTrue(m.retrieve_evidence("event:r"))
        self.assertIn("c", m.snapshot()["retrieval_index"])

    def test_drop_policy_removes_index_only_archive_survives(self):
        m = self.memory("E0", token_budget=10, compression_mode="drop")
        observe(m, "r", "vegan")
        result = m.begin_subtask(request(extraction={"constraints": [constraint()]}))
        self.assertEqual(result["status"], "PARTIAL_CONTEXT")
        self.assertNotIn("c", m.snapshot()["retrieval_index"])
        self.assertEqual(len(m.snapshot()["constraints"]), 1)
        self.assertEqual(m.retrieve_evidence("event:r")["content"], "vegan")

    def test_e4_no_revision_control_preserves_committed_branch(self):
        m = self.memory("E4")
        m.begin_subtask(request())
        m.commit_subtask({**request(), "action": "Blue Bistro", "decisions": [decision("Blue Bistro")]})
        result = m.commit_subtask({**request(step=1), "action": "invalidating observation", "invalidate_decision_refs": ["decision:Alice:day2:dinner"]})
        self.assertIn("REVISION_DISABLED", {e["code"] for e in result["diagnostics"]})
        self.assertEqual(m.snapshot()["decisions"][0]["status"], "COMMITTED")
        self.assertEqual(m.snapshot()["recovery_log"], [])

    def test_invalidated_outcome_does_not_leak_back_via_named_traveler(self):
        m = self.memory("E5")
        m.begin_subtask(request())
        m.commit_subtask({**request(), "action": "Blue Bistro", "decisions": [decision("Blue Bistro")]})
        m.commit_subtask({**request(step=1), "action": "invalidating observation", "invalidate_decision_refs": ["decision:Alice:day2:dinner"]})
        result = m.begin_subtask(request("Bob", "Alice dinner", 2))
        self.assertNotIn("Blue Bistro", result["context"])
        self.assertEqual(len(m.snapshot()["outcomes"]), 2)

    def test_byte_estimator_handles_unicode_reproducibly(self):
        self.assertEqual(token_estimate(""), 0)
        self.assertEqual(token_estimate("abcd"), 1)
        self.assertEqual(token_estimate("東京"), 2)

    def test_tiny_exact_counter_budget_is_respected(self):
        m = self.memory("E3", token_budget=2)
        m.set_token_counter(len)
        observe(m, "r", "vegan")
        result = m.begin_subtask(request(extraction={"constraints": [constraint()]}))
        self.assertEqual(result["status"], "BUDGET_OVERFLOW")
        self.assertLessEqual(len(result["context"]), 2)

    def test_explicit_binding_avoids_unrelated_prior_slots(self):
        m = self.memory("E3")
        m.begin_subtask(request())
        dinner = decision("Blue Bistro")
        lunch = {**decision("Yellow Cafe"), "slot": "lunch"}
        m.commit_subtask({**request(), "action": "Dinner Blue Bistro; lunch Yellow Cafe", "decisions": [dinner, lunch]})
        observe(m, "r", "Bob dinner equals Alice dinner", 1)
        c = constraint(traveler="Bob")
        c["target"] = {"ref": "decision:Alice:day2:dinner"}
        result = m.begin_subtask(request("Bob", "same dinner as Alice", 1, extraction={"constraints": [c]}))
        self.assertIn("decision:Alice:day2:dinner@v1", result["closure_ids"])
        self.assertNotIn("decision:Alice:day2:lunch@v1", result["closure_ids"])

    def test_tool_tables_are_not_repeated_in_visible_context(self):
        m = self.memory("E3")
        observe(m, "r", "Alice dinner vegan")
        c = constraint(evidence_quote="Alice dinner vegan", narrative="NOISY_TOOL_TABLE" * 100)
        result = m.begin_subtask(request(extraction={"constraints": [c]}))
        self.assertNotIn("evidence_quote", result["context"])
        self.assertNotIn("NOISY_TOOL_TABLE", result["context"])
        self.assertIn("source_refs", result["context"])
        self.assertEqual(m.snapshot()["constraints"][0]["evidence_quote"], "Alice dinner vegan")

    def test_parser_uncertainty_is_visible(self):
        m = self.memory("E3")
        result = m.begin_subtask(request(extraction={"constraints": [], "parse_success": False, "unresolved": [{"reason": "ambiguous traveler"}]}))
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertIn("ambiguous traveler", result["context"])
        self.assertEqual({e["code"] for e in result["diagnostics"]}, {"PARSER_FAILURE", "PARSER_UNRESOLVED"})

    def test_numeric_trace_does_not_match_a_larger_number(self):
        m = self.memory("E4")
        observe(m, "r", "Alice dinner price 400")
        c = constraint(value=40)
        c["subject"]["attribute"] = "price"
        result = m.begin_subtask(request(extraction={"constraints": [c]}))
        self.assertEqual(result["status"], "VALIDATION_FAILED")
        self.assertIn("EVIDENCE_MISMATCH", {e["code"] for e in result["diagnostics"]})

    def test_binding_dependency_cycle_is_diagnosed(self):
        m = self.memory("E3")
        observe(m, "r", "Alice vegan")
        a = constraint("a", depends_on=["b"])
        b = constraint("b", depends_on=["a"])
        result = m.begin_subtask(request(extraction={"constraints": [a, b]}))
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertIn("DEPENDENCY_CYCLE", {e["code"] for e in result["diagnostics"]})

    def test_original_v0_iterative_glossary_option(self):
        m = self.memory("E0", glossary_hops=True, max_glossary_hops=4)
        observe(m, "r", "Frodo follows Bilbo; Bilbo lives in Shire")
        definitions = []
        for term, text, related in (("Frodo", "Frodo follows Bilbo", ["Bilbo"]), ("Bilbo", "Bilbo lives in Shire", ["Shire"]), ("Shire", "Shire is a named location", [])):
            definitions.append({"term": term, "text": text, "related_terms": related, "parent_scope": {"group": "g"}, "evidence_refs": ["event:r"]})
        result = m.begin_subtask(request(query="Where is Frodo?", extraction={"definitions": definitions}))
        self.assertEqual(result["status"], "OK")
        self.assertEqual(set(result["glossary_ids"]), {"definition:Frodo", "definition:Bilbo", "definition:Shire"})
        self.assertFalse(result["features"]["typed_ledger"])
        self.assertFalse(result["features"]["dependency_closure"])

    def test_failed_new_branch_preserves_latest_valid_earlier_choice(self):
        m = self.memory("E4")
        m.begin_subtask(request())
        m.commit_subtask({**request(), "action": "Blue Bistro", "decisions": [decision("Blue Bistro")]})
        m.commit_subtask({**request(step=1), "action": "Rejected Restaurant", "decisions": [decision("Rejected Restaurant")], "branch_status": "FAILED", "status": "FAILED"})
        result = m.begin_subtask(request("Bob", "Alice dinner", 2))
        self.assertIn("Blue Bistro", result["context"])
        self.assertNotIn("Rejected Restaurant", result["context"])
        self.assertIn("decision:Alice:day2:dinner@v1", result["closure_ids"])


if __name__ == "__main__":
    unittest.main()
