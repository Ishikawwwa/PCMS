import pytest

from .black_box_memory import BlackBoxMemory
from .live_memory import Evidence


def test_trusted_observation_subject_overrides_shared_topic_from_writer(tmp_path):
    def complete(operation, system, data, schema):
        assert data["evidence_subject"] in {"Orchid", "Maple"}
        return {"records": [{"span": 0, "concepts": ["Deployment"], "status": "confirmed",
                             "table": "Deployment", "row": "region", "subject": "Deployment"}]}

    path = tmp_path / "memory.jsonl"
    path.with_suffix(".pending.json").write_text(
        '{"id":"one","kind":"tool","text":"Orchid switched to west."}', encoding="utf-8")
    memory = BlackBoxMemory(path, complete, len)
    memory.observe(Evidence("one", "tool", "Orchid switched to west.", subject="Orchid"))
    memory.observe(Evidence("two", "tool", "Maple switched to north.", subject="Maple"))
    restored = BlackBoxMemory(path, complete, len)
    snapshot = restored.snapshot()
    for subject, source in (("Orchid", "one"), ("Maple", "two")):
        row = snapshot.rows["Deployment"][f'["{subject}","region"]']
        assert row.subject == subject
        assert row.source == source
        assert snapshot.available(row)
        assert row in snapshot.subspaces[subject]


def test_ordered_tool_updates_keep_both_records_and_use_latest_row(tmp_path):
    def complete(operation, system, data, schema):
        assert operation == "write"
        return {"records": [
            {"span": index, "concepts": [subject], "status": "confirmed",
             "table": "Deployment", "row": "region", "subject": subject, "constraint_family": None}
            for index, subject in ((2, "Orchid"), (1, "Maple"), (0, "Orchid"))
        ]}

    path = tmp_path / "memory.jsonl"
    memory = BlackBoxMemory(path, complete, len)
    memory.observe(Evidence("changes", "tool", "Orchid switched to west.\nMaple switched to north.\nOrchid switched to east."))
    restored = BlackBoxMemory(path, complete, len)
    snapshot = restored.snapshot()
    records = snapshot.subspaces["Orchid"]
    assert [record.text for record in records] == ["Orchid switched to west.", "Orchid switched to east."]
    orchid_key, maple_key = '["Orchid","region"]', '["Maple","region"]'
    rows = snapshot.rows["Deployment"]
    assert rows[orchid_key] == records[1]
    assert rows[maple_key] == snapshot.subspaces["Maple"][0]
    assert snapshot.available(rows[maple_key])
    assert restored.catalog(snapshot)["row_subjects"]["Deployment"] == {orchid_key: "Orchid", maple_key: "Maple"}
    assert not snapshot.available(records[0])


def test_similar_constraints_share_a_family_and_keep_distinct_scopes(tmp_path):
    def complete(operation, system, data, schema):
        current = data["current_row_keys"]
        if data["source_kind"] == "tool":
            assert current == {"Orchid": [{"subject": "Orchid", "field": f"constraint / latency / {scope}"}
                                          for scope in ("startup", "steady state")]}
            assert data["known_constraint_families"] == {"Orchid": ["latency"]}
            rows = [(None, "observed / latency")]
        else:
            rows = [("Latency", "startup"), ("latency", "steady state")]
        return {"records": [{"span": 0, "concepts": ["Latency", "latency_20_to_40_ms"], "status": "confirmed",
                             "table": "Orchid", "row": row, "subject": "Orchid", "constraint_family": family}
                            for family, row in rows]}

    path = tmp_path / "memory.jsonl"
    memory = BlackBoxMemory(path, complete, len)
    requirement = "Orchid must respond in 20–40 ms at startup and within 10 ms in steady state."
    memory.observe(Evidence("requirement", "user", requirement))
    memory.observe(Evidence("sample", "tool", "Orchid responded in 32 ms."))
    snapshot = BlackBoxMemory(path, complete, len).snapshot()
    assert set(snapshot.subspaces) == {"Orchid", "Latency"}
    rows = snapshot.rows["Orchid"]
    for scope in ("startup", "steady state"):
        row = rows[f'["Orchid","constraint / latency / {scope}"]']
        assert (row.text, row.source, row.constraint_family) == (requirement, "requirement", "latency")
    observed = rows['["Orchid","observed / latency"]']
    assert observed.source == "sample"
    assert observed.constraint_family is None


@pytest.mark.parametrize("entity,field,before,after,query", [
    ("Neri", "meeting room", "Room A", "Room B", "Where should Neri meet the team?"),
    ("Orchid", "deployment region", "west", "east", "Where is Orchid deployed now?"),
])
def test_recall_uses_current_source_across_domains(tmp_path, entity, field, before, after, query):
    def complete(operation, system, data, schema):
        if operation == "read":
            assert data["query"] == query
            assert len(data["records"]) == 1
            return {"required_ids": list(data["records"]), "concepts": []}
        assert data["source_kind"] == "tool"
        return {"records": [{"span": data["spans"][0]["span"], "concepts": [entity],
                             "status": "confirmed", "table": entity, "row": field, "subject": entity}]}

    journal = tmp_path / "memory.jsonl"
    memory = BlackBoxMemory(journal, complete, len)
    original = f"{entity}: {field} = {before}"
    updated = f"{entity}: {field} = {after}"
    memory.observe(Evidence("old", "tool", original))
    memory.observe(Evidence("new", "tool", updated))
    restored = BlackBoxMemory(journal, complete, len)

    current = restored.recall(query, token_limit=200)
    assert current["delivered_ids"] == ["new:0"]
    assert updated in current["context"]
    historical = restored.recall(query, token_limit=200, through=1)
    assert historical["delivered_ids"] == ["old:0"]
    assert original in historical["context"]
    assert restored.recall(query, token_limit=1)["delivered_ids"] == []


def test_exact_reference_precedes_more_recent_related_record(tmp_path):
    def complete(operation, system, data, schema):
        if operation == "read":
            return {"required_ids": ["invented:0", "key:0"], "concepts": ["Aurora"]}
        return {"records": [{"span": 0, "concepts": ["Aurora"], "status": "confirmed",
                             "table": None, "row": None, "subject": None}]}

    memory = BlackBoxMemory(tmp_path / "memory.jsonl", complete, len)
    memory.observe(Evidence("key", "tool", "Aurora uses key K7."))
    memory.observe(Evidence("label", "tool", "Aurora uses label L2."))
    result = memory.recall("Which key does Aurora use?", token_limit=60)
    assert result["initial_ids"][0] == "label:0"
    assert result["delivered_ids"] == ["key:0"]
    assert result["tokens"] <= 60


def test_jev_reads_each_routed_concept_and_packs_full_source_within_budget(tmp_path):
    judged = []

    def complete(operation, system, data, schema):
        if operation == "concept-route":
            assert "records" not in data
            return {"concepts": ["Neri", "Orchid", "unknown"]}
        assert operation == "write"
        entity = data["spans"][0]["text"].split()[0]
        return {"records": [{"span": 0, "concepts": [entity], "status": "confirmed",
                             "table": None, "row": None, "subject": None}]}

    def judge(query, concept, records):
        judged.append((concept, list(records)))
        return {record_id: 0.1 if record_id == "lunch:0" else 0.9 for record_id in records}

    memory = BlackBoxMemory(tmp_path / "memory.jsonl", complete, len,
                            read_mode="jev-compact", judge=judge)
    meeting = "Neri has booked Room B for the team meeting this afternoon."
    memory.observe(Evidence("meeting", "tool", meeting))
    memory.observe(Evidence("copy", "tool", meeting))
    memory.observe(Evidence("lunch", "tool", "Neri has ordered lunch."))
    memory.observe(Evidence("region", "tool", "Orchid was deployed to the east region earlier this morning."))
    result = memory.recall("Where is the meeting and where is Orchid deployed?", token_limit=256)
    assert judged == [("Neri", ["meeting:0", "copy:0", "lunch:0"]), ("Orchid", ["region:0"])]
    assert result["delivered_ids"] == ["meeting:0", "region:0"]
    assert result["tokens"] <= 256
    assert result["context"].count(meeting) == 1
    constrained = memory.recall("Where is the meeting and where is Orchid deployed?", token_limit=110)
    assert constrained["delivered_ids"] == ["meeting:0"]
    assert meeting in constrained["context"]
