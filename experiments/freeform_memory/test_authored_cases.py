"""Mechanical integration of ten disclosed authored development workflows.

No annotation file is loaded here. These checks establish causal preservation,
source grounding, interface separation and real tokenizer budget compatibility;
they do not test natural-language interpretation or whether any choice is good.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import json
from pathlib import Path
import socket

import pytest
from tokenizers import Tokenizer

from constraint_memory.config import verify_tokenizer
from experiments.freeform_memory.runtime import (
    build_observed,
    commit_option,
    planner_messages,
    prepare_context,
    writer_messages,
)


ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "data/transfer/freeform_development.json").read_text(encoding="utf-8"))
CHOICES = [(case, offer["option_id"]) for case in CASES for offer in case["options"]]
NO_ORACLE_FIELDS = {
    "semantic_predicates", "predicate_id", "expected_options", "expected_decision",
    "predicate_statuses", "correct_option_id", "correct_options", "annotations",
    "ground_truth", "passed", "answer_key",
}


def json_pointer(content, pointer):
    """Resolve fixture paths independently of the production source accessor."""
    result = content
    for raw in pointer.split("/")[1:] if pointer else []:
        key = raw.replace("~1", "/").replace("~0", "~")
        result = result[int(key)] if isinstance(result, list) else result[key]
    return result


def assert_no_oracle_fields(value):
    if isinstance(value, dict):
        assert not set(value).intersection(NO_ORACLE_FIELDS)
        for child in value.values():
            assert_no_oracle_fields(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_oracle_fields(child)


@pytest.fixture(autouse=True)
def deny_annotation_credentials_and_network(monkeypatch):
    original_open = builtins.open
    original_path_open = Path.open

    def check_path(value):
        if isinstance(value, (str, bytes, Path)):
            name = str(value).replace("\\", "/").rsplit("/", 1)[-1].lower()
            if name in {"freeform_annotations.json", ".env", "author_fixtures.py"}:
                raise AssertionError("Runtime integration attempted to read scorer-only material or credentials")

    def guarded_open(file, *args, **kwargs):
        check_path(file)
        return original_open(file, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        check_path(path)
        return original_path_open(path, *args, **kwargs)

    def deny_connection(*args, **kwargs):
        raise AssertionError("Runtime integration must not use the network")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(socket, "create_connection", deny_connection)
    monkeypatch.setattr(socket.socket, "connect", deny_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_connection)


@pytest.fixture(scope="session")
def token_counter():
    path, manifest = verify_tokenizer(ROOT)
    assert manifest["model"] == "Qwen/Qwen3.6-27B"
    tokenizer = Tokenizer.from_file(str(path))
    return lambda value: len(tokenizer.encode(value).ids)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_all_authored_events_and_fact_revisions_build_without_semantic_contracts(case):
    before = copy.deepcopy(case)
    memory, step, users = build_observed(case)
    assert memory.constraints == {}
    assert step == max(event["step"] for event in case["events"]) + 1
    assert [row["content"] for row in users] == [event["content"] for event in case["events"] if event["kind"] == "user"] + [case["query"]]
    source_events = {event["event_id"]: event for event in case["events"]}
    for identity, event in source_events.items():
        assert memory.events[identity]["content"] == event["content"]
        assert memory.events[identity]["kind"] == event["kind"]
    expected_records = {}
    for event in case["events"]:
        for declaration in event.get("facts", []):
            expected_records.setdefault(declaration["ref"], []).append((event, declaration))
    assert set(memory.facts) == set(expected_records)
    for reference, declarations in expected_records.items():
        assert len(memory.facts[reference]) == len(declarations)
        for index, (event, declaration) in enumerate(declarations):
            record = memory.facts[reference][index]
            assert record["version"] == index + 1
            assert record["created_at"] == event["step"]
            assert record["value"] == json_pointer(event["content"], declaration["pointer"])
            assert record["source"]["event_id"] == event["event_id"]
            assert record["source"]["pointer"] == declaration["pointer"]
            assert record["unit"] == declaration.get("unit")
    assert case == before


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_all_authored_fact_ledgers_fit_pinned_qwen_memory_cap(case, token_counter):
    before = copy.deepcopy(case)
    context = prepare_context(case, "FACT_LEDGER", token_counter, 4096)
    assert context["status"] == "OK", (case["case_id"], context["required_tokens"])
    assert token_counter(context["context"]) == context["required_tokens"] <= 4096
    assert json.loads(context["context"]) == context["sections"]
    assert set(context["sections"]) == {"RAW_USER_REQUIREMENTS", "CURRENT_FACTS", "UNKNOWN_FACTS", "FACT_REVISIONS"}
    assert context["task_success"] is None and context["hard_semantic_checks"] is False
    assert [row["content"] for row in context["sections"]["RAW_USER_REQUIREMENTS"]] == [event["content"] for event in case["events"] if event["kind"] == "user"] + [case["query"]]
    assert_no_oracle_fields(context["sections"])
    assert case == before


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_authored_writer_receives_only_raw_requirements_fact_histories_and_schema(case):
    memory, _, users = build_observed(case)
    messages = writer_messages(case)
    assert [row["role"] for row in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {"raw_user_messages", "current_request", "output_fields", "operators", "fact_histories", "interpretation_semantics"}
    assert payload["raw_user_messages"] == users
    assert payload["current_request"] == case["query"]
    assert payload["fact_histories"] == memory.snapshot()["facts"]
    assert payload["output_fields"] == sorted({field["ref"] for option in case["options"] for field in option["properties"]})
    assert payload["interpretation_semantics"] == "UNVERIFIED"
    assert_no_oracle_fields(payload)
    # The writer does not get offered identities, hence cannot be handed a
    # correct-choice identity through catalog ordering or scoring metadata.
    for option in case["options"]:
        assert option["option_id"] not in messages[1]["content"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_authored_planner_receives_exact_observed_offers_with_binding_sources(case, token_counter):
    context = prepare_context(case, "FACT_LEDGER", token_counter, 4096)
    messages = planner_messages(case, context)
    assert [row["role"] for row in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {"request", "offers", "option_property_bindings", "memory"}
    events = {event["event_id"]: event for event in case["events"]}
    assert payload["offers"] == [json_pointer(events[option["event_id"]]["content"], option["pointer"]) for option in case["options"]]
    assert payload["option_property_bindings"] == case["options"]
    assert payload["request"] == case["query"]
    assert payload["memory"] == context["context"]
    assert_no_oracle_fields(payload)


@pytest.mark.parametrize("case,identity", CHOICES, ids=[case["case_id"] + "/" + identity for case, identity in CHOICES])
def test_every_authored_choice_commits_only_its_own_grounded_fields_without_grading(case, identity):
    before = copy.deepcopy(case)
    result = commit_option(case, {"option_id": identity})
    option = next(option for option in case["options"] if option["option_id"] == identity)
    event = next(event for event in case["events"] if event["event_id"] == option["event_id"])
    assert result["status"] == "COMMITTED_UNVERIFIED"
    assert result["task_success"] is None
    assert result["actual_response"] == {"option_id": identity}
    assert set(result["selected_facts"]) == {field["ref"] for field in option["properties"]}
    assert not result["snapshot"]["constraints"]
    source_hash = hashlib.sha256(json.dumps(event["content"], sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    for declaration in option["properties"]:
        record = result["selected_facts"][declaration["ref"]]
        assert record["value"] == json_pointer(event["content"], declaration["pointer"])
        assert record["unit"] == declaration.get("unit")
        assert record["source"] == {"event_id": event["event_id"], "pointer": declaration["pointer"], "event_sha256": source_hash}
        entity = result["snapshot"]["facts"][record["entity_ref"]["ref"]][record["entity_ref"]["version"] - 1]
        assert entity["value"] == identity
        assert result["snapshot"]["events"][entity["source"]["event_id"]]["kind"] == "action"
    # Fields absent from this observed offer stay absent, even when another
    # offer supplies the same reference. There is no certificate completion.
    all_fields = {field["ref"] for offer in case["options"] for field in offer["properties"]}
    for absent in all_fields - set(result["selected_facts"]):
        assert absent not in result["snapshot"]["facts"]
    assert_no_oracle_fields(result)
    assert case == before


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_abstention_and_unknown_identity_do_not_trigger_semantic_evaluation(case):
    abstained = commit_option(case, {"option_id": None})
    assert abstained == {"status": "ABSTAINED", "actual_response": {"option_id": None}, "selected_facts": {}}
    unknown = commit_option(case, {"option_id": "not-an-offered-identity"})
    assert unknown == {"status": "UNKNOWN_OPTION", "actual_response": {"option_id": "not-an-offered-identity"}}


def test_equal_time_check_in_conflict_survives_the_fact_ledger(token_counter):
    case = next(case for case in CASES if case["case_id"] == "freeform-check-in-disagreement")
    memory, step, _ = build_observed(case)
    assert memory.resolve("sera.floor", step) is None
    context = prepare_context(case, "FACT_LEDGER", token_counter, 4096)
    assert {"reference": "sera.floor", "status": "UNKNOWN_OR_CONFLICTING"} in context["sections"]["UNKNOWN_FACTS"]
    assert [record["value"] for record in context["sections"]["FACT_REVISIONS"]["sera.floor"]] == ["1", "2", "3"]
    assert not any(record["reference"] == "sera.floor" for record in context["sections"]["CURRENT_FACTS"])


def test_partial_catalog_fields_are_omitted_declarations_not_nonexistent_pointers():
    for case_id, identity, missing_reference in [
        ("freeform-certified-motor", "8eb427", "chosen.adapter_cert"),
        ("freeform-hospital-dispatch", "7da317", "chosen.carrier_cert"),
    ]:
        case = next(case for case in CASES if case["case_id"] == case_id)
        build_observed(case)
        option = next(option for option in case["options"] if option["option_id"] == identity)
        assert missing_reference not in {field["ref"] for field in option["properties"]}
        assert missing_reference not in commit_option(case, {"option_id": identity})["selected_facts"]


def test_runtime_rejects_accidental_annotation_injection_before_preparation():
    case = copy.deepcopy(CASES[0])
    case["semantic_predicates"] = [{"predicate_id": "scorer-only-poison"}]
    with pytest.raises(ValueError, match="task inputs only"):
        build_observed(case)


def integration_counts():
    """Budget diagnostics only; does not load annotations or score choices."""
    path, _ = verify_tokenizer(ROOT)
    tokenizer = Tokenizer.from_file(str(path))
    counter = lambda value: len(tokenizer.encode(value).ids)
    rows = []
    for case in CASES:
        context = prepare_context(case, "FACT_LEDGER", counter, 4096)
        writer = writer_messages(case)
        planner = planner_messages(case, context)
        rows.append({"case_id": case["case_id"], "memory_tokens": context["required_tokens"],
                     "writer_content_tokens": sum(counter(message["content"]) for message in writer),
                     "planner_content_tokens": sum(counter(message["content"]) for message in planner),
                     "offers": len(case["options"]), "status": context["status"]})
    return rows


if __name__ == "__main__":
    print(json.dumps(integration_counts(), indent=2))
