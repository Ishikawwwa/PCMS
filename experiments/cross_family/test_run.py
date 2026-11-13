"""Independent offline checks of the isolated cross-family coordinator.

Scripted sessions and invented scorer receipts test orchestration, not model
performance. No credential, hidden answer, network, or model call is needed.
"""
from __future__ import annotations

import builtins
from collections import Counter
import copy
import json
import math
from pathlib import Path
import socket
from types import SimpleNamespace
import uuid

import pytest

from constraint_memory.benchmark_adapter import load_splits
from constraint_memory.memory import create_backend
from constraint_memory.polza import PolzaClient, ProviderError
from experiments.cross_family import run


ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def destination():
    # Inherited workspace ACLs avoid this managed Windows host's inaccessible
    # mode-0700 pytest temporary directories, without changing frozen tests.
    path = ROOT / "tmp/cross-family-independent-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def protocol():
    return run.build_protocol(ROOT)


def forbid_private_reads(monkeypatch):
    original_path_open = Path.open
    original_open = builtins.open

    def check(path):
        if isinstance(path, (str, Path)):
            value = str(path).replace("\\", "/").lower()
            assert "/data/hidden/" not in value, "Planning opened hidden answer data"
            assert "/data/audit_annotations/" not in value, "Planning opened oracle annotations"
            assert not ("/data/visible/" in value and Path(path).name.startswith("turn_")), "Preparation opened a future request"
            assert Path(path).name.lower() != ".env", "Preparation opened credentials"

    def path_open(path, *args, **kwargs):
        check(path)
        return original_path_open(path, *args, **kwargs)

    def ordinary_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(builtins, "open", ordinary_open)


def test_preparation_needs_no_hidden_data_credentials_client_or_network(monkeypatch):
    forbid_private_reads(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation attempted live client or network access")

    monkeypatch.setattr(run, "make_client", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    if hasattr(run, "load_env"):
        monkeypatch.setattr(run, "load_env", forbidden)
    actual = run.build_protocol(ROOT)
    assert len(actual["group_ids"]) == 7
    json.dumps(actual, allow_nan=False)


def test_protocol_preserves_seven_unexposed_development_families_and_fixed_work(protocol):
    splits = load_splits(ROOT)
    group_ids = protocol["group_ids"]
    assert len(group_ids) == len(set(group_ids)) == 7
    assert set(group_ids) <= {str(value) for value in splits["partitions"]["dev"]}
    families = {splits["groups"][group]["base_template_id"] for group in group_ids}
    exposed_family = splits["groups"]["6"]["base_template_id"]
    assert families == set(splits["families"]["dev"]) - {exposed_family}
    assert not families & set(splits["families"]["validation"] + splits["families"]["holdout"])
    assert all(splits["groups"][group]["traveler_count"] == 7 for group in group_ids)
    assert protocol["config"]["experiment"]["partition"] == "dev"
    assert protocol["config"]["experiment"]["primary_contrast"] == ["E1", "E2"]
    assert protocol["bounds"] == {
        "group_conditions": 14,
        "added_traveler_commitments": 98,
        "max_actor_requests": 2940,
        "max_writer_requests": 98,
        "max_total_model_requests": 3038,
    }


def test_schedule_is_paired_balanced_deterministic_and_contains_no_extra_control(protocol):
    schedule = protocol["schedule"]
    assert len(schedule) == 14
    assert Counter((entry["group_id"], entry["variant"]) for entry in schedule) == Counter(
        (group, variant) for group in protocol["group_ids"] for variant in ("E1", "E2")
    )
    first_positions = Counter()
    for position in range(0, len(schedule), 2):
        first, second = schedule[position:position + 2]
        assert first["group_id"] == second["group_id"]
        assert {first["variant"], second["variant"]} == {"E1", "E2"}
        first_positions[first["variant"]] += 1
    assert sorted(first_positions.values()) == [3, 4]
    assert run.build_protocol(ROOT)["schedule"] == schedule


def test_conditions_differ_only_in_dependency_closure_and_memory_is_a_ceiling(protocol):
    config = protocol["config"]
    backends = [create_backend(variant, config["memory"]) for variant in ("E1", "E2")]
    changed = {key for key in backends[0].features if backends[0].features[key] != backends[1].features[key]}
    assert changed == {"dependency_closure"}
    assert backends[0].features["dependency_closure"] is False
    assert backends[1].features["dependency_closure"] is True
    assert config["agent"]["max_agent_visible_memory_tokens"] == 32768
    assert config["agent"]["max_steps_per_traveler"] == 30
    assert config["agent"]["max_completion_tokens"] == 16384
    assert config["agent"]["max_extraction_tokens"] == 16384
    sensitivity = run.build_protocol(ROOT, memory_tokens=8192)
    assert sensitivity["group_ids"] == protocol["group_ids"]
    assert sensitivity["schedule"] == protocol["schedule"]
    assert sensitivity["bounds"] == protocol["bounds"]
    assert sensitivity["config"]["agent"]["max_agent_visible_memory_tokens"] == 8192
    assert config["memory"]["public_base_outside_memory"] is True


@pytest.mark.parametrize("section,field,value", [
    ("agent", "model", "other/model"),
    ("agent", "temperature", 0.25),
    ("agent", "routing", {"only": ["chutes/fp8"], "allow_fallbacks": True}),
    ("agent", "max_steps_per_traveler", 31),
    ("agent", "max_completion_tokens", 8192),
    ("experiment", "seed", 20261014),
    ("experiment", "primary_contrast", ["E1", "E3"]),
    ("memory", "dependency_closure", True),
    ("memory", "targeted_recovery", True),
])
def test_effective_validator_rejects_protocol_or_mechanism_drift(protocol, section, field, value):
    config = copy.deepcopy(protocol["config"])
    config[section][field] = value
    with pytest.raises((ValueError, AssertionError)):
        run.validate_effective(config, 32768)


@pytest.mark.parametrize("cost", [10000.0, None])
def test_native_client_keeps_finite_call_caps_but_no_default_money_or_aggregate_stop(
    monkeypatch, protocol, destination, cost
):
    captured, payloads = {}, []

    def transport(method, path, payload):
        assert method == "POST" and path == "/chat/completions"
        payloads.append(copy.deepcopy(payload))
        return {
            "id": "offline-receipt-" + str(len(payloads)),
            "model": "qwen/qwen3.6-27b", "provider": "offline-test",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2000000, "completion_tokens": 1,
                      "total_tokens": 2000001, "cost_rub": cost},
        }

    original_client = run.JournalClient

    def offline_constructor(**kwargs):
        captured.update(kwargs)
        return original_client(**kwargs, api_key="offline-only-not-a-credential", transport=transport)

    monkeypatch.setattr(run, "JournalClient", offline_constructor)
    client = run.make_client(protocol["config"], destination)
    assert isinstance(client, PolzaClient)
    assert math.isinf(captured["max_total_tokens"]) and captured["max_total_tokens"] > 0
    assert captured["max_cost_rub"] is None
    assert captured["model"] == "qwen/qwen3.6-27b"
    assert captured["provider"] == {"only": ["chutes/fp8"], "allow_fallbacks": False}
    with client:
        client.set_scope("E1:120", "OfflineTraveler")
        client.complete([{"role": "user", "content": "Invented offline receipt."}])
        client.complete([{"role": "user", "content": "Invented offline writer receipt."}], operation="extraction")
    assert len(payloads) == 2
    assert all(payload["max_tokens"] == 16384 for payload in payloads)
    assert all(payload["model"] == "qwen/qwen3.6-27b" for payload in payloads)
    assert all(json.dumps(payload, allow_nan=False) for payload in payloads)
    assert all(record["model_identity_validation"] == "EXACT_MATCH" for record in client.records)
    assert client.get_usage_stats()["total_cost_rub"] == (20000.0 if cost is not None else None)


def test_catalog_accepts_exact_alias_only_and_does_not_substitute(destination):
    snapshot = destination / "catalog.json"
    write_json(snapshot, {"data": [{"id": "qwen/qwen3.6-27b-nearby"}]})
    with PolzaClient(api_key="offline-only", transport=lambda *args: pytest.fail("Unexpected network")) as client:
        with pytest.raises(ProviderError, match="Exact configured model"):
            client.verify_model(snapshot)
    write_json(snapshot, {"data": [{"id": "qwen/qwen3.6-27b", "offline_test": True}]})
    with PolzaClient(api_key="offline-only", transport=lambda *args: pytest.fail("Unexpected network")) as client:
        assert client.verify_model(snapshot)["id"] == "qwen/qwen3.6-27b"
        assert client.catalog_provenance["mode"] == "cached_catalog_snapshot"


def journal_client(destination, transport, **kwargs):
    return run.JournalClient(destination=destination, api_key="offline-only", transport=transport,
        max_total_tokens=math.inf, max_cost_rub=None, max_tokens=16384, **kwargs)


def test_rejected_model_receipt_is_durable_before_native_validation_and_remains_billable(destination):
    def transport(*args):
        return {"id": "rejected-offline", "model": "wrong/model",
            "choices": [{"message": {"content": "ignored"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost_rub": 1.25}}

    with journal_client(destination, transport) as client:
        client.set_scope("E1:120", "OfflineTraveler")
        original_record = client._record_usage

        def check_before_validation(record):
            saved = rows(destination / "committed_responses.jsonl")
            assert saved[-1]["raw_response"]["id"] == "rejected-offline"
            original_record(record)

        client._record_usage = check_before_validation
        with pytest.raises(ProviderError, match="different model"):
            client.complete([{"role": "user", "content": "offline"}])
        assert len(client.records) == 1
        assert client.records[0]["response_accepted"] is False
        assert client.get_usage_stats()["total_cost_rub"] == 1.25
    journal = rows(destination / "request_journal.jsonl")
    assert [item["event"] for item in journal] == ["INTENT_ONLY_NOT_DISPATCH_PROOF", "REQUEST_FAILED"]
    assert journal[-1]["request_sent"] is True


def test_malformed_tool_arguments_keep_original_response_and_usage_without_retry(destination):
    calls = []

    def transport(*args):
        calls.append(args)
        return {"id": "bad-tool-offline", "model": "qwen/qwen3.6-27b",
            "choices": [{"message": {"content": None, "tool_calls": [{"id": "tool1",
                "function": {"name": "offline", "arguments": "{bad JSON"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost_rub": .25}}

    with journal_client(destination, transport) as client:
        client.set_scope("E2:120", "OfflineTraveler")
        with pytest.raises(ProviderError, match="malformed tool-call"):
            client.chat_with_tools([{"role": "user", "content": "offline"}], [{"type": "function"}])
        assert len(client.records) == len(calls) == 1
    assert rows(destination / "committed_responses.jsonl")[0]["raw_response"]["id"] == "bad-tool-offline"


def test_ambiguous_sent_failure_is_journaled_once_and_never_retried(destination):
    calls = []

    def transport(*args):
        calls.append(args)
        raise ProviderError("offline interrupted response", request_sent=True, request_id="unknown-offline")

    with journal_client(destination, transport) as client:
        client.set_scope("E1:120", "OfflineTraveler")
        with pytest.raises(ProviderError, match="interrupted"):
            client.complete([{"role": "user", "content": "offline"}])
        assert len(calls) == len(client.records) == 1
        assert client.records[0]["total_tokens"] is None
        assert client.get_usage_stats()["total_cost_rub"] is None
    assert not (destination / "committed_responses.jsonl").exists()
    assert rows(destination / "request_journal.jsonl")[-1]["request_sent"] is True


def test_changed_freeze_blocks_dispatch_before_transport_or_intent(destination):
    def changed():
        raise ValueError("offline freeze changed")

    with journal_client(destination, lambda *args: pytest.fail("Changed source reached transport"), freeze_check=changed) as client:
        client.set_scope("E1:120", "OfflineTraveler")
        with pytest.raises(ValueError, match="freeze changed"):
            client.complete([{"role": "user", "content": "offline"}])
        assert client.records == []
    assert not (destination / "request_journal.jsonl").exists()


def test_interrupt_without_receipt_preserves_unknown_billing_and_journal(destination):
    def interrupted(*args):
        raise KeyboardInterrupt()

    with journal_client(destination, interrupted) as client:
        client.set_scope("E1:120", "OfflineTraveler")
        with pytest.raises(KeyboardInterrupt):
            client.complete([{"role": "user", "content": "offline"}])
        stats = client.get_usage_stats()
        assert stats["journal_attempts"] == stats["unaccounted_request_attempts"] == 1
        assert stats["total_cost_rub"] is None and stats["total_input_tokens"] is None
    assert rows(destination / "request_journal.jsonl")[-1]["error_type"] == "KeyboardInterrupt"


def test_repeated_names_do_not_create_an_extra_writer_stop(destination):
    def transport(*args):
        return {"id": "offline-repeat", "model": "qwen/qwen3.6-27b",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost_rub": 0}}

    with journal_client(destination, transport) as client:
        client.set_scope("E1:120", "RepeatedName")
        for _ in range(7):
            client.complete([{"role": "user", "content": "offline arrival"}], operation="extraction")
        with pytest.raises(ValueError, match="seven-arrival session"):
            client.complete([{"role": "user", "content": "extra arrival"}], operation="extraction")
        assert len(client.records) == 7


def fake_study(protocol, destination, *, fail_at=None, wrong_family=False):
    splits = load_splits(ROOT)
    state = SimpleNamespace(instances=[], completed=[], calls=[], active=False)
    readiness = {"provenance": {"dataset_sha256": "f" * 64}, "splits": splits}
    marker = object()

    class Session:
        def __init__(self, root, run_dir, variant, group_id, config, actual_readiness,
                     tokenizer, client, planner_factory, base_formatter):
            assert root == ROOT and run_dir == destination
            assert actual_readiness is readiness and tokenizer is marker and client is marker
            assert not state.active
            assert len(rows(destination / "session_completions.jsonl")) == len(state.completed)
            self.variant, self.group_id = variant, str(group_id)
            state.instances.append(self)

        def run(self):
            state.active = True
            if fail_at is not None and len(state.completed) == fail_at:
                state.active = False
                raise RuntimeError("Scripted transport interruption; no invented answer")
            family = splits["groups"][self.group_id]["base_template_id"]
            predictions = [{"group_id": self.group_id, "base_template_id": family,
                            "variant": self.variant, "traveler_index": index,
                            "plan": None if index == 3 else [{"offline_choice": index}],
                            "success": index != 3, "error_message": "scripted missing submission" if index == 3 else None}
                           for index in range(1, 8)]
            with (destination / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                for prediction in predictions:
                    stream.write(json.dumps(prediction) + "\n")
            row = {"group_id": self.group_id, "variant": self.variant, "base_template_id": family,
                   "replicate_id": 20261013, "traveler_count": 7, "person_count": 7, "usage": [],
                   "travelers": [{"traveler_index": index, "steps": 1, "offline_test": True} for index in range(7)]}
            state.completed.append((self.group_id, self.variant))
            state.active = False
            return predictions, row

    def scorer(upstream_root, hidden_path, predictions, group_ids, *, expected_dataset_sha):
        assert not state.active
        assert len(state.completed) == len(protocol["schedule"])
        assert len(rows(destination / "session_completions.jsonl")) == len(protocol["schedule"])
        assert group_ids == protocol["group_ids"] and expected_dataset_sha == "f" * 64
        assert Path(hidden_path) == ROOT / "data/hidden/memoryarena_travel.jsonl"
        assert len(predictions) == 49
        assert {prediction["group_id"] for prediction in predictions} == set(group_ids)
        assert all(prediction["plan"] is None for prediction in predictions if prediction["traveler_index"] == 3)
        variants = {prediction["variant"] for prediction in predictions}
        assert len(variants) == 1
        variant = variants.pop()
        state.calls.append(variant)
        # Reverse the result order to expose positional/zip score joins.
        per_group = []
        for group in reversed(group_ids):
            number = group_ids.index(group)
            per_group.append({"group_id": group,
                "base_template_id": "wrong-family" if wrong_family else splits["groups"][group]["base_template_id"],
                "official_metrics": {"SR": 0.0, "PS": (number + 1) / 7, "SPS": .5},
                "traveler_metrics": [{"traveler_index": index, "official_metrics": {"PS": float(index != 3)}}
                                     for index in range(1, 8)]})
        return {"per_group": per_group, "official_metrics": {"SR": 0.0, "PS": .5, "SPS": .5},
                "official_metric_metadata": {"scorer_isolated": True, "offline_test": True},
                "metric_definition": {"SPS": "scripted diagnostic SPS"}}

    state.invoke = lambda: run.run_study(ROOT, destination, protocol, readiness, marker, marker,
                                         session_factory=Session, scorer=scorer)
    return state


def test_sessions_are_fresh_sequential_and_scores_follow_all_actual_completions(
    monkeypatch, protocol, destination
):
    forbid_private_reads(monkeypatch)
    state = fake_study(protocol, destination)
    result = state.invoke()
    expected_order = [(entry["group_id"], entry["variant"]) for entry in protocol["schedule"]]
    assert state.completed == expected_order
    assert len({id(instance) for instance in state.instances}) == 14
    assert sorted(state.calls) == ["E1", "E2"]
    assert len(result) == len(rows(destination / "results.jsonl")) == 14
    assert len(rows(destination / "predictions.jsonl")) == 98
    assert len(rows(destination / "session_completions.jsonl")) == 14
    for row in result:
        assert row["official_metrics"]["PS"] == (protocol["group_ids"].index(row["group_id"]) + 1) / 7
        assert row["official_metric_metadata"]["offline_test"] is True
        assert [person["traveler_index"] for person in row["travelers"]] == list(range(7))
        assert row["travelers"][2]["official_metrics"]["PS"] == 0.0
        assert all(person["steps"] == 1 for person in row["travelers"]), "Coordinator must preserve native early stops"


def test_interruption_retains_actual_prefix_without_scoring_or_fabricating_zeros(protocol, destination):
    state = fake_study(protocol, destination, fail_at=2)
    with pytest.raises(RuntimeError, match="Scripted transport interruption"):
        state.invoke()
    assert len(state.completed) == len(rows(destination / "session_completions.jsonl")) == 2
    assert len(rows(destination / "predictions.jsonl")) == 14
    assert state.calls == []
    assert rows(destination / "results.jsonl") == []
    assert list(destination.glob("official_scores_*.json")) == []


def test_official_family_mismatch_is_rejected_instead_of_silently_joined(protocol, destination):
    state = fake_study(protocol, destination, wrong_family=True)
    with pytest.raises((ValueError, AssertionError), match="[Ff]amily"):
        state.invoke()
    assert len(state.completed) == 14
    assert rows(destination / "results.jsonl") == []
