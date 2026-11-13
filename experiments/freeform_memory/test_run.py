"""Independent fake-client checks of the free-form experiment driver.

All files belong to per-test temporary roots. No real credentials, tokenizer,
network, catalog lookup, or model completion is used. Fake usage is marked.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import json
import math
from pathlib import Path
import socket
from types import SimpleNamespace
import uuid
import zipfile

import pytest
import tokenizers

from constraint_memory.polza import BudgetExceeded, ProviderError
from experiments.freeform_memory import run

ACTUAL_VERIFY_FREEZE = run.verify_freeze


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_case(index):
    identity = f"driver-case-{index}"
    text = "Choose a listed batch with total at most the current allowance."
    case = {"case_id": identity, "domain": "generic-batch", "query": "Choose one current offer.", "format": "json",
            "events": [
                {"event_id": "initial", "step": 0, "kind": "observation", "content": {"limit": 100},
                 "facts": [{"ref": "allowance", "pointer": "/limit", "unit": "EUR"}]},
                {"event_id": "request", "step": 1, "kind": "user", "content": text},
                {"event_id": "revised", "step": 2, "kind": "observation", "content": {"limit": 80},
                 "facts": [{"ref": "allowance", "pointer": "/limit", "unit": "EUR"}]},
                {"event_id": "offers", "step": 3, "kind": "observation", "content": [
                    {"option_id": f"offer-{index}-a", "total": 70}, {"option_id": f"offer-{index}-b", "total": 90}]}],
            "options": [{"option_id": f"offer-{index}-{suffix}", "event_id": "offers", "pointer": f"/{position}",
                         "properties": [{"ref": "chosen.total", "pointer": f"/{position}/total", "unit": "EUR"}]}
                        for position, suffix in enumerate(("a", "b"))]}
    quote = "total at most the current allowance"
    start = text.index(quote)
    annotation = {"case_id": identity, "semantic_predicates": [{"predicate_id": "offline-only-limit",
                  "source": {"event_id": "request", "span": {"start": start, "end": start + len(quote)}, "quote": quote},
                  "expression": {"subject": {"ref": "chosen.total"}, "operator": "LTE",
                                 "target": {"ref": "allowance", "binding_mode": "LIVE"}, "unit": "EUR"}}]}
    return case, annotation


@pytest.fixture
def driver(monkeypatch):
    # Python's Windows mode-0700 temporary directories are inaccessible in
    # this managed workspace. Normal inherited-ACL UUID directories retain
    # the same per-test isolation without modifying the frozen conftest/suite.
    root = Path(__file__).resolve().parents[2] / "tmp/freeform-driver-review" / uuid.uuid4().hex
    paired = [fixture_case(index) for index in range(2)]
    state = SimpleNamespace(root=root, cases=[pair[0] for pair in paired], annotations=[pair[1] for pair in paired],
                            events=[], constructor_kwargs=[], clients=[], dispatches=[], invocations=[],
                            annotation_reads=0, load_env_calls=[], lock_entries=0, freeze_checks=0,
                            freeze_changed=False, after_dispatch=None, malformed_writer=False,
                            malformed_envelope_at=None, fail_at=None, cost=1.0, missing_cost=False,
                            scorer_refusal=False, fsync_calls=0)
    cases_path = root / "data/transfer/freeform_development.json"
    annotation_path = root / "data/transfer/freeform_annotations.json"
    for relative, content in {"src/sentinel.py": "VALUE = 1\n", "experiments/freeform_memory/sentinel.py": "VALUE = 2\n",
                              "configs/travel.yaml": "{}\n", "reports/polza_catalog_verified_connection.json": "{}\n",
                              "external/tokenizer.mock": "mock tokenizer; not a real model\n"}.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    write_json(cases_path, state.cases)
    write_json(annotation_path, {"cases": state.annotations})
    pin = {"source_sha256": {"src/sentinel.py": run.sha256(root / "src/sentinel.py")},
           "prototype_sha256": {"experiments/freeform_memory/sentinel.py": run.sha256(root / "experiments/freeform_memory/sentinel.py")},
           "runtime_inputs_sha256": {relative: run.sha256(root / relative) for relative in (
               "data/transfer/freeform_development.json", "configs/travel.yaml", "external/tokenizer.mock",
               "reports/polza_catalog_verified_connection.json")},
           "annotations_sha256": run.sha256(annotation_path), "verification": {"passed": 1}}
    state.pin = pin
    state.destination = lambda run_id="fake-driver": root / "runs" / run_id
    state.invoke = lambda *extra: run.main(["--live", "--budget-rub", "12", "--run-id", "fake-driver", *extra])

    class MockTokenizer:
        @classmethod
        def from_file(cls, path):
            assert Path(path) == root / "external/tokenizer.mock"
            return cls()

        def encode(self, text):
            return SimpleNamespace(ids=range(len(text)))

    class MockLock:
        def __init__(self, requested_root):
            assert requested_root == root

        def __enter__(self):
            state.lock_entries += 1
            state.events.append(("lock",))
            return self

        def __exit__(self, *args):
            return False

    def verify_freeze():
        state.freeze_checks += 1
        if state.freeze_changed:
            raise ValueError("Test freeze source changed")
        return copy.deepcopy(pin)

    def load_env(path):
        assert path == root / ".env"
        state.load_env_calls.append(path)
        state.events.append(("credential-loader-stub",))

    original_durable = run.durable
    original_fsync = run.os.fsync

    def fsync(descriptor):
        original_fsync(descriptor)
        state.fsync_calls += 1

    def durable(path, value, **kwargs):
        before = state.fsync_calls
        original_durable(path, value, **kwargs)
        assert state.fsync_calls > before, "Durable artifacts must be synced before the operation returns"
        state.events.append(("durable", Path(path).name, copy.deepcopy(value)))

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.records = []
            self.scope = None
            state.constructor_kwargs.append(copy.deepcopy(kwargs))
            state.clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.events.append(("client-close",))
            return False

        def verify_model(self, **kwargs):
            assert kwargs["snapshot_path"] == root / "reports/polza_catalog_verified_connection.json"
            state.events.append(("fake-catalog-verified",))

        def set_scope(self, group, scope):
            assert group == "freeform-suite"
            self.scope = scope

        def record(self, operation, *, failed=False, unknown=False):
            row = {"fake_client": True, "operation": operation,
                   "category": "maintenance" if operation == "extraction" else "agent",
                   "input_tokens": None if unknown else 20, "output_tokens": None if unknown else 5,
                   "total_tokens": None if unknown else 25,
                   "cost_rub": None if unknown or state.missing_cost else state.cost,
                   "request_sent": True, "request_status": "failed" if failed else "completed"}
            self.records.append(row)
            run.durable(self.kwargs["usage_path"], row, append=True)

        def complete(self, messages, *, operation, json_output):
            assert json_output is True
            state.invocations.append((self.scope, operation))
            journal = rows(self.kwargs["usage_path"].parent / "request_journal.jsonl")
            assert journal[-1]["event"] == "INTENT_ONLY_NOT_DISPATCH_PROOF"
            assert journal[-1]["phase"] == ("writer" if operation == "extraction" else "planner")
            assert journal[-1]["messages"] == messages
            # The fake implements only the accounting boundary needed to test
            # the coordinator. It never invokes the actual Polza transport.
            if self.kwargs["max_cost_rub"] is not None and any(record["cost_rub"] is None for record in self.records):
                raise BudgetExceeded("fake: prior billing remains unknown")
            if self.kwargs["max_cost_rub"] is not None and sum(record["cost_rub"] for record in self.records) >= self.kwargs["max_cost_rub"]:
                raise BudgetExceeded("fake: explicit spending threshold reached")
            state.dispatches.append({"scope": self.scope, "operation": operation, "messages": copy.deepcopy(messages)})
            number = len(state.dispatches)
            state.events.append(("fake-dispatch", self.scope, operation))
            if state.fail_at == number:
                self.record(operation, failed=True, unknown=True)
                raise ProviderError("fake ambiguous transport failure", request_sent=True)
            self.record(operation)
            case_id, condition = self.scope.rsplit(":", 1)
            case = next(case for case in state.cases if case["case_id"] == case_id)
            text = ("{not-json" if state.malformed_writer else json.dumps({"constraints": [], "unresolved": []})) if operation == "extraction" else json.dumps({"option_id": case["options"][0]["option_id"]})
            raw = {"id": f"fake-response-{number}", "model": self.kwargs["model"],
                   "choices": [{"message": {"content": text}}]}
            if state.malformed_envelope_at == number:
                raw["choices"] = []
            if state.after_dispatch:
                state.after_dispatch(number, self.scope, operation)
            state.events.append(("fake-received", copy.deepcopy(raw)))
            return raw

        def get_usage_stats(self):
            def total(field):
                return sum(record[field] for record in self.records) if all(record[field] is not None for record in self.records) else None
            return {"fake_client": True, "calls": len(self.records), "total_input_tokens": total("input_tokens"),
                    "total_output_tokens": total("output_tokens"), "known_input_tokens": sum(record["input_tokens"] or 0 for record in self.records),
                    "known_output_tokens": sum(record["output_tokens"] or 0 for record in self.records),
                    "total_cost_rub": total("cost_rub"), "known_cost_rub": sum(record["cost_rub"] or 0 for record in self.records)}

    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "SuiteLock", MockLock)
    monkeypatch.setattr(run, "PolzaClient", FakeClient)
    monkeypatch.setattr(run, "load_env", load_env)
    monkeypatch.setattr(run, "load_config", lambda path: {"agent": {"model": "qwen/qwen3.6-27b", "routing": {"order": ["fake-provider"]},
                                                                  "stream": True, "request_timeout_seconds": 7}, "experiment": {"seed": 2468}})
    monkeypatch.setattr(run, "verify_freeze", verify_freeze)
    monkeypatch.setattr(run, "verify_tokenizer", lambda requested_root: (root / "external/tokenizer.mock", {"mock": True}))
    # The real redactor consults POLZA_API_KEY in the process environment.
    # Fake-only payloads have no secrets, so avoid even that credential lookup.
    monkeypatch.setattr(run, "redact", lambda value: copy.deepcopy(value))
    monkeypatch.setattr(run.os, "fsync", fsync)
    monkeypatch.setattr(run, "durable", durable)
    monkeypatch.setattr(tokenizers, "Tokenizer", MockTokenizer)

    original_path_read_bytes, original_path_read_text = Path.read_bytes, Path.read_text
    original_open, original_path_open = builtins.open, Path.open

    def credential_guard(path):
        if isinstance(path, (str, bytes, Path)) and str(path).replace("\\", "/").rsplit("/", 1)[-1] == ".env":
            raise AssertionError("No real credential reads are permitted in driver tests")

    def guarded_open(path, *args, **kwargs):
        credential_guard(path)
        return original_open(path, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        credential_guard(path)
        return original_path_open(path, *args, **kwargs)

    def before_annotation_read(path):
        if Path(path) == annotation_path:
            actions = rows(state.destination() / "actions.jsonl")
            assert actions, "Offline annotations were opened before a durable action"
            assert actions[-1]["action"]["actual_response"] == actions[-1]["response"]
            state.annotation_reads += 1
            state.events.append(("annotation-read", actions[-1]["case_id"], actions[-1]["condition"]))

    def read_bytes(path):
        before_annotation_read(path)
        return original_path_read_bytes(path)

    def read_text(path, *args, **kwargs):
        before_annotation_read(path)
        return original_path_read_text(path, *args, **kwargs)

    def deny_network(*args, **kwargs):
        raise AssertionError("No network is permitted in fake-client driver tests")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    return state


def test_default_cli_prepares_without_annotations_credentials_lock_or_client(driver):
    assert run.main([]) == 0
    report = json.loads((driver.root / "reports/freeform_transfer_preparation.json").read_text())
    assert report["status"] == "prepared_no_model_calls" and report["annotations_loaded"] is False
    assert report["model_transfer_measured"] is False and report["cases"] == 2
    assert driver.annotation_reads == 0 and driver.load_env_calls == [] and driver.constructor_kwargs == []
    assert driver.lock_entries == 0 and not (driver.root / "runs").exists()


@pytest.mark.parametrize("arguments", [["--budget-rub", "0"], ["--budget-rub", "-1"], ["--budget-rub", "NaN"], ["--budget-rub", "Infinity"], ["--budget-rub", "1e9999"], ["--budget-rub", "1e-9999"]])
def test_optional_paid_budget_must_be_finite_positive_before_input_or_credentials(driver, arguments):
    (driver.root / "data/transfer/freeform_development.json").unlink()
    with pytest.raises(SystemExit) as error:
        run.main(["--live", *arguments])
    assert error.value.code == 2
    assert driver.load_env_calls == [] and driver.clients == [] and driver.annotation_reads == 0


def test_default_live_policy_uses_fixed_calls_without_a_money_or_aggregate_token_cut(driver):
    driver.cost = 20
    assert run.main(["--live", "--run-id", "fake-driver"]) == 0
    assert len(driver.dispatches) == 3 * len(driver.cases) == 6
    assert len(rows(driver.destination() / "actions.jsonl")) == 2 * len(driver.cases)
    kwargs = driver.constructor_kwargs[0]
    assert kwargs["max_cost_rub"] is None
    assert math.isinf(kwargs["max_total_tokens"]) and kwargs["max_total_tokens"] > 0
    assert kwargs["max_tokens"] == kwargs["extraction_max_tokens"] == 8192
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["usage"]["calls"] == 6
    assert manifest["usage"]["total_cost_rub"] == 120
    assert manifest["suite_budget_rub"] is None and manifest["suite_token_cap"] is None
    assert manifest["max_model_requests"] == 3 * len(driver.cases)


def test_uncapped_fixed_run_keeps_unknown_billing_unknown_without_inventing_a_money_stop(driver):
    driver.missing_cost = True
    assert run.main(["--live", "--run-id", "fake-driver"]) == 0
    assert len(driver.dispatches) == 6
    assert all(row["cost_rub"] is None for row in rows(driver.destination() / "usage.jsonl"))
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["usage"]["total_cost_rub"] is None
    assert manifest["usage"]["calls"] == 6 and manifest["suite_budget_rub"] is None


def test_paired_fake_run_interleaves_conditions_and_uses_one_writer_plus_one_planner(driver):
    assert driver.invoke() == 0
    assert [(row["scope"], row["operation"]) for row in driver.dispatches] == [
        ("driver-case-0:FACT_LEDGER", "agent"), ("driver-case-0:ADVISORY_IR", "extraction"),
        ("driver-case-0:ADVISORY_IR", "agent"), ("driver-case-1:ADVISORY_IR", "extraction"),
        ("driver-case-1:ADVISORY_IR", "agent"), ("driver-case-1:FACT_LEDGER", "agent")]
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_conditions"] == 4
    assert manifest["usage"]["calls"] == 6 and manifest["usage"]["total_cost_rub"] == 6
    assert all(row["fake_client"] for row in rows(driver.destination() / "usage.jsonl"))
    kwargs = driver.constructor_kwargs[0]
    assert kwargs["temperature"] == 0 and kwargs["seed"] == 2468 and kwargs["max_cost_rub"] == 12
    assert kwargs["max_tokens"] == kwargs["extraction_max_tokens"] == 8192
    assert kwargs["reasoning"] == {"enabled": True} and kwargs["extraction_reasoning"] == {"enabled": False}
    assert math.isinf(kwargs["max_total_tokens"]) and kwargs["max_total_tokens"] > 0
    # Both planner conditions go through the same client and operation, hence
    # share its planner settings. Only writer operation requests extraction.
    assert len(driver.clients) == 1 and sum(row["operation"] == "agent" for row in driver.dispatches) == 4


def test_call_intents_and_raw_text_are_durable_before_dispatch_and_decode(driver, monkeypatch):
    original_decode = run.decode
    decode_checks = []

    def decode(text):
        saved = rows(driver.destination() / "committed_responses.jsonl")
        assert saved and any(row.get("response_text") == text or
                             row.get("raw_response", {}).get("choices", [{}])[0].get("message", {}).get("content") == text
                             for row in saved)
        decode_checks.append(text)
        return original_decode(text)

    monkeypatch.setattr(run, "decode", decode)
    assert driver.invoke() == 0
    assert len(decode_checks) == len(driver.dispatches) == 6
    journal = rows(driver.destination() / "request_journal.jsonl")
    assert sum(row["event"] == "INTENT_ONLY_NOT_DISPATCH_PROOF" for row in journal) == 6
    assert sum(row["event"] == "RESPONSE_COMMITTED" for row in journal) == 6


def test_durable_actual_action_precedes_every_annotation_read_and_annotation_archiving(driver):
    assert driver.invoke() == 0
    assert driver.annotation_reads == 4
    actions = rows(driver.destination() / "actions.jsonl")
    assert len(actions) == 4 and all(row["action"]["status"] == "COMMITTED_UNVERIFIED" for row in actions)
    for index, event in enumerate(driver.events):
        if event[0] == "annotation-read":
            action_events = [value for value in driver.events[:index] if value[:2] == ("durable", "actions.jsonl")]
            assert action_events[-1][2]["case_id"] == event[1] and action_events[-1][2]["condition"] == event[2]
    with zipfile.ZipFile(driver.destination() / "frozen_inputs.zip") as archive:
        assert archive.namelist().count("data/transfer/freeform_annotations.json") == 1
        assert {"src/sentinel.py", "experiments/freeform_memory/sentinel.py", "data/transfer/freeform_development.json"}.issubset(archive.namelist())


def test_loader_refuses_annotations_when_no_action_is_committed(driver):
    destination = driver.destination()
    destination.mkdir(parents=True)
    with pytest.raises((FileNotFoundError, ValueError)):
        run.load_after_commit(destination, driver.pin, driver.cases[0], {"option_id": "offer-0-a"})
    assert driver.annotation_reads == 0


def test_malformed_writer_uses_exact_ledger_fallback_and_still_charges_maintenance(driver):
    driver.malformed_writer = True
    assert driver.invoke() == 0
    contexts = rows(driver.destination() / "contexts.jsonl")
    for case in driver.cases:
        ledger = next(row for row in contexts if row["case_id"] == case["case_id"] and row["condition"] == "FACT_LEDGER")
        advisory = next(row for row in contexts if row["case_id"] == case["case_id"] and row["condition"] == "ADVISORY_IR")
        assert advisory["writer_parse_error"]
        assert "fallback" in advisory["context"]["writer_status"]
        assert advisory["context"]["context"] == ledger["context"]["context"]
    usage = rows(driver.destination() / "usage.jsonl")
    assert sum(row["category"] == "maintenance" for row in usage) == 2
    assert len(driver.dispatches) == 6


def test_scorer_refusal_stops_without_erasing_or_replacing_actual_action(driver, monkeypatch):
    original_loader = run.load_after_commit

    def refuse(*args):
        grade, raw = original_loader(*args)
        return {**grade, "grading_valid": False}, raw

    monkeypatch.setattr(run, "load_after_commit", refuse)
    assert driver.invoke() == 1
    actions = rows(driver.destination() / "actions.jsonl")
    assert len(actions) == 1 and actions[0]["response"] == {"option_id": "offer-0-a"}
    assert actions[0]["action"]["status"] == "COMMITTED_UNVERIFIED"
    assert not rows(driver.destination() / "results.jsonl")
    assert len(driver.dispatches) == 1
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "grading" in manifest["error"]


def test_independent_label_disagreement_stops_without_erasing_actual_action(driver, monkeypatch):
    original_loader = run.load_after_commit

    def disagree(*args):
        grade, raw = original_loader(*args)
        return {**grade, "expected_labels_check": {"consistent": False}}, raw

    monkeypatch.setattr(run, "load_after_commit", disagree)
    assert driver.invoke() == 1
    assert len(rows(driver.destination() / "actions.jsonl")) == 1
    assert not rows(driver.destination() / "results.jsonl")
    assert len(driver.dispatches) == 1
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "labels disagree" in manifest["error"].lower()


def test_reaching_spending_threshold_halts_without_retry_or_unrecorded_dispatch(driver):
    driver.cost = 12
    assert driver.invoke() == 1
    assert len(driver.dispatches) == len(driver.clients[0].records) == 1
    assert len(rows(driver.destination() / "actions.jsonl")) == 1
    assert any(row["event"] == "REQUEST_FAILED" for row in rows(driver.destination() / "request_journal.jsonl"))
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_ambiguous_transport_failure_stops_once_and_preserves_unknown_not_zero_usage(driver):
    driver.fail_at = 1
    assert driver.invoke() == 1
    assert len(driver.dispatches) == len(driver.invocations) == 1
    failures = [row for row in rows(driver.destination() / "request_journal.jsonl") if row["event"] == "REQUEST_FAILED"]
    assert len(failures) == 1 and failures[0]["request_sent"] is True
    assert failures[0]["new_usage"][0]["cost_rub"] is None
    assert failures[0]["new_usage"][0]["total_tokens"] is None
    assert not rows(driver.destination() / "actions.jsonl") and driver.annotation_reads == 0
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["usage"]["calls"] == 1 and manifest["usage"]["total_cost_rub"] is None


def test_missing_billing_never_becomes_zero_and_prevents_another_dispatch(driver):
    driver.missing_cost = True
    assert driver.invoke() == 1
    assert len(driver.dispatches) == 1
    usage = rows(driver.destination() / "usage.jsonl")
    assert len(usage) == 1 and usage[0]["cost_rub"] is None
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["usage"]["total_cost_rub"] is None


def test_source_change_after_writer_stops_before_following_planner_dispatch(driver):
    def change_after_writer(number, scope, operation):
        if operation == "extraction":
            driver.freeze_changed = True

    driver.after_dispatch = change_after_writer
    assert driver.invoke() == 1
    assert [(row["scope"], row["operation"]) for row in driver.dispatches] == [
        ("driver-case-0:FACT_LEDGER", "agent"), ("driver-case-0:ADVISORY_IR", "extraction")]
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "freeze" in manifest["error"].lower()


def test_preexisting_run_id_cannot_repeat_a_paid_run_or_reload_credentials(driver):
    assert driver.invoke() == 0
    prior_calls, prior_loads = len(driver.dispatches), len(driver.load_env_calls)
    with pytest.raises(SystemExit) as error:
        driver.invoke()
    assert error.value.code == 2
    assert len(driver.dispatches) == prior_calls and len(driver.load_env_calls) == prior_loads
    assert len(driver.clients) == 1


def test_budget_overflow_skips_every_planner_and_creates_no_actual_actions(driver):
    for case in driver.cases:
        case["events"][1]["content"] += " padding" * 1000
    write_json(driver.root / "data/transfer/freeform_development.json", driver.cases)
    driver.pin["runtime_inputs_sha256"]["data/transfer/freeform_development.json"] = run.sha256(driver.root / "data/transfer/freeform_development.json")
    assert driver.invoke() == 0
    assert len(driver.dispatches) == 2 and all(row["operation"] == "extraction" for row in driver.dispatches)
    assert not rows(driver.destination() / "actions.jsonl") and driver.annotation_reads == 0
    assert len(rows(driver.destination() / "results.jsonl")) == 4
    assert all(row["status"] == "BUDGET_OVERFLOW" for row in rows(driver.destination() / "results.jsonl"))
    with zipfile.ZipFile(driver.destination() / "frozen_inputs.zip") as archive:
        assert "data/transfer/freeform_annotations.json" not in archive.namelist()


def test_received_malformed_envelope_is_durable_before_envelope_field_access(driver):
    driver.malformed_envelope_at = 1
    assert driver.invoke() == 1
    received = next(event[1] for event in driver.events if event[0] == "fake-received")
    committed = rows(driver.destination() / "committed_responses.jsonl")
    assert committed and any(received == value for row in committed for value in row.values())
    assert not rows(driver.destination() / "actions.jsonl") and driver.annotation_reads == 0


def test_freeze_failure_prevents_credentials_and_all_model_dispatch(driver):
    driver.freeze_changed = True
    with pytest.raises(ValueError, match="freeze"):
        driver.invoke()
    assert driver.load_env_calls == [] and driver.clients == [] and driver.dispatches == []
    assert not (driver.root / "runs").exists()


def test_annotation_hash_change_preserves_action_but_refuses_offline_grading(driver):
    def change_annotation(number, scope, operation):
        if number == 1:
            (driver.root / "data/transfer/freeform_annotations.json").write_text('{"cases":[]}', encoding="utf-8")

    driver.after_dispatch = change_annotation
    assert driver.invoke() == 1
    assert len(rows(driver.destination() / "actions.jsonl")) == 1
    assert not rows(driver.destination() / "results.jsonl")
    assert len(driver.dispatches) == 1
    manifest = json.loads((driver.destination() / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "annotation changed" in manifest["error"].lower()


@pytest.mark.parametrize("relative", ["src/sentinel.py", "experiments/freeform_memory/sentinel.py", "configs/travel.yaml", "data/transfer/freeform_development.json"])
def test_real_freeze_verifier_rejects_changed_source_and_runtime_inputs(driver, relative):
    write_json(driver.root / "reports/freeform_transfer_development_freeze.json", driver.pin)
    assert ACTUAL_VERIFY_FREEZE() == driver.pin
    changed = driver.root / relative
    changed.write_text(changed.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="source differs|input changed"):
        ACTUAL_VERIFY_FREEZE()
    assert driver.load_env_calls == [] and driver.clients == [] and driver.annotation_reads == 0
