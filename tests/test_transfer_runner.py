"""Independent, credential-free checks of the portable transfer diagnostic.

The tiny arithmetic/set oracle below is used only after a scripted selection.
It is never included in generated task inputs or planner prompts.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from constraint_memory.memory.binding_tables import TABLE_GUIDANCE, unpack_sections
from constraint_memory.transfer import build_memory, context_for, messages_for, evaluate_selection


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = load_script("independent_transfer_generator", "prepare_portable_transfer.py")
CASES = GENERATOR.cases()


def visible_rows(case):
    memory, step = build_memory(case)
    return [memory._source(option["event_id"], option["pointer"], step)[0] for option in case["options"]]


def independent_feasible(case, row):
    """Direct domain arithmetic, with no memory-core/evaluator invocation."""
    facts = {}
    for event in case["events"]:
        if event["event_id"] in {"initial", "revision"}:
            facts.update(event["content"])
    if case["domain"] == "scheduling":
        return row["start"] >= facts["prior.finish"] + 10 and row["end"] <= facts["deadline"]
    if case["domain"] == "procurement":
        return row["cost"] <= facts["budget"] and row["lead"] <= facts["lead.limit"]
    if case["domain"] == "deployment":
        return set(row["flags"]) >= set(facts["required.flags"]) and row["region"] in facts["allowed.regions"]
    if case["domain"] == "inventory":
        return row["available"] >= facts["reserved"] and row["zone"] == facts["required.zone"]
    raise AssertionError("Unsupported test domain")


def all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_keys(item)


def sections(context, style):
    if style == "tables":
        assert context.startswith(TABLE_GUIDANCE)
        return unpack_sections(json.loads(context[len(TABLE_GUIDANCE):]))
    if style == "json":
        return json.loads(context)
    return {name: json.loads(rows) for name, rows in (line.split(": ", 1) for line in context.splitlines())}


def test_generator_is_disclosed_deterministic_36_case_matrix():
    assert len(CASES) == 36
    assert CASES == GENERATOR.cases()
    assert len({case["case_id"] for case in CASES}) == 36
    assert {(case["domain"], case["case_id"].split("-")[1], case["format"]) for case in CASES} == {
        (domain, load, style)
        for domain in ("scheduling", "procurement", "deployment", "inventory")
        for load in ("short", "pressure", "infeasible")
        for style in ("tables", "json", "messages")
    }
    # No task labels, success labels or evaluator answer keys are stored.
    forbidden = {"expected", "expected_answer", "answer", "correct_option", "gold", "passed", "task_success", "checks"}
    assert not forbidden.intersection(all_keys(CASES))
    changed = GENERATOR.cases()
    changed[0]["events"][0]["content"]["prior.finish"] = -999
    assert GENERATOR.cases() == CASES


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_exact_observed_options_independent_postcommit_score_and_no_case_mutation(case):
    before = copy.deepcopy(case)
    rows = visible_rows(case)
    assert len(rows) == 3
    assert all(re.fullmatch(r"[0-9a-f]{20}", row["option_id"]) for row in rows)
    independently_feasible = []
    for row in rows:
        expected = independent_feasible(case, row)
        independently_feasible.append(expected)
        scored = evaluate_selection(case, {"option_id": row["option_id"]})
        assert scored["status"] == "EVALUATED"
        assert scored["passed"] is expected
        assert len(scored["checks"]) == 2
        assert all(check["task_success"] is None for check in scored["checks"])
    assert sum(independently_feasible) == (0 if "-infeasible-" in case["case_id"] else 1)
    abstention = evaluate_selection(case, {"option_id": None})
    assert abstention["status"] == "ABSTAINED"
    assert abstention["passed"] is (not any(independently_feasible))
    assert abstention["feasible_option_exists"] is any(independently_feasible)
    for condition in ("HISTORY", "CONSTRAINT"):
        ready = context_for(case, condition, len, 1000000)
        prompt = messages_for(case, ready)
        content = json.loads(prompt[1]["content"])
        assert len(prompt) == 2 and prompt[0]["role"] == "system" and prompt[1]["role"] == "user"
        assert content["current_options"] == rows
        assert content["option_property_bindings"] == case["options"]
        assert set(content) == {"request", "required_contract_refs", "current_options", "option_property_bindings", "memory"}
        assert not {"passed", "checks", "feasible_option_exists", "response_text"}.intersection(all_keys(content))
    assert case == before


@pytest.mark.parametrize("domain", ["scheduling", "procurement", "deployment", "inventory"])
@pytest.mark.parametrize("load", ["short", "pressure", "infeasible"])
def test_presentation_roundtrip_and_option_shuffle_are_correlated_formats(domain, load):
    trio = [case for case in CASES if case["case_id"].startswith(domain + "-" + load + "-")]
    restored = [sections(context_for(case, "CONSTRAINT", len, 1000000)["context"], case["format"]) for case in trio]
    assert restored[0] == restored[1] == restored[2]
    offered = [[row["option_id"] for row in visible_rows(case)] for case in trio]
    assert set(offered[0]) == set(offered[1]) == set(offered[2])
    # At least one format changes the offered order; identities never encode a position.
    assert len({tuple(order) for order in offered}) >= 2
    assert sorted(visible_rows(trio[0]), key=lambda row: row["option_id"]) == sorted(visible_rows(trio[1]), key=lambda row: row["option_id"])


@pytest.mark.parametrize("style", ["tables", "json", "messages"])
def test_context_capacity_boundary_and_lossless_frontier_overflow(style):
    case = next(case for case in CASES if case["case_id"] == "scheduling-short-" + style)
    before = copy.deepcopy(case)
    for condition in ("HISTORY", "CONSTRAINT"):
        full = context_for(case, condition, len, 1000000)
        exact = context_for(case, condition, len, len(full["context"]))
        assert exact == full
        limited = context_for(case, condition, len, len(full["context"]) - 1)
        if condition == "CONSTRAINT":
            assert limited["status"] == "BUDGET_OVERFLOW"
            assert limited["context"] == "BUDGET_OVERFLOW"
            assert limited["required_tokens"] == len(full["context"])
            assert limited["omitted_events"] == []
        else:
            assert limited["status"] == "PARTIAL_CONTEXT"
            assert limited["omitted_events"][0] == "initial"
            assert len(limited["context"]) <= len(full["context"]) - 1
    assert case == before


@pytest.mark.parametrize("domain", ["scheduling", "procurement", "deployment", "inventory"])
def test_pressure_history_loses_policy_and_revision_and_reports_that_confound(domain):
    case = next(case for case in CASES if case["case_id"] == domain + "-pressure-json")
    before = copy.deepcopy(case)
    history = context_for(case, "HISTORY", len, 4096)
    frontier = context_for(case, "CONSTRAINT", len, 4096)
    assert history["status"] == "PARTIAL_CONTEXT"
    assert {"initial", "policy", "revision"} <= set(history["omitted_events"])
    assert {"initial", "policy", "revision"}.isdisjoint(row["event_id"] for row in json.loads(history["context"]))
    assert frontier["status"] == "OK"
    assert frontier["omitted_events"] == []
    assert len(history["context"]) <= 4096 and len(frontier["context"]) <= 4096
    assert case == before


@pytest.mark.parametrize("response", [None, [], "text", 17, {}, {"option_id": None, "passed": True}, {"option_id": None, "reason": "none"}])
def test_invalid_submission_is_never_abstention_success(response):
    result = evaluate_selection(CASES[0], response)
    assert result == {"status": "INVALID_SUBMISSION", "passed": False, "checks": []}


@pytest.mark.parametrize("identity", ["not-offered", "", 0, False, ["one"], {"id": "one"}])
def test_unknown_identity_is_not_coerced_to_an_offered_option(identity):
    result = evaluate_selection(CASES[0], {"option_id": identity})
    assert result == {"status": "UNKNOWN_OPTION", "passed": False, "checks": []}


@pytest.mark.parametrize("mutation", ["expected_key", "unknown_contract", "option_id", "duplicate_id", "foreign_property", "outside_row", "future_order", "unobserved_offer"])
def test_invalid_case_evidence_rejected_before_prompt(mutation):
    case = copy.deepcopy(CASES[0])
    if mutation == "expected_key": case["expected_answer"] = "should never enter planner input"
    elif mutation == "unknown_contract": case["active_constraint_refs"].append("not-observed")
    elif mutation == "option_id": case["options"][0]["option_id"] = "different-literal"
    elif mutation == "duplicate_id": case["options"][1]["option_id"] = case["options"][0]["option_id"]
    elif mutation == "foreign_property": case["options"][0]["properties"][0]["pointer"] = case["options"][1]["properties"][0]["pointer"]
    elif mutation == "outside_row": case["options"][0]["properties"][0]["pointer"] = "/0"
    elif mutation == "future_order": case["events"][1]["step"] = 99
    elif mutation == "unobserved_offer": case["events"][-1]["kind"] = "task_contract"
    with pytest.raises(ValueError):
        context_for(case, "CONSTRAINT", len, 1000000)


@pytest.mark.parametrize("mode", ["missing", "contradictory"])
def test_missing_or_conflicting_target_cannot_produce_selected_success(mode):
    case = copy.deepcopy(next(case for case in CASES if case["case_id"] == "procurement-short-json"))
    if mode == "missing":
        for event in case["events"]:
            if event["event_id"] in {"initial", "revision"}:
                event["content"].pop("budget", None)
                event["facts"] = [fact for fact in event["facts"] if fact["ref"] != "budget"]
    else:
        case["events"].insert(3, {"event_id": "independent-conflict", "step": 2, "kind": "observation",
            "content": {"budget": 91}, "facts": [{"ref": "budget", "pointer": "/budget", "unit": "EUR"}]})
    before = copy.deepcopy(case)
    for row in visible_rows(case):
        result = evaluate_selection(case, {"option_id": row["option_id"]})
        assert not result["passed"]
        assert result["checks"][0]["status"] == "UNVERIFIED"
    abstention = evaluate_selection(case, {"option_id": None})
    assert abstention["passed"] and not abstention["feasible_option_exists"]
    assert case == before


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = load_script("independent_transfer_runner", "run_portable_transfer.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "data/transfer").mkdir(parents=True)
    (tmp_path / "data/transfer/development.json").write_text(json.dumps(CASES[:2]), encoding="utf-8")
    monkeypatch.setattr(module, "verify_tokenizer", lambda root: (tmp_path / "fake-tokenizer.json", "pinned"))
    class FakeTokenizer:
        @staticmethod
        def from_file(path):
            return SimpleNamespace(encode=lambda text: SimpleNamespace(ids=list(text)))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer))
    return module


def test_default_preparation_never_reads_credentials_initializes_client_or_scores(runner, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation attempted a credential, model, or evaluator operation")
    for name in ("load_env", "PolzaClient", "evaluate_selection", "verify_freeze"):
        monkeypatch.setattr(runner, name, forbidden)
    assert runner.main([]) == 0
    summary = json.loads((runner.ROOT / "reports/portable_transfer_preparation.json").read_text())
    assert summary["status"] == "prepared_no_model_calls" and not summary["model_transfer_measured"]
    assert summary["cases"] == 2
    assert all("context" not in value for row in summary["contexts"] for value in row["contexts"].values())
    assert not (runner.ROOT / "runs").exists()


def test_live_bad_freeze_stops_before_run_creation_credentials_or_client(runner, monkeypatch):
    calls = []
    def reject(path):
        calls.append("freeze")
        raise ValueError("deliberately changed frozen source")
    monkeypatch.setattr(runner, "verify_freeze", reject)
    monkeypatch.setattr(runner, "load_env", lambda path: calls.append("credentials"))
    monkeypatch.setattr(runner, "PolzaClient", lambda **kwargs: calls.append("client"))
    with pytest.raises(ValueError, match="changed frozen source"):
        runner._main(["--live", "--run-id", "offline-frozen-gate"])
    assert calls == ["freeze"]
    assert not (runner.ROOT / "runs").exists()


def test_verified_freeze_pins_full_source_tests_input_and_counts(runner):
    (runner.ROOT / "src").mkdir()
    (runner.ROOT / "tests").mkdir()
    (runner.ROOT / "reports").mkdir()
    for relative in ("src/core.py", "tests/example.py", "options.json"):
        (runner.ROOT / relative).write_text("offline fixture", encoding="utf-8")
    cases_path = runner.ROOT / "data/transfer/development.json"
    pin = {"cases_sha256": runner.sha256(cases_path), "source_and_tests_sha256": {
        relative: runner.sha256(runner.ROOT / relative) for relative in ("src/core.py", "tests/example.py")},
        "inputs_sha256": {"options.json": runner.sha256(runner.ROOT / "options.json")}, "tests": {"passed": 1}}
    freeze_path = runner.ROOT / "reports/portable_transfer_development_freeze.json"
    freeze_path.write_text(json.dumps(pin), encoding="utf-8")
    assert runner.verify_freeze(cases_path) == pin
    (runner.ROOT / "tests/example.py").write_text("modified after test", encoding="utf-8")
    with pytest.raises(ValueError, match="Source/tests"):
        runner.verify_freeze(cases_path)


def scripted_live(runner, monkeypatch, *, content='{"option_id":null}', dispatch_error=None):
    """A local fake client, not a model simulation or model performance result."""
    log = []
    pin = {"source_and_tests_sha256": {}, "inputs_sha256": {}}
    monkeypatch.setattr(runner, "verify_freeze", lambda path: (log.append("freeze"), copy.deepcopy(pin))[1])
    monkeypatch.setattr(runner, "load_env", lambda path: log.append("credential_stub_no_read"))
    monkeypatch.setattr(runner, "load_config", lambda path: {"agent": {
        "model": "test/exact-model", "routing": {"provider": "test-provider"},
        "reasoning": {"effort": "none"}, "stream": False, "request_timeout_seconds": 10},
        "experiment": {"seed": 4}})

    class ScriptedClient:
        def __init__(self, **settings):
            self.settings, self.records, self.scopes, self.prompts = settings, [], [], []
            self.current_scope = None

        def __enter__(self):
            log.append("client_enter")
            return self

        def __exit__(self, *args):
            log.append("client_exit")

        def verify_model(self, **kwargs):
            log.append("catalog_stub_no_network")

        def set_scope(self, *scope):
            self.current_scope = scope
            self.scopes.append(scope)

        def complete(self, messages, **kwargs):
            destination = self.settings["usage_path"].parent
            requests = [json.loads(line) for line in (destination / "requests.jsonl").read_text().splitlines()]
            assert requests[-1]["messages"] == messages
            assert "response_text" not in requests[-1]
            assert kwargs == {"json_output": True, "operation": "agent"}
            log.append("dispatch")
            self.prompts.append(copy.deepcopy(messages))
            if dispatch_error:
                raise RuntimeError(dispatch_error)
            number = len(self.records) + 1
            self.records.append({"request_id": "offline-id-" + str(number), "model": "test/exact-model",
                "provider": "test-provider", "cost_rub": 0.01, "usage_status": "known", "total_tokens": 7})
            runner.append_jsonl(self.settings["usage_path"], self.records[-1])
            log.append("returned")
            return {"id": "offline-id-" + str(number), "model": "test/exact-model",
                "choices": [{"message": {"content": content}}]}

        def get_usage_stats(self):
            return {"calls": len(self.records), "total_tokens": len(self.records) * 7, "cost_rub": len(self.records) * 0.01}

    instances = []
    def factory(**settings):
        instance = ScriptedClient(**settings)
        instances.append(instance)
        return instance
    monkeypatch.setattr(runner, "PolzaClient", factory)
    return log, instances, pin


@pytest.mark.parametrize("content,status", [
    ('{"option_id":null}', "ABSTAINED"),
    ('{"option_id":"unknown-option"}', "UNKNOWN_OPTION"),
    ('{"option_id":null,"reason":"invented"}', "INVALID_SUBMISSION"),
    ('non-JSON response', "INVALID_SUBMISSION"),
])
def test_scripted_dispatch_single_attempt_persisted_before_evaluator_and_no_score_feedback(runner, monkeypatch, content, status):
    log, instances, _ = scripted_live(runner, monkeypatch, content=content)
    destination = runner.ROOT / "runs/offline-scripted"
    original_evaluator = runner.evaluate_selection
    def after_commit(case, response):
        assert log[-1] == "returned"
        committed = [json.loads(line) for line in (destination / "committed_responses.jsonl").read_text().splitlines()]
        assert committed[-1]["response_text"] == content
        assert committed[-1]["case_id"] == case["case_id"]
        assert committed[-1]["request_id"] == instances[0].records[-1]["request_id"]
        assert committed[-1]["usage"] == instances[0].records[-1]
        log.append("postcommit_score")
        return original_evaluator(case, response)
    monkeypatch.setattr(runner, "evaluate_selection", after_commit)
    assert runner._main(["--live", "--run-id", "offline-scripted"]) == 0
    client = instances[0]
    assert len(client.prompts) == 4
    assert len(set(client.scopes)) == 4
    assert [scope[1].split(":")[-1] for scope in client.scopes] == ["HISTORY", "CONSTRAINT", "CONSTRAINT", "HISTORY"]
    assert log.count("freeze") == 6  # Initial, one per condition, final.
    assert log.count("dispatch") == log.count("postcommit_score") == 4
    assert client.settings["max_total_tokens"] == 200000 and client.settings["max_cost_rub"] == 20
    rows = [json.loads(line) for line in (destination / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(row["status"] == status and not row["passed"] for row in rows)
    assert all(len(prompt) == 2 and not {"passed", "checks", "feasible_option_exists", "response_text"}.intersection(
        all_keys(json.loads(prompt[1]["content"]))) for prompt in client.prompts)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["max_model_calls_per_case_condition"] == 1
    assert manifest["passed"] == {"HISTORY": 0, "CONSTRAINT": 0}


def test_scoring_exception_preserves_already_committed_response_and_never_retries(runner, monkeypatch):
    log, instances, _ = scripted_live(runner, monkeypatch)
    destination = runner.ROOT / "runs/offline-score-failure"
    def fail_score(case, response):
        assert (destination / "committed_responses.jsonl").exists()
        raise ValueError("offline scorer stopped after answer commit")
    monkeypatch.setattr(runner, "evaluate_selection", fail_score)
    assert runner._main(["--live", "--run-id", "offline-score-failure"]) == 1
    assert len(instances[0].prompts) == log.count("dispatch") == 1
    assert len((destination / "committed_responses.jsonl").read_text().splitlines()) == 1
    assert len((destination / "usage.jsonl").read_text().splitlines()) == 1
    assert not (destination / "results.jsonl").exists()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_conditions"] == 0
    assert not manifest["model_transfer_measured"]


def test_ambiguous_dispatch_has_one_recorded_request_and_no_automatic_retry(runner, monkeypatch):
    log, instances, _ = scripted_live(runner, monkeypatch, dispatch_error="connection lost after dispatch; accounting unknown")
    assert runner._main(["--live", "--run-id", "offline-dispatch-failure"]) == 1
    destination = runner.ROOT / "runs/offline-dispatch-failure"
    assert len(instances[0].prompts) == log.count("dispatch") == 1
    assert len((destination / "requests.jsonl").read_text().splitlines()) == 1
    assert not (destination / "committed_responses.jsonl").exists()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "accounting unknown" in manifest["error"]


def test_changed_freeze_between_conditions_stops_before_next_dispatch(runner, monkeypatch):
    log, instances, pin = scripted_live(runner, monkeypatch)
    count = 0
    def frozen(path):
        nonlocal count
        count += 1
        log.append("freeze")
        return copy.deepcopy(pin) if count < 3 else {**pin, "changed": True}
    monkeypatch.setattr(runner, "verify_freeze", frozen)
    assert runner._main(["--live", "--run-id", "offline-mutated-freeze"]) == 1
    assert log.count("dispatch") == len(instances[0].records) == 1
    manifest = json.loads((runner.ROOT / "runs/offline-mutated-freeze/manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["completed_conditions"] == 1
    assert "freeze changed" in manifest["error"]
