"""Standalone handoff preserves pins and one owned process, with no model calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("standalone_suite", SCRIPTS / "run_benchmark_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def prepared(root, candidate="E3I", *, legacy=False):
    overlay = {"experiment": {"primary_contrast": ["E0", candidate], "candidate": candidate,
                             "declared_inheritance_contract": candidate in {"E3I","E3R"},
                             "tool_city_state_annotation_normalization": candidate == "E3R",
                             "tool_price_unit": "USD" if candidate == "E3R" else None}, "memory": {"backend": candidate}}
    base = {"agent": {"max_completion_tokens": 16384, "max_agent_visible_memory_tokens": 4096,
                      "max_steps_per_traveler": 30, "max_cost_rub_per_group": 100}}
    files = {"src/example.py": "pass\n", "tests/test_example.py": "pass\n",
             "scripts/run_first_pilot.py": "pass\n", "scripts/run_benchmark_suite.py": "pass\n",
             "scripts/archive_run_source.py": "pass\n", "scripts/replay_pinned.py": "pass\n",
             f"configs/variants/{candidate}.yaml": json.dumps(overlay), "pyproject.toml": "tested-config",
             "configs/travel.yaml": json.dumps(base), "uv.lock": "locked", "reports/polza_catalog_verified_connection.json": "{}"}
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    pin = {"source_sha256": {"src/example.py": suite.digest(root / "src/example.py")},
           "tests_sha256": {"tests/test_example.py": suite.digest(root / "tests/test_example.py")},
           "candidate": candidate, "variants": ["E0", candidate], "tests": {"passed": 1},
           "run_script_sha256": suite.digest(root / "scripts/run_first_pilot.py"),
           "suite_script_sha256": suite.digest(root / "scripts/run_benchmark_suite.py"),
           "overlay_sha256": suite.digest(root / f"configs/variants/{candidate}.yaml"),
           "base_config_sha256": suite.digest(root / "configs/travel.yaml"),
           "uv_lock_sha256": suite.digest(root / "uv.lock"),
           "catalog_snapshot_sha256": suite.digest(root / "reports/polza_catalog_verified_connection.json"),
           "archive_script_sha256": suite.digest(root / "scripts/archive_run_source.py"),
           "replay_script_sha256": suite.digest(root / "scripts/replay_pinned.py"),
           "pyproject_sha256": suite.digest(root / "pyproject.toml")}
    if legacy:
        for key in ("tests_sha256", "candidate", "variants", "archive_script_sha256", "replay_script_sha256", "pyproject_sha256"):
            pin.pop(key)
    (root / f"reports/{candidate}_development_source_freeze.json").write_text(json.dumps(pin), encoding="utf-8")
    return pin


@pytest.mark.parametrize("changed", ["src/example.py", "tests/test_example.py", "configs/travel.yaml", "configs/variants/E3I.yaml",
                                     "scripts/run_first_pilot.py", "scripts/run_benchmark_suite.py", "scripts/archive_run_source.py",
                                     "scripts/replay_pinned.py", "pyproject.toml", "reports/polza_catalog_verified_connection.json", "uv.lock"])
def test_changed_inputs_fail_before_model_job(tmp_path, changed):
    prepared(tmp_path)
    assert suite.verify_frozen(tmp_path)
    (tmp_path / changed).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|differ|configuration"):
        suite.verify_frozen(tmp_path)


def test_extra_source_is_not_silently_accepted(tmp_path):
    prepared(tmp_path)
    (tmp_path / "src/extra.py").write_text("pass")
    with pytest.raises(ValueError, match="differs"):
        suite.verify_frozen(tmp_path)


def test_extra_test_source_is_not_silently_accepted(tmp_path):
    prepared(tmp_path)
    (tmp_path / "tests/test_extra.py").write_text("pass")
    with pytest.raises(ValueError, match="Tests differ"):
        suite.verify_frozen(tmp_path)


@pytest.mark.parametrize("candidate", ["E3F", "E3T", "E3I", "E3R"])
def test_each_named_candidate_has_its_own_overlay_and_freeze(tmp_path, candidate):
    pin = prepared(tmp_path, candidate)
    assert suite.verify_frozen(tmp_path, candidate) == pin


def test_historical_e3f_without_test_pins_remains_compatible_if_every_old_input_matches(tmp_path):
    pin = prepared(tmp_path, "E3F", legacy=True)
    original = (tmp_path / "reports/E3F_development_source_freeze.json").read_bytes()
    assert suite.verify_frozen(tmp_path, "E3F") == pin
    assert (tmp_path / "reports/E3F_development_source_freeze.json").read_bytes() == original


@pytest.mark.parametrize("field", ["tests_sha256", "candidate", "variants", "archive_script_sha256", "replay_script_sha256", "pyproject_sha256"])
def test_new_candidate_freezes_require_complete_test_and_helper_evidence(tmp_path, field):
    pin = prepared(tmp_path)
    del pin[field]
    (tmp_path / "reports/E3I_development_source_freeze.json").write_text(json.dumps(pin))
    with pytest.raises(ValueError):
        suite.verify_frozen(tmp_path)


@pytest.mark.parametrize("tests", [{}, {"passed": 0}, {"passed": False}])
def test_passing_test_record_is_required_before_any_candidate_can_start(tmp_path, tests):
    pin = prepared(tmp_path)
    pin["tests"] = tests
    (tmp_path / "reports/E3I_development_source_freeze.json").write_text(json.dumps(pin))
    with pytest.raises(ValueError, match="passing test suite"):
        suite.verify_frozen(tmp_path)


@pytest.mark.parametrize("change", ["primary_contrast", "candidate", "backend", "contract"])
def test_overlay_pair_backend_and_common_contract_must_match_selection(tmp_path, change):
    prepared(tmp_path)
    path = tmp_path / "configs/variants/E3I.yaml"
    overlay = json.loads(path.read_text())
    if change == "primary_contrast":
        overlay["experiment"][change] = ["E3T", "E3I"]
    elif change == "candidate":
        overlay["experiment"][change] = "E3T"
    elif change == "backend":
        overlay["memory"][change] = "E3T"
    else:
        overlay["experiment"]["declared_inheritance_contract"] = False
    path.write_text(json.dumps(overlay))
    with pytest.raises(ValueError, match="overlay|contract"):
        suite.verify_frozen(tmp_path)


@pytest.mark.parametrize("run_id", ["..", "../escape", "sub/path", "existing"])
def test_run_identity_rejects_unsafe_or_used_paths(tmp_path, run_id):
    (tmp_path / "runs/existing").mkdir(parents=True)
    with pytest.raises(ValueError):
        suite.validate_run_id(tmp_path, run_id)


def test_unknown_billing_is_not_displayed_as_known_zero(tmp_path):
    (tmp_path / "usage.jsonl").write_text(json.dumps({"request_status": "failed", "cost_rub": None}) + "\n")
    assert suite.progress(tmp_path) == {"completed_model_calls": 0, "known_cost_rub": 0, "unknown_cost_requests": 1}


def test_owned_process_is_archived_without_network_or_repeat(tmp_path):
    prepared(tmp_path)
    run = tmp_path / "runs/fixture"
    manifest = {"run_id": "fixture", "status": "complete", "source_sha256": {"src/example.py": suite.digest(tmp_path / "src/example.py")}}
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(manifest))
    counter = tmp_path / "child_count.txt"
    code = "from pathlib import Path; p=Path('child_count.txt'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); print('fixture finished')"
    result = suite.run_child([sys.executable, "-c", code], tmp_path, tmp_path / "process.log", run_dir=run, heartbeat_seconds=0)
    assert result == 0 and counter.read_text() == "1"
    sidecar = json.loads((run / "source_archive.json").read_text())
    assert sidecar["original_source_pin_verification"] == "full"
    assert hashlib.sha256((run / "source_archive.zip").read_bytes()).hexdigest() == sidecar["archive_sha256"]


def lock_child(root, ready):
    # A base interpreter gives an exact owned PID on Windows, avoiding venv
    # launcher redirection. The lock module's imports are standard-library only.
    code = """import json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_benchmark_suite import SuiteLock
with SuiteLock(sys.argv[2]):
 Path(sys.argv[3]).write_text(json.dumps({'pid':os.getpid()}))
 sys.stdin.readline()
"""
    return subprocess.Popen([sys._base_executable, "-B", "-c", code, str(SCRIPTS), str(root), str(ready)],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def await_child_ready(child, ready):
    deadline = time.monotonic() + 5
    while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), "Owned offline lock fixture did not acquire its lock"
    assert json.loads(ready.read_text())["pid"] == child.pid
    assert child.poll() is None


def close_child(child):
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=5)
    if child.stdin:
        child.stdin.close()
    if child.stderr:
        child.stderr.close()


def test_real_child_lock_blocks_another_run_id_before_tests_or_calls(tmp_path, monkeypatch):
    prepared(tmp_path)
    ready = tmp_path / "ready.json"
    child = lock_child(tmp_path, ready)
    try:
        await_child_ready(child, ready)
        monkeypatch.setattr(suite, "ROOT", tmp_path)
        monkeypatch.setattr(suite, "run_child", lambda *args, **kwargs: pytest.fail("Contended suite must not start a child"))
        assert suite.main(["--run-id", "different-run-id"]) == 2
        assert not (tmp_path / "reports/suite_runs/different-run-id").exists()
        with pytest.raises(suite.SuiteAlreadyRunning):
            with suite.SuiteLock(tmp_path):
                pass
        child.stdin.write(b"release\n")
        child.stdin.flush()
        assert child.wait(timeout=5) == 0
        with suite.SuiteLock(tmp_path):
            pass
    finally:
        close_child(child)


def test_process_death_releases_os_lock_without_deleting_persistent_file(tmp_path):
    ready = tmp_path / "ready.json"
    child = lock_child(tmp_path, ready)
    try:
        await_child_ready(child, ready)
        child.terminate()
        child.wait(timeout=5)
        assert (tmp_path / "tmp/benchmark_suite.lock").exists()
        with suite.SuiteLock(tmp_path):
            pass
    finally:
        close_child(child)


@pytest.mark.parametrize("candidate", ["E3T", "E3I"])
def test_selected_candidate_changes_only_explicit_frozen_pair_and_overlay(tmp_path, monkeypatch, candidate):
    pin = prepared(tmp_path, candidate)
    calls = []
    rid = "fixture-" + candidate
    def child(command, root, log_path, run_dir=None, **kwargs):
        calls.append(command)
        with pytest.raises(suite.SuiteAlreadyRunning):
            with suite.SuiteLock(tmp_path):
                pass
        if "scripts/run_first_pilot.py" in command:
            variants = command[command.index("--variants") + 1:command.index("--variant-config")]
            assert variants == ["E0", candidate]
            assert command[command.index("--variant-config") + 1] == f"configs/variants/{candidate}.yaml"
            run_dir.mkdir(parents=True)
            manifest = {"run_id": rid, "status": "complete", "source_sha256": pin["source_sha256"],
                        "variants": variants, "config": {"experiment": {"primary_contrast": variants}}}
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            for variant in variants:
                (run_dir / f"official_scores_{variant}.json").write_text(json.dumps({"official_metrics": {"SR": 0, "PS": 0, "SPS": 0}}))
        return 0
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "run_child", child)
    assert suite.main(["--candidate", candidate, "--run-id", rid]) == 0
    assert len(calls) == 3 and calls[0][1:4] == ["-m", "pytest", "-q"]
    assert "scripts/replay_pinned.py" in calls[2]
    status = json.loads((tmp_path / "reports/suite_runs" / rid / "status.json").read_text())
    assert status["status"] == "complete" and status["candidate"] == candidate
    assert status["variants"] == ["E0", candidate]
    assert status["request_attempt_journal_integrated"] is False
    assert "forced process termination" in status["accounting_limitation"]
    assert (tmp_path / "runs" / rid / "source_archive.zip").exists()


def test_new_default_candidate_and_generated_id_are_e3r_in_tests_only_mode(tmp_path, monkeypatch):
    prepared(tmp_path,"E3R")
    calls = []
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "run_child", lambda command, *args, **kwargs: calls.append(command) or 0)
    assert suite.main(["--tests-only"]) == 0
    latest = json.loads((tmp_path / "reports/suite_runs/latest.json").read_text())
    assert latest["run_id"].startswith("qwen36-E0-E3R-vpn-off-")
    assert len(calls) == 1 and "pytest" in calls[0]


def test_changed_test_after_verification_stops_before_paid_pilot(tmp_path, monkeypatch):
    prepared(tmp_path)
    calls = []
    def child(command, *args, **kwargs):
        calls.append(command)
        (tmp_path / "tests/test_example.py").write_text("changed after tests")
        return 0
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "run_child", child)
    assert suite.main(["--candidate","E3I","--run-id", "changed-test-run"]) == 2
    assert len(calls) == 1 and "pytest" in calls[0]


def test_missing_candidate_freeze_stops_before_tests_and_model_requests(tmp_path, monkeypatch):
    prepared(tmp_path)
    (tmp_path / "reports/E3I_development_source_freeze.json").unlink()
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "run_child", lambda *args, **kwargs: pytest.fail("Missing freeze must prevent child creation"))
    assert suite.main(["--candidate","E3I","--run-id", "missing-freeze-run"]) == 2


def test_failed_pilot_is_not_repeated_and_lock_is_released(tmp_path, monkeypatch):
    prepared(tmp_path)
    calls = []
    def child(command, *args, **kwargs):
        calls.append(command)
        return 2 if "scripts/run_first_pilot.py" in command else 0
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "run_child", child)
    assert suite.main(["--candidate","E3I","--run-id", "failed-pilot-run"]) == 2
    assert len(calls) == 2
    with suite.SuiteLock(tmp_path):
        pass


def test_completed_run_with_untested_source_pins_cannot_be_marked_verified(tmp_path, monkeypatch):
    prepared(tmp_path)
    calls = []
    rid = "untested-source-run"
    def child(command, root, log_path, run_dir=None, **kwargs):
        calls.append(command)
        if "scripts/run_first_pilot.py" in command:
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": rid, "status": "complete",
                "source_sha256": {"src/example.py": "0" * 64}, "variants": ["E0", "E3I"],
                "config": {"experiment": {"primary_contrast": ["E0", "E3I"]}}}))
        return 0
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "run_child", child)
    assert suite.main(["--candidate","E3I","--run-id", rid]) == 2
    assert len(calls) == 2
    status = json.loads((tmp_path / "reports/suite_runs" / rid / "status.json").read_text())
    assert status["status"] == "failed" and "source pins differ" in status["error"]
