"""Standalone tests + bounded paired development pilot, with progress and archives.

Run from the prepared workspace's Python environment. No dependency downloads,
assistant connection, or VPN controls are used. Credentials stay in ignored .env
and are loaded only by the existing model coordinator.
"""
from __future__ import annotations

import argparse
import codecs
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from archive_run_source import archive_run_source


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("E3F", "E3T", "E3I", "E3R")
ACCOUNTING_LIMITATION = ("Without a durable request-attempt journal, forced process termination can leave a paid "
                        "request absent from saved usage; recorded unknown-cost counts do not cover such attempts.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_config(root, candidate):
    if candidate not in CANDIDATES:
        raise ValueError("Unsupported candidate; select a supported frozen frontier candidate")
    import yaml
    root = Path(root)
    base = yaml.safe_load((root / "configs/travel.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load((root / f"configs/variants/{candidate}.yaml").read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        raise ValueError("Base and candidate configuration must be objects")
    experiment, memory = overlay.get("experiment"), overlay.get("memory")
    if (not isinstance(experiment, dict) or not isinstance(memory, dict) or
            experiment.get("primary_contrast") != ["E0", candidate] or
            experiment.get("candidate") != candidate or memory.get("backend") != candidate):
        raise ValueError("Candidate overlay must declare its selected E0/candidate contrast and backend")
    if candidate in {"E3I","E3R"} and experiment.get("declared_inheritance_contract") is not True:
        raise ValueError(candidate+" requires the declared inheritance contract shared by both conditions")
    if candidate=="E3R" and experiment.get("tool_city_state_annotation_normalization") is not True:
        raise ValueError("E3R requires the city tool adapter shared by both conditions")
    if candidate=="E3R" and experiment.get("tool_price_unit")!="USD":
        raise ValueError("E3R requires the declared price schema shared by both conditions")
    agent = {**base.get("agent", {}), **overlay.get("agent", {})}
    return overlay, agent


def verify_frozen(root, candidate="E3I"):
    root = Path(root).resolve()
    candidate_config(root, candidate)
    pin = json.loads((root / f"reports/{candidate}_development_source_freeze.json").read_text(encoding="utf-8"))
    if not isinstance(pin, dict):
        raise ValueError("Candidate freeze must be an object")
    legacy = candidate == "E3F" and "tests_sha256" not in pin
    if not legacy and (pin.get("candidate") != candidate or pin.get("variants") != ["E0", candidate]):
        raise ValueError("Frozen candidate identity or contrast differs from selection")
    verification = pin.get("tests", {})
    if (not isinstance(verification, dict) or type(verification.get("passed")) is not int or verification["passed"] < 1):
        raise ValueError("Candidate lacks a recorded passing test suite")
    actual = {path.relative_to(root).as_posix(): digest(path) for path in sorted((root / "src").rglob("*.py"))}
    if not actual or actual != pin.get("source_sha256"):
        raise ValueError("Source differs from the tested candidate freeze; stop before model calls")
    if not legacy:
        actual_tests = {path.relative_to(root).as_posix(): digest(path) for path in sorted((root / "tests").rglob("*.py"))}
        if not actual_tests or actual_tests != pin.get("tests_sha256"):
            raise ValueError("Tests differ from the tested candidate freeze; stop before model calls")
    files = {"run_script_sha256": "scripts/run_first_pilot.py", "overlay_sha256": f"configs/variants/{candidate}.yaml",
             "base_config_sha256": "configs/travel.yaml", "uv_lock_sha256": "uv.lock",
             "suite_script_sha256": "scripts/run_benchmark_suite.py",
             "catalog_snapshot_sha256": "reports/polza_catalog_verified_connection.json"}
    extra_inputs = {"archive_script_sha256": "scripts/archive_run_source.py",
                    "replay_script_sha256": "scripts/replay_pinned.py", "pyproject_sha256": "pyproject.toml"}
    files.update({key: path for key, path in extra_inputs.items() if not legacy or key in pin})
    for field, relative in files.items():
        if digest(root / relative) != pin.get(field):
            raise ValueError(f"Frozen input changed: {relative}; stop before model calls")
    return pin


class SuiteAlreadyRunning(RuntimeError):
    pass


class SuiteLock:
    """One OS-held lock for all candidates and run IDs in this workspace.

    The persistent file is not a PID lock; only the live OS lock grants ownership.
    Process death releases it. Child cleanup and paid accounting still require
    inspection after forced termination; this lock is not a request journal.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "tmp/benchmark_suite.lock"
        self._stream = None

    def __enter__(self):
        if not self.path.resolve().is_relative_to(self.root) or self.path.is_symlink():
            raise ValueError("Suite lock must remain in its workspace")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                if not stream.read(1):
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise SuiteAlreadyRunning("Another benchmark suite owns this workspace; no new tests or paid run started") from None
        self._stream = stream
        return self

    def __exit__(self, *_):
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def validate_run_id(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in {".", ".."}:
        raise ValueError("Run ID must be a safe single directory name")
    for relative in ("runs", "runs/provider_preflights", "reports/suite_runs"):
        if (Path(root) / relative / run_id).exists():
            raise ValueError("Run ID already exists; saved runs cannot be overwritten")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def progress(run_dir):
    rows = []
    try:
        for line in (Path(run_dir) / "usage.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "request_status" in row:
                rows.append(row)
    except FileNotFoundError:
        pass
    completed = sum(row.get("request_status") == "completed" for row in rows)
    known_cost = sum(row["cost_rub"] for row in rows if row.get("cost_rub") is not None)
    unknown = sum(row.get("cost_rub") is None for row in rows)
    return {"completed_model_calls": completed, "known_cost_rub": known_cost, "unknown_cost_requests": unknown}


def run_child(command, root, log_path, run_dir=None, heartbeat_seconds=30):
    """Observe one owned process; never restart a failed or ambiguous model job."""
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        offset = output.tell()
        child = subprocess.Popen(command, cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        last_heartbeat = time.monotonic()
        archived = False
        try:
            while True:
                with log_path.open("rb") as reader:
                    reader.seek(offset)
                    chunk = reader.read()
                    offset = reader.tell()
                if chunk:
                    print(decoder.decode(chunk), end="", flush=True)
                if run_dir is not None and not archived:
                    manifest = read_json(Path(run_dir) / "manifest.json")
                    if manifest and manifest.get("source_sha256"):
                        try:
                            archive_run_source(Path(root), Path(run_dir))
                        except json.JSONDecodeError:
                            pass  # The coordinator may be updating its manifest.
                        else:
                            archived = True
                            print("Source archived and verified.", flush=True)
                finished = child.poll()
                if finished is not None:
                    with log_path.open("rb") as reader:
                        reader.seek(offset)
                        print(decoder.decode(reader.read(), final=True), end="", flush=True)
                    return finished
                if time.monotonic() - last_heartbeat >= heartbeat_seconds:
                    if run_dir is None:
                        print("Local verification is still running...", flush=True)
                    else:
                        values = progress(run_dir)
                        print(f"Running: {values['completed_model_calls']} completed model calls; "
                              f"known run billing {values['known_cost_rub']:.4f} RUB; "
                              f"requests with unknown billing {values['unknown_cost_requests']}.", flush=True)
                    last_heartbeat = time.monotonic()
                time.sleep(0.5)
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise


def run_selected_suite(args):
    variants = ["E0", args.candidate]
    validate_run_id(ROOT, args.run_id)
    freeze = verify_frozen(ROOT, args.candidate)
    _, agent = candidate_config(ROOT, args.candidate)
    suite_dir = ROOT / "reports/suite_runs" / args.run_id
    suite_dir.mkdir(parents=True)
    run_dir = ROOT / "runs" / args.run_id
    record = {"run_id": args.run_id, "status": "running", "phase": "tests", "pid": os.getpid(),
              "candidate": args.candidate, "variants": variants, "development_groups_per_condition": 1,
              "source_freeze": freeze, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "request_attempt_journal_integrated": False, "accounting_limitation": ACCOUNTING_LIMITATION}
    status_path = suite_dir / "status.json"

    def save():
        status_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (suite_dir.parent / "latest.json").write_text(json.dumps({"run_id": args.run_id, "status_path": str(status_path)}, indent=2) + "\n", encoding="utf-8")

    save()
    print(f"Run ID: {args.run_id}\nCandidate: {args.candidate}\nLogs: {suite_dir}\nKeep this terminal open until COMPLETED or STOPPED.", flush=True)
    try:
        code = run_child([sys.executable, "-m", "pytest", "-q", "--basetemp=tmp/suite-" + args.run_id], ROOT, suite_dir / "tests.log")
        if code:
            raise RuntimeError(f"Local tests failed (exit {code}); no model experiment started")
        verify_frozen(ROOT, args.candidate)
        if args.tests_only:
            record.update(status="complete", phase="tests_only")
            save()
            print("COMPLETED: local tests only; zero model calls.", flush=True)
            return 0
        record["phase"] = "live_pilot"
        save()
        print(f"Starting one matched development group for E0 and {args.candidate}, Qwen3.6-27B via Polza.\n"
              f"Shared caps: {agent['max_completion_tokens']} completion tokens, "
              f"{agent['max_agent_visible_memory_tokens']} memory tokens, {agent['max_steps_per_traveler']} steps per arrival.\n"
              f"Each condition stops new calls after its {agent['max_cost_rub_per_group']} RUB spending threshold; one in-flight call can exceed it.\n"
              "The preflight is separately billed. VPN settings are not changed by this script.\n"
              + ACCOUNTING_LIMITATION, flush=True)
        code = run_child([sys.executable, "scripts/run_first_pilot.py", "--variants", *variants, "--variant-config",
            f"configs/variants/{args.candidate}.yaml", "--limit", "1", "--run-id", args.run_id, "--catalog-snapshot",
            "reports/polza_catalog_verified_connection.json"], ROOT, suite_dir / "pilot.log", run_dir)
        if code:
            raise RuntimeError(f"Pilot stopped (exit {code}); inspect saved failure/accounting, do not automatically retry")
        manifest = read_json(run_dir / "manifest.json")
        if not manifest or manifest.get("status") != "complete":
            raise RuntimeError("Pilot process exited without a complete run manifest")
        verify_frozen(ROOT, args.candidate)
        actual_source = manifest.get("source_sha256", {})
        if (not isinstance(actual_source, dict) or
                {key.replace("\\", "/"): value for key, value in actual_source.items()} != freeze["source_sha256"]):
            raise RuntimeError("Completed run source pins differ from the tested candidate freeze")
        if (manifest.get("variants") != variants or
                manifest.get("config", {}).get("experiment", {}).get("primary_contrast") != variants):
            raise RuntimeError("Completed run does not identify the selected frozen pair")
        archive_run_source(ROOT, run_dir)
        record["phase"] = "pinned_replay"
        save()
        code = run_child([sys.executable, "scripts/replay_pinned.py", str(run_dir)], ROOT, suite_dir / "replay.log")
        if code:
            raise RuntimeError("Pinned replay did not verify; preserve the run for inspection")
        verify_frozen(ROOT, args.candidate)
        for variant in variants:
            score = read_json(run_dir / f"official_scores_{variant}.json")
            if not score:
                raise RuntimeError(f"Missing official score for {variant}")
            print(f"{variant}: {json.dumps(score['official_metrics'])}", flush=True)
        record.update(status="complete", phase="complete", run_dir=str(run_dir),
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save()
        print(f"COMPLETED\nResults: {run_dir}\nReconnect the VPN and tell Codex: completed {args.run_id}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error),
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save()
        print(f"STOPPED: {error}\nEvidence: {suite_dir}\nReconnect the VPN and tell Codex: stopped {args.run_id}", flush=True)
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, default="E3R", help="Tested frozen candidate paired with E0")
    parser.add_argument("--run-id")
    parser.add_argument("--tests-only", action="store_true", help="Run local tests without any model requests")
    args = parser.parse_args(argv)
    args.run_id = args.run_id or "qwen36-E0-" + args.candidate + "-vpn-off-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        with SuiteLock(ROOT):
            return run_selected_suite(args)
    except (Exception, KeyboardInterrupt) as error:
        print(f"STOPPED: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
