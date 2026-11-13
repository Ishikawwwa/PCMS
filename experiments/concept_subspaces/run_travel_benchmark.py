"""Run the full pinned travel release with independent, inspectable paired groups."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from constraint_memory.secrets import redact

from .lab_trace import LabTrace, read_json, atomic_json
from .run_live import ROOT


ARMS = ("mage", "pcms")
RETRYABLE = ("HTTP 429", "HTTP 502", "HTTP 503", "HTTP 504", "network request failed", "stream failed")
REPAIR_CHECK_GROUPS = (2, 4, 6, 8, 10, 12, 14)


def usage(folder, arm):
    counts = dict(requests=0, failed=0, tokens=0, cost_rub=0.0, unknown_tokens=0, unknown_cost=0)
    for path in folder.glob(f"*/{arm}/usage.jsonl"):
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            counts["requests"] += 1
            counts["failed"] += row.get("request_status") != "completed"
            for key, field, missing in (("tokens", "total_tokens", "unknown_tokens"),
                                        ("cost_rub", "cost_rub", "unknown_cost")):
                if isinstance(row.get(field), (int, float)):
                    counts[key] += row[field]
                else:
                    counts[missing] += 1
    return counts


def summarize(rows, partitions, arms=ARMS):
    output = {}
    for partition in ("all", "holdout", "validation", "dev"):
        groups = [rows[group] for group in partitions[partition]]
        paired = [row for row in groups if row["status"] == "complete"]
        output[partition] = {
            "planned": len(groups), "paired": len(paired),
            "failed": sum(row["status"] == "failed" for row in groups),
            "arms": {arm: {
                "groups_passed": sum(row[arm]["task_success"] for row in paired),
                "travelers_passed": sum(sum(row[arm]["sub_successes"]) for row in paired),
                "travelers_scored": sum(row["travelers"] for row in paired),
                "average_group_progress": (sum(sum(row[arm]["sub_successes"]) / row["travelers"]
                                               for row in paired) / len(paired)) if paired else None,
                **{key: sum(row.get(arm, {}).get("usage", {}).get(key, 0) for row in groups)
                   for key in ("requests", "failed", "tokens", "cost_rub", "unknown_tokens", "unknown_cost")},
            } for arm in arms},
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "travel-release", "train-csv", "lab-control"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--memory-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--actor-thinking", action="store_true")
    parser.add_argument("--pcms-only", action="store_true")
    parser.add_argument("--verify-repair-first", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Workers must be between 1 and 16")
    arms = ("pcms",) if args.pcms_only else ARMS
    split = read_json(ROOT / "data/manifests/splits.json")
    partitions = split["partitions"]
    groups = partitions["all"]
    rows = {group: {"group": group, "travelers": split["groups"][str(group)]["traveler_count"],
                    "status": "queued", "attempt": 0} for group in groups}
    for partition in ("dev", "validation", "holdout"):
        for group in partitions[partition]:
            rows[group]["partition"] = partition
    # Alternate arm order across groups to reduce time-of-day effects.
    check_groups = REPAIR_CHECK_GROUPS if args.verify_repair_first else ()
    queue = [*check_groups, *(group for group in groups if group not in check_groups)]
    active = {}
    ready_at = {}
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "benchmark": "travel", "partition": "all", "groups": len(groups),
        "travelers": sum(row["travelers"] for row in rows.values()), "workers": args.workers,
        "conditions": arms, "actor_thinking": args.actor_thinking, "max_steps": args.max_steps,
        "memory_tokens": args.memory_tokens, "model": "qwen/qwen3.6-27b",
        "repair_check_groups": check_groups,
        "route": {"only": ["chutes/fp8"], "allow_fallbacks": False},
        "started": datetime.now(timezone.utc).isoformat(),
        "pilot_seconds_per_traveler": (14 * 60 + 20 + (0 if args.pcms_only else 49 * 60 + 48)) / 5,
    }
    atomic_json(args.out / "manifest.json", manifest)
    trace = LabTrace(args.out)
    trace.emit("run", "Full travel benchmark started", **manifest)

    def publish():
        atomic_json(args.out / "benchmark.json", {"groups": list(rows.values()), "splits": summarize(rows, partitions, arms)})

    def launch(group):
        row = rows[group]
        row["attempt"] += 1
        row["status"] = "running"
        folder = args.out / f"group-{group:03d}" / f"attempt-{row['attempt']}"
        folder.parent.mkdir(exist_ok=True)
        checkpoint = folder.with_suffix(".state.json")
        command = [sys.executable, "-X", "utf8", "-m", "experiments.concept_subspaces.run_mage_travel",
                   "--partition", "all", "--group", str(group), "--out", str(folder),
                   "--travel-release", str(args.travel_release), "--train-csv", str(args.train_csv),
                   "--max-steps", str(args.max_steps), "--memory-tokens", str(args.memory_tokens),
                   "--actor-thinking" if args.actor_thinking else "--no-actor-thinking",
                   "--first-arm", ARMS[(group - 1) % 2], "--lab-control", str(args.lab_control),
                   "--lab-state", str(checkpoint)]
        if args.pcms_only:
            command.append("--pcms-only")
        with folder.with_suffix(".log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        active[group] = (process, folder, checkpoint)
        row["path"] = folder.relative_to(args.out).as_posix()
        trace.emit("run", "Travel group started", group=group, attempt=row["attempt"])

    def collect(group, folder, mode):
        row = rows[group]
        info = read_json(folder / "manifest.json", {})
        status = info.get("status")
        row.pop("error", None)
        for arm in arms:
            result = read_json(folder / arm / "result.json", {})
            row[arm] = {key: result[key] for key in ("task_success", "sub_successes") if key in result}
            row[arm]["usage"] = usage(folder.parent, arm)
        if status == "complete" and all(len(row[arm].get("sub_successes", [])) == row["travelers"] for arm in arms):
            row["status"] = "complete"
        elif mode in {"stop", "terminate"} or status == "stopped":
            row["status"] = "stopped"
        else:
            row["status"] = "failed"
            row["error"] = info.get("error") or "Worker exited without a complete result"
            if row["attempt"] == 1 and any(fragment in row["error"] for fragment in RETRYABLE):
                row["status"] = "queued"
                queue.append(group)
                ready_at[group] = time.monotonic() + 60
        trace.emit("run", "Travel group ended", **row)

    state_path = args.lab_control.with_suffix(".state.json")
    publish()
    mode = "run"
    try:
        while queue or active:
            command = read_json(args.lab_control, {"mode": "run", "revision": 0})
            mode = command["mode"]
            changed = False
            for group, (process, folder, _) in list(active.items()):
                if mode == "terminate" and process.poll() is None:
                    process.terminate()
                    process.wait()
                if process.poll() is not None:
                    collect(group, folder, mode)
                    del active[group]
                    changed = True
            fatal = next((row["error"] for row in rows.values() if row["status"] == "failed"
                          and any(fragment in row["error"] for fragment in
                                  ("HTTP 401", "HTTP 402", "POLZA_API_KEY is missing", "Unexpected travel dataset revision",
                                   "Unexpected MAGE author revision"))), None)
            if fatal:
                manifest.update(status="failed", error=fatal)
                break
            repair_pending = any(rows[group]["status"] != "complete" for group in check_groups)
            repair_failed = next((rows[group] for group in check_groups if rows[group]["status"] == "failed"), None)
            if repair_failed:
                manifest.update(status="failed", error=f"Repair check group {repair_failed['group']}: {repair_failed['error']}")
                break
            if mode == "run":
                eligible = [group for group in queue if (not repair_pending or group in check_groups)
                            and ready_at.get(group, 0) <= time.monotonic()]
                for group in eligible[:args.workers - len(active)]:
                    queue.remove(group)
                    launch(group)
                    changed = True
            elif mode in {"stop", "terminate"} and not active:
                break
            if changed:
                publish()
            states = [read_json(checkpoint, {}) for _, _, checkpoint in active.values()]
            paused = sum(state.get("status") == "paused" and state.get("revision") == command["revision"] for state in states)
            status = "running" if mode == "run" else "paused" if paused == len(active) else "pause_requested"
            atomic_json(state_path, {"revision": command["revision"], "status": status,
                                     "operation": f"{paused} of {len(active)} active groups paused"})
            time.sleep(1)
        if manifest["status"] == "running":
            manifest["status"] = {"stop": "stopped", "terminate": "terminated"}.get(
                mode, "failed" if any(row["status"] == "failed" for row in rows.values()) else "complete")
    except BaseException as error:
        manifest.update(status="failed", error=redact(str(error)))
        raise
    finally:
        for group, (process, folder, _) in active.items():
            if process.poll() is None:
                process.terminate()
                process.wait()
            collect(group, folder, "terminate")
        publish()
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.out / "manifest.json", manifest)
        trace.emit("run", "Full travel benchmark ended", status=manifest["status"])


if __name__ == "__main__":
    main()
