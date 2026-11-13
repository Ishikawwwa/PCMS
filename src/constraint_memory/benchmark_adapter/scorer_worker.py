"""Private worker: the only runtime component loading benchmark answer keys."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import types
from pathlib import Path

from .upstream import import_official, mount_upstream, TRAVEL_PACKAGE, UPSTREAM_SHA


def normalize_predictions(predictions: list[dict], expected: dict[int, int]) -> list[dict]:
    indexed: dict[tuple[int, int], dict] = {}
    for prediction in predictions:
        if "persons" in prediction:
            gid = int(prediction["id"])
            records = prediction["persons"]
        else:
            gid = int(prediction["group_id"])
            records = [{"person_idx": prediction["traveler_index"], "plan": prediction.get("plan")}]
        if gid not in expected:
            raise ValueError(f"Unexpected prediction group {gid}")
        for record in records:
            index = int(record["person_idx"])
            if index < 1 or index > expected[gid]:
                raise ValueError(f"Unexpected prediction traveler ({gid}, {index})")
            key = (gid, index)
            if key in indexed:
                raise ValueError(f"Duplicate prediction traveler {key}")
            indexed[key] = {"person_idx": index, "plan": record.get("plan")}
    # Upstream evaluate only loops submitted members. Materialize every expected
    # member, assigning null to missing/failed plans so they count as failures.
    return [{"id": gid, "persons": [indexed.get((gid, i), {"person_idx": i, "plan": None})
                                     for i in range(1, expected[gid] + 1)]} for gid in sorted(expected)]


def score(payload: dict) -> dict:
    dataset_path = Path(payload["dataset_path"])
    digest = hashlib.sha256()
    with dataset_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    dataset_sha = digest.hexdigest()
    if payload.get("expected_dataset_sha") and dataset_sha != payload["expected_dataset_sha"]:
        raise ValueError("Hidden dataset checksum differs from the frozen dataset pin")
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_by_id = {int(row["id"]): row for row in rows}
    if len(raw_by_id) != len(rows):
        raise ValueError("Duplicate hidden dataset group IDs")
    group_ids = payload.get("group_ids")
    if group_ids is None:
        group_ids = sorted({int(p.get("id", p.get("group_id"))) for p in payload["predictions"]})
    selected = [int(value) for value in group_ids]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Scoring requires nonempty unique expected group IDs")
    if any(gid not in raw_by_id for gid in selected):
        raise ValueError("Scoring requested an unknown group")
    expected = {gid: len(raw_by_id[gid]["answers"]) for gid in selected}
    submission = normalize_predictions(payload["predictions"], expected)
    gt = {}
    base_plans = {}
    for gid in selected:
        row = raw_by_id[gid]
        base_plans[gid] = row["base_person"]["daily_plans"]
        for index, plan in enumerate(row["answers"], start=1):
            gt[(gid, index)] = plan
    upstream_root = Path(payload["upstream_root"])
    mount_upstream(upstream_root)
    # eval.py imports its data loader but only evaluate() needs data. Supply an
    # intentionally disabled loader, then bind the pinned local ground truth;
    # no Hugging Face cache lookup or unpinned remote load can occur.
    loader_name = TRAVEL_PACKAGE + ".data_loader"
    loader = types.ModuleType(loader_name)

    def disabled_loader():
        raise RuntimeError("Unpinned upstream dataset loading is disabled")

    loader.load_travel_data = disabled_loader
    sys.modules[loader_name] = loader
    official = import_official(upstream_root, "eval")
    official.load_ground_truth = lambda: (gt, base_plans, expected)
    stdout_capture = io.StringIO()
    official.load_submissions = lambda unused_path: {(row["id"], person["person_idx"]): person["plan"]
                                                    for row in submission for person in row["persons"]}
    with contextlib.redirect_stdout(stdout_capture):
        official_percent = official.evaluate("<isolated pinned submission>")
    per_group = []
    for submitted in submission:
        gid = submitted["id"]
        person_metrics = []
        constraint_rates = []
        passed_persons = 0
        for member in submitted["persons"]:
            index = member["person_idx"]
            reference = gt[(gid, index)]
            passed = official.check_person_full_pass(reference, member["plan"])
            passed_persons += int(passed)
            slots = official.find_constraint_slots(gt, base_plans, gid, index)
            passed_slots = sum(int(official.check_slot_pass(reference, member["plan"], day, slot)) for day, slot in slots)
            if slots:
                constraint_rates.append(passed_slots / len(slots))
            person_metrics.append({"traveler_index": index, "full_pass": passed, "changed_slots": len(slots),
                                   "changed_slots_passed": passed_slots, "SPS": passed_slots / len(slots) if slots else None,
                                   "official_metrics": {"PS": float(passed), "SPS": passed_slots / len(slots) if slots else None},
                                   "missing_or_unparsed": not bool(member["plan"])})
        base = raw_by_id[gid]["base_person"]
        family = hashlib.sha256(json.dumps(base["daily_plans"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        metrics = {"SR": float(passed_persons == expected[gid]), "PS": passed_persons / expected[gid],
                   "SPS": sum(constraint_rates) / len(constraint_rates) if constraint_rates else None}
        per_group.append({"group_id": str(gid), "base_template_id": family, "traveler_count": expected[gid],
                          "person_count": expected[gid], "passed_person_count": passed_persons,
                          "official_metrics": metrics, **metrics, "traveler_metrics": person_metrics})
    result = {"official_metrics": {"SR": official_percent["sr"] / 100, "PS": official_percent["ps"] / 100, "SPS": official_percent["sps"] / 100},
            "official_metrics_percent": {"SR": official_percent["sr"], "PS": official_percent["ps"], "SPS": official_percent["sps"]},
            "group_mean_PS": sum(row["PS"] for row in per_group) / len(per_group),
            "per_group": per_group, "official_output": stdout_capture.getvalue(),
            "metric_metadata": {"source": "MemoryArena env/env_systems/travel_planner_env/eval.py", "upstream_sha": UPSTREAM_SHA,
                                "dataset_sha256": dataset_sha, "dataset_checksum_verified": bool(payload.get("expected_dataset_sha")),
                                "scale": "ratio", "similarity_threshold": official.SIM_TH, "slots": official.SLOTS,
                                "SR": "fraction of groups where every expected added traveler passes every evaluated slot on every gold day",
                                "PS": "official global person-weighted hard full-plan pass rate; group_mean_PS also reported separately",
                                "SPS": "mean over groups of mean over travelers with nonempty gold-versus-base changed-slot set of passed changed slots",
                                "SPS_is_paper_sPS": "unverified; deliberately retains upstream SPS name",
                                "zero_changed_slot_travelers": "excluded from SPS only",
                                "missing_prediction_policy": "materialize all expected traveler IDs with null plan, official failure",
                                "scorer_isolated": True, "planner_modules_loaded": [m for m in sys.modules if m == "agent.travel_planner"]}}
    result["official_metric_metadata"] = result["metric_metadata"]
    return result


def main():
    try:
        result = score(json.load(sys.stdin))
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
