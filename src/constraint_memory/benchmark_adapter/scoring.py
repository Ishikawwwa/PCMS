"""Invoke the pinned official evaluator in a separate hidden-data process."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def score_official(upstream_root: Path, dataset_path: Path, predictions: list[dict],
                   group_ids: list[str] | None = None, *, timeout_seconds: float = 120,
                   expected_dataset_sha: str | None = None) -> dict:
    # The planner module does not import scorer code or hidden data. This entry
    # point sends the paths solely to a distinct scoring worker after planning.
    environment = {key: value for key, value in os.environ.items() if "API_KEY" not in key.upper() and "TOKEN" not in key.upper()}
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["PYTHONIOENCODING"] = "utf-8"
    payload = {"upstream_root": str(Path(upstream_root).resolve()), "dataset_path": str(Path(dataset_path).resolve()),
               "predictions": predictions, "group_ids": group_ids, "expected_dataset_sha": expected_dataset_sha}
    process = subprocess.run([sys.executable, "-m", "constraint_memory.benchmark_adapter.scorer_worker"],
                             input=json.dumps(payload, ensure_ascii=False), capture_output=True, encoding="utf-8",
                             env=environment, timeout=timeout_seconds)
    if process.returncode:
        raise RuntimeError("Official scorer failed: " + process.stderr[-4000:])
    response = json.loads(process.stdout)
    if response.get("error"):
        raise ValueError(response["error"])
    return response
