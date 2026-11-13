"""Verify and replay one local run with its archived, hash-checked Python source.

No arbitrary archive path is accepted. Only source_archive.zip bound to a local
workspace run's manifest and sidecar can be loaded. Extraction is contained in
run/replay_source, uses no recursive deletion, and refuses unexpected/mismatched
existing files. --check-only verifies source evidence without replaying a live or
incomplete run. Source pinning is not model/weight or environment reproducibility.
Partially verified original source pins remain partial after archive checking or
replay. Actual replay of such a capture requires --allow-partial-provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from archive_run_source import (ARCHIVE_NAME, contained, file_sha256, linked, read_manifest, run_scope,
                                verify_archive)


def extract_pinned_source(workspace: Path, run_dir: Path) -> Path:
    workspace, run_dir = run_scope(workspace, run_dir)
    evidence = verify_archive(workspace, run_dir)
    manifest, original = read_manifest(run_dir, allow_redacted=evidence["original_source_pin_verification"] == "partial")
    sidecar = json.loads((run_dir / "source_archive.json").read_text(encoding="utf-8"))
    expected = sidecar["archive_members_sha256"]
    logical_destination = run_dir / "replay_source"
    if linked(logical_destination):
        raise ValueError("Replay source destination may not be a symlink")
    destination = contained(logical_destination, run_dir)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Replay source destination is not a directory")
    if destination.exists():
        for existing in destination.rglob("*"):
            contained(existing, destination)
            if linked(existing):
                raise ValueError("Replay source contains a linked path")
            if existing.is_file():
                relative = existing.relative_to(destination).as_posix()
                if relative not in expected or file_sha256(existing) != expected[relative]:
                    raise ValueError("Replay source contains unexpected or modified files")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(run_dir / ARCHIVE_NAME) as archive:
        for member, expected_sha in sorted(expected.items()):
            target = contained(destination / Path(member), destination)
            if target.exists():
                if not target.is_file() or file_sha256(target) != expected_sha:
                    raise ValueError("Existing extracted source differs from its archive")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            contained(target, destination)  # recheck resolved parents before write
            with archive.open(member) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            if file_sha256(target) != expected_sha:
                raise ValueError("Extracted source SHA-256 mismatch")
    return destination


def replay_pinned(workspace: Path, run_dir: Path, *, check_only=False, allow_partial_provenance=False) -> dict:
    workspace, run_dir = run_scope(workspace, run_dir)
    evidence = verify_archive(workspace, run_dir)
    if check_only:
        return {**evidence, "replay_executed": False, "check_only": True}
    if evidence["original_source_pin_verification"] == "partial" and not allow_partial_provenance:
        raise ValueError("Original source provenance is partial; use --check-only or explicitly acknowledge --allow-partial-provenance")
    manifest, expected = read_manifest(run_dir, allow_redacted=allow_partial_provenance)
    if manifest.get("status") not in {"complete", "failed", "blocked", "interrupted", "cancelled"}:
        raise ValueError("Run is still active; use --check-only until it is terminal")
    operations_path = run_dir / "memory_ops.jsonl"
    if not operations_path.is_file():
        raise ValueError("Memory operations are missing; no replay can be verified")
    instances = set()
    for line in operations_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            operation = json.loads(line)
            if operation.get("op") == "reset":
                variant, group = str(operation["variant"]), str(operation["group_id"])
                if not variant or any(part in variant for part in ("/", "\\", ":", "..")) or not group.isdigit():
                    raise ValueError("Recorded namespace is unsafe for snapshot paths")
                instances.add((variant, group))
    if not instances or any(not (run_dir / f"snapshots/{variant}/{group}.json").is_file() for variant, group in instances):
        raise ValueError("Every recorded namespace needs a completed snapshot before replay")
    pinned = extract_pinned_source(workspace, run_dir)
    environment = {key: value for key, value in os.environ.items()
                   if not any(secret in key.upper() for secret in ("API_KEY", "TOKEN", "PASSWORD", "SECRET"))}
    environment["PYTHONPATH"] = str(pinned / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    entry = """
import inspect,json,sys
from pathlib import Path
source_root=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(source_root))
def no_network(event,args):
    if event in {'socket.connect','socket.getaddrinfo'}:
        raise RuntimeError('Pinned memory replay forbids network access')
sys.addaudithook(no_network)
from constraint_memory.orchestration.replay import replay_run
if not Path(inspect.getfile(replay_run)).resolve().is_relative_to(source_root):
    raise RuntimeError('Replay module was not loaded from pinned source')
print(json.dumps({'pinned_replay_results':replay_run(Path(sys.argv[1]))},default=str))
"""
    process = subprocess.run([sys.executable, "-c", entry, str(run_dir), str(pinned / "src")],
                             cwd=workspace, env=environment, encoding="utf-8", capture_output=True)
    if process.returncode:
        raise RuntimeError("Pinned memory replay failed: " + process.stderr[-4000:])
    payload = json.loads(process.stdout)
    return {**evidence, "check_only": False, "replay_executed": True, "pinned_source_path": str(pinned), **payload}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Existing local run under workspace runs/")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-only", action="store_true", help="Verify the archive without extracting or replaying an incomplete run")
    parser.add_argument("--allow-partial-provenance", action="store_true",
                        help="Explicitly replay a current source capture whose missing original pins remain unverified")
    args = parser.parse_args()
    print(json.dumps(replay_pinned(args.workspace_root, args.run_dir, check_only=args.check_only,
                                  allow_partial_provenance=args.allow_partial_provenance), indent=2))


if __name__ == "__main__":
    main()
