"""Synthetic, local, network-free checks for pinned source archive tools."""
import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
archiver = importlib.import_module("archive_run_source")
replayer = importlib.import_module("replay_pinned")


def fixture_run(workspace, status="running"):
    root = workspace / "workspace"
    source = root / "src/constraint_memory/orchestration/replay.py"
    source.parent.mkdir(parents=True)
    source.write_text("def replay_run(run_dir):\n    return [{'group_id':'1','match':True}]\n", encoding="utf-8")
    for package in [root / "src/constraint_memory/__init__.py", root / "src/constraint_memory/orchestration/__init__.py"]:
        package.write_text("", encoding="utf-8")
    source_map = {p.relative_to(root).as_posix(): archiver.file_sha256(p) for p in (root / "src").rglob("*.py")}
    run = root / "runs/synthetic"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "synthetic", "status": status, "source_sha256": source_map}), encoding="utf-8")
    (root / ".env").write_text("PRIVATE_KEY=excluded", encoding="utf-8")
    (root / "uv.lock").write_text("synthetic lock", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data/hidden.json").write_text("excluded", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs/private.yaml").write_text("excluded", encoding="utf-8")
    return root, run, source


def rewrite_archive(run, payloads):
    path = run / archiver.ARCHIVE_NAME
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name, payload in payloads.items():
            stream.writestr(name, payload)
    sidecar_path = run / archiver.SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["archive_sha256"] = archiver.file_sha256(path)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")


def test_archive_contains_only_frozen_source_and_check_only_does_not_extract(tmp_path):
    root, run, source = fixture_run(tmp_path)
    manifest_before = (run / "manifest.json").read_bytes()
    result = archiver.archive_run_source(root, run)
    assert result["verified"] and result["source_files"] == 3
    assert result["uv_lock_current_matches"] is True
    assert (run / "manifest.json").read_bytes() == manifest_before
    with zipfile.ZipFile(run / archiver.ARCHIVE_NAME) as archive:
        assert sorted(archive.namelist()) == sorted(json.loads(manifest_before)["source_sha256"])
        assert all(name.startswith("src/") and name.endswith(".py") for name in archive.namelist())
    check = replayer.replay_pinned(root, run, check_only=True)
    assert check["verified"] and not check["replay_executed"]
    assert not (run / "replay_source").exists()
    with pytest.raises(ValueError, match="still active"):
        replayer.replay_pinned(root, run)


def test_source_drift_fails_before_creating_archive(tmp_path):
    root, run, source = fixture_run(tmp_path)
    source.write_text("changed after run freeze", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen"):
        archiver.archive_run_source(root, run)
    assert not (run / archiver.ARCHIVE_NAME).exists()
    assert not (run / archiver.SIDECAR_NAME).exists()


def test_redacted_manifest_hash_fails_without_inventing_an_original_pin(tmp_path):
    root, run, source = fixture_run(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"][source.relative_to(root).as_posix()] = "[REDACTED]"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid source SHA-256"):
        archiver.archive_run_source(root, run)
    assert not (run / archiver.ARCHIVE_NAME).exists()
    assert not (run / archiver.SIDECAR_NAME).exists()


def test_explicit_redacted_capture_preserves_original_map_and_never_upgrades_provenance(tmp_path):
    root, run, source = fixture_run(tmp_path, status="complete")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    name = source.relative_to(root).as_posix()
    manifest["source_sha256"][name] = "[REDACTED]"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_manifest = manifest_path.read_bytes()
    result = archiver.archive_run_source(root, run, allow_redacted_pins=True)
    assert result["verified"] == "archive_integrity"
    assert result["original_source_pin_verification"] == "partial"
    assert result["original_source_verified"] is False and result["available_original_pins_verified"] == 2
    assert result["missing_original_pins"][0]["observed_current_sha256"] == archiver.file_sha256(source)
    assert result["missing_original_pins"][0]["original_pin_verified"] is False
    assert manifest_path.read_bytes() == original_manifest
    sidecar = json.loads((run / archiver.SIDECAR_NAME).read_text())
    assert sidecar["source_sha256"] == manifest["source_sha256"]
    assert sidecar["source_sha256"][name] == "[REDACTED]"
    check = replayer.replay_pinned(root, run, check_only=True)
    assert check["verification_scope"] == "archive_integrity" and not check["original_source_verified"]
    with pytest.raises(ValueError, match="partial"):
        replayer.replay_pinned(root, run)
    (run / "memory_ops.jsonl").write_text(json.dumps({"op": "reset", "variant": "E0", "group_id": "1"}) + "\n", encoding="utf-8")
    snapshot = run / "snapshots/E0/1.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}", encoding="utf-8")
    replay = replayer.replay_pinned(root, run, allow_partial_provenance=True)
    assert replay["replay_executed"] and replay["original_source_pin_verification"] == "partial"
    assert replay["verified"] == "archive_integrity" and replay["original_source_verified"] is False


@pytest.mark.parametrize("invalid", ["UNKNOWN", "[redacted]", None])
def test_redacted_mode_accepts_only_the_literal_marker(tmp_path, invalid):
    root, run, source = fixture_run(tmp_path)
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_sha256"][source.relative_to(root).as_posix()] = invalid
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid source SHA-256"):
        archiver.archive_run_source(root, run, allow_redacted_pins=True)
    assert not (run / archiver.ARCHIVE_NAME).exists()


def test_partial_capture_still_rejects_changed_available_pin_and_tampered_map(tmp_path):
    root, run, source = fixture_run(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"][source.relative_to(root).as_posix()] = "[REDACTED]"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    known = root / "src/constraint_memory/__init__.py"
    known.write_text("changed known original", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen"):
        archiver.archive_run_source(root, run, allow_redacted_pins=True)
    assert not (run / archiver.ARCHIVE_NAME).exists()
    known.write_text("", encoding="utf-8")
    archiver.archive_run_source(root, run, allow_redacted_pins=True)
    sidecar_path = run / archiver.SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["source_sha256"][source.relative_to(root).as_posix()] = archiver.file_sha256(source)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the run manifest"):
        archiver.verify_archive(root, run)


def test_partial_provenance_cannot_be_mislabeled_full_or_hide_missing_record(tmp_path):
    root, run, source = fixture_run(tmp_path)
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_sha256"][source.relative_to(root).as_posix()] = "[REDACTED]"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    archiver.archive_run_source(root, run, allow_redacted_pins=True)
    sidecar_path = run / archiver.SIDECAR_NAME
    original = json.loads(sidecar_path.read_text())
    changed = {**original, "original_source_pin_verification": "full"}
    sidecar_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        archiver.verify_archive(root, run)
    changed = {**original, "missing_original_pins": []}
    sidecar_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing original-pin"):
        archiver.verify_archive(root, run)


def test_redacted_member_capture_hash_still_detects_archive_payload_tampering(tmp_path):
    root, run, source = fixture_run(tmp_path)
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_sha256"][source.relative_to(root).as_posix()] = "[REDACTED]"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    archiver.archive_run_source(root, run, allow_redacted_pins=True)
    with zipfile.ZipFile(run / archiver.ARCHIVE_NAME) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    payloads[source.relative_to(root).as_posix()] = b"tampered captured source"
    rewrite_archive(run, payloads)
    with pytest.raises(ValueError, match="member SHA-256"):
        archiver.verify_archive(root, run)


@pytest.mark.parametrize("bad", ["../escape.py", "src/../escape.py", "C:/escape.py", "src/C:/escape.py", "src/NUL.py", "src/x./a.py",
                                "src/CON .py", "src/bad?.py", "src/bad\nname.py"])
def test_unsafe_windows_source_paths_are_rejected(bad):
    with pytest.raises(ValueError):
        archiver.normalized_member(bad)


def test_zip_hash_and_individual_member_hash_tampering_are_rejected(tmp_path):
    root, run, source = fixture_run(tmp_path)
    archiver.archive_run_source(root, run)
    with zipfile.ZipFile(run / archiver.ARCHIVE_NAME) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    target = source.relative_to(root).as_posix()
    payloads[target] = b"changed archived source"
    rewrite_archive(run, payloads)
    with pytest.raises(ValueError, match="member SHA-256"):
        archiver.verify_archive(root, run)
    sidecar = json.loads((run / archiver.SIDECAR_NAME).read_text())
    sidecar["archive_sha256"] = "0" * 64
    (run / archiver.SIDECAR_NAME).write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="archive SHA-256"):
        archiver.verify_archive(root, run)


def test_unlisted_zip_members_and_external_run_scope_are_rejected(tmp_path):
    root, run, source = fixture_run(tmp_path)
    archiver.archive_run_source(root, run)
    with zipfile.ZipFile(run / archiver.ARCHIVE_NAME) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    payloads["../escape.py"] = b"unlisted"
    rewrite_archive(run, payloads)
    with pytest.raises(ValueError, match="unlisted"):
        archiver.verify_archive(root, run)
    with pytest.raises(ValueError, match="escapes"):
        archiver.verify_archive(root, tmp_path / "external-run")


def test_pinned_replay_uses_saved_code_after_live_source_changes(tmp_path):
    root, run, source = fixture_run(tmp_path, status="complete")
    archiver.archive_run_source(root, run)
    source.write_text("raise RuntimeError('current code must not load')\n", encoding="utf-8")
    (run / "memory_ops.jsonl").write_text(json.dumps({"op": "reset", "variant": "E0", "group_id": "1"}) + "\n", encoding="utf-8")
    snapshot = run / "snapshots/E0/1.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}", encoding="utf-8")
    result = replayer.replay_pinned(root, run)
    assert result["replay_executed"] and result["pinned_replay_results"] == [{"group_id": "1", "match": True}]
    assert not list((run / "replay_source").rglob("*.pyc"))
    replay_source = run / "replay_source" / source.relative_to(root)
    assert "current code must not load" not in replay_source.read_text()


def test_existing_replay_source_mismatch_is_refused_without_deletion(tmp_path):
    root, run, source = fixture_run(tmp_path)
    archiver.archive_run_source(root, run)
    destination = replayer.extract_pinned_source(root, run)
    marker = destination / "unlisted.py"
    marker.write_text("preserve this file", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected or modified"):
        replayer.extract_pinned_source(root, run)
    assert marker.read_text() == "preserve this file"
