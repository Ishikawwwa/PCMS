"""Archive a run's exact manifest-listed Python source after checking every hash.

Only source_sha256 entries under src/ are included. Credentials, data, and config
files are excluded. The main run manifest is never modified. This preserves code
for memory replay; it does not preserve model weights, provider inference, or the
installed dependency environment. The uv.lock hash is observed at archive time.
The explicit --allow-redacted-pins mode preserves current bytes for literal
[REDACTED] pins, while declaring their original-run provenance unverified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path

MAX_MEMBERS = 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
ARCHIVE_NAME = "source_archive.zip"
SIDECAR_NAME = "source_archive.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_map_hash(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def contained(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(parent.resolve()) or resolved == parent.resolve():
        raise ValueError("Target path escapes its declared workspace/run directory")
    return resolved


def linked(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def run_scope(workspace: Path, run_dir: Path) -> tuple[Path, Path]:
    workspace = workspace.resolve(strict=True)
    runs = workspace / "runs"
    if linked(runs) or (runs.exists() and not runs.resolve().is_relative_to(workspace)):
        raise ValueError("Workspace runs directory is not contained in the workspace")
    run_dir = contained(run_dir if run_dir.is_absolute() else workspace / run_dir, runs)
    if not run_dir.is_dir() or not (run_dir / "manifest.json").is_file():
        raise ValueError("A local workspace run with manifest.json is required")
    return workspace, run_dir


def normalized_member(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("Invalid source archive member name")
    normalized = name.replace("\\", "/")
    parts = normalized.split("/")
    if parts[0] != "src" or len(parts) < 2 or any(part in ("", ".", "..") or ":" in part for part in parts):
        raise ValueError("Only safe relative source files under src/ are allowed")
    reserved = {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]}
    if any(part.endswith((".", " ")) or re.search(r'[<>:"|?*\x00-\x1f]', part)
           or part.split(".", 1)[0].rstrip(" .").casefold() in reserved for part in parts):
        raise ValueError("Source member is unsafe on Windows")
    if not normalized.endswith(".py"):
        raise ValueError("Source archive permits manifest-listed Python files only")
    return normalized


def validated_source_map(values: dict, *, allow_redacted=False) -> dict[str, str]:
    if not isinstance(values, dict) or not values or len(values) > MAX_MEMBERS:
        raise ValueError("Manifest must provide a nonempty bounded source_sha256 map")
    normalized = {}
    windows_names = set()
    for original, digest in values.items():
        member = normalized_member(original)
        if not isinstance(digest, str) or (not re.fullmatch(r"[0-9a-f]{64}", digest)
                                          and not (allow_redacted and digest == "[REDACTED]")):
            raise ValueError("Manifest contains an invalid source SHA-256")
        if member.casefold() in windows_names:
            raise ValueError("Source names collide after Windows path normalization")
        windows_names.add(member.casefold())
        normalized[member] = digest
    return normalized


def read_manifest(run_dir: Path, *, allow_redacted=False) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_dir.name:
        raise ValueError("Run identity does not match its workspace directory")
    members = validated_source_map(manifest.get("source_sha256"), allow_redacted=allow_redacted)
    return manifest, members


def missing_original_pins(source_map, observed):
    return [{"path": normalized_member(original), "manifest_path": original, "original_pin": "[REDACTED]",
             "observed_current_sha256": observed[normalized_member(original)], "original_pin_verified": False}
            for original, digest in sorted(source_map.items()) if digest == "[REDACTED]"]


def verify_archive(workspace: Path, run_dir: Path) -> dict:
    """Verify only the archive owned by this local run, never an external ZIP."""
    workspace, run_dir = run_scope(workspace, run_dir)
    if linked(run_dir / ARCHIVE_NAME) or linked(run_dir / SIDECAR_NAME):
        raise ValueError("Run source archive/sidecar may not be symlinks")
    archive_path = contained(run_dir / ARCHIVE_NAME, run_dir)
    sidecar_path = contained(run_dir / SIDECAR_NAME, run_dir)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    declared = sidecar.get("original_source_pin_verification", "full")
    if declared not in {"full", "partial"}:
        raise ValueError("Source sidecar has an invalid original-pin verification declaration")
    manifest, original = read_manifest(run_dir, allow_redacted=declared == "partial")
    expected = validated_source_map(sidecar.get("archive_members_sha256"))
    if sidecar.get("schema_version") != 1 or sidecar.get("run_id") != manifest["run_id"] or sidecar.get("archive_filename") != ARCHIVE_NAME:
        raise ValueError("Source sidecar does not identify this local run archive")
    if sidecar.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("Archive source pins differ from the run manifest")
    if sidecar.get("manifest_source_map_sha256") != source_map_hash(manifest["source_sha256"]):
        raise ValueError("Source-map fingerprint mismatch")
    if set(original) != set(expected):
        raise ValueError("Archive member names differ from the manifest")
    for member, digest in original.items():
        if digest != "[REDACTED]" and digest != expected[member]:
            raise ValueError("Archive member pins differ from available original manifest pins")
    missing = missing_original_pins(manifest["source_sha256"], expected)
    if (declared == "partial") != bool(missing):
        raise ValueError("Original-pin verification declaration contradicts the manifest")
    if sidecar.get("missing_original_pins", []) != missing:
        raise ValueError("Missing original-pin records differ from the manifest/current capture")
    if sidecar.get("observed_source_sha256", expected) != expected:
        raise ValueError("Observed source pins differ from the archived member pins")
    scope = "archive_integrity" if missing else "archive_integrity_and_original_source_pins"
    if sidecar.get("verification_scope", scope) != scope:
        raise ValueError("Source verification scope contradicts its original-pin evidence")
    if sidecar.get("uv_lock_sha256") is not None and not re.fullmatch(r"[0-9a-f]{64}", str(sidecar["uv_lock_sha256"])):
        raise ValueError("Source sidecar contains an invalid dependency-lock hash")
    if file_sha256(archive_path) != sidecar.get("archive_sha256"):
        raise ValueError("Source archive SHA-256 mismatch")
    total = 0
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected) or len(infos) > MAX_MEMBERS:
            raise ValueError("Source archive has missing, duplicate, or unlisted members")
        seen = set()
        for info in infos:
            member = normalized_member(info.filename)
            if info.filename != member or member.casefold() in seen or member not in expected:
                raise ValueError("Source archive member is noncanonical, duplicated, or unlisted")
            seen.add(member.casefold())
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 1:
                raise ValueError("Source archive must contain regular, unencrypted source files")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Unsupported source archive compression")
            total += info.file_size
            if info.file_size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("Source archive exceeds its allocation limit")
            digest = hashlib.sha256()
            size = 0
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > info.file_size or size > MAX_MEMBER_BYTES:
                        raise ValueError("Source archive member exceeded its declared size")
                    digest.update(chunk)
            if size != info.file_size or digest.hexdigest() != expected[member]:
                raise ValueError(f"Source archive member SHA-256 mismatch: {member}")
    lock_path = workspace / "uv.lock"
    current_lock = file_sha256(lock_path) if lock_path.is_file() else None
    return {"run_id": manifest["run_id"], "run_status": manifest.get("status"), "source_files": len(expected),
            "source_bytes": total, "archive_path": str(archive_path), "sidecar_path": str(sidecar_path),
            "archive_sha256": sidecar["archive_sha256"], "source_map_sha256": sidecar["manifest_source_map_sha256"],
            "uv_lock_sha256_at_archive": sidecar.get("uv_lock_sha256"),
            "uv_lock_current_matches": current_lock == sidecar.get("uv_lock_sha256") if sidecar.get("uv_lock_sha256") else None,
            "verified": "archive_integrity" if missing else True,
            "archive_integrity_verified": True, "verification_scope": scope,
            "original_source_pin_verification": declared, "original_source_verified": not bool(missing),
            "available_original_pins_verified": len(expected) - len(missing), "missing_original_pins": missing,
            "source_provenance_limitation": "Missing original digests were captured at archive time; their original-run version is not verified." if missing else None}


def archive_run_source(workspace: Path, run_dir: Path, *, allow_redacted_pins=False) -> dict:
    workspace, run_dir = run_scope(workspace, run_dir)
    manifest, original = read_manifest(run_dir, allow_redacted=allow_redacted_pins)
    expected = {}
    archive_path = contained(run_dir / ARCHIVE_NAME, run_dir)
    sidecar_path = contained(run_dir / SIDECAR_NAME, run_dir)
    if archive_path.exists() or sidecar_path.exists():
        if archive_path.exists() and sidecar_path.exists():
            return verify_archive(workspace, run_dir)
        raise ValueError("Incomplete existing source archive; refusing to overwrite evidence")
    contents = {}
    total = 0
    for member, expected_sha in sorted(original.items()):
        logical = workspace / Path(member)
        source = contained(logical, workspace)
        if linked(logical) or not source.is_file() or source.stat().st_size > MAX_MEMBER_BYTES:
            raise ValueError(f"Source member is missing, linked, or too large: {member}")
        with source.open("rb") as stream:
            body = stream.read(MAX_MEMBER_BYTES + 1)
        if len(body) > MAX_MEMBER_BYTES:
            raise ValueError(f"Source member exceeds its allocation limit: {member}")
        total += len(body)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Manifest source exceeds its total allocation limit")
        observed_sha = hashlib.sha256(body).hexdigest()
        if expected_sha != "[REDACTED]" and observed_sha != expected_sha:
            raise ValueError(f"Current source differs from frozen run manifest: {member}")
        expected[member] = observed_sha
        contents[member] = body
    # Every available original pin was verified before any artifact was created.
    # Explicitly allowed redacted pins are CURRENT captures, not recovered pins.
    # Save captured bytes, avoiding a second read racing a later source edit.
    temporary_archive = contained(run_dir / (ARCHIVE_NAME + ".partial"), run_dir)
    temporary_sidecar = contained(run_dir / (SIDECAR_NAME + ".partial"), run_dir)
    if temporary_archive.exists() or temporary_sidecar.exists():
        raise ValueError("An existing partial source archive requires inspection")
    with zipfile.ZipFile(temporary_archive, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, body in contents.items():
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, body)
    lock_path = workspace / "uv.lock"
    missing = missing_original_pins(manifest["source_sha256"], expected)
    sidecar = {"schema_version": 1, "run_id": manifest["run_id"], "archive_filename": ARCHIVE_NAME,
               "archive_sha256": file_sha256(temporary_archive), "source_sha256": manifest["source_sha256"],
               "archive_members_sha256": expected, "manifest_source_map_sha256": source_map_hash(manifest["source_sha256"]),
               "observed_source_sha256": expected, "missing_original_pins": missing,
               "original_source_pin_verification": "partial" if missing else "full",
               "verification_scope": "archive_integrity" if missing else "archive_integrity_and_original_source_pins",
               "uv_lock_sha256": file_sha256(lock_path) if lock_path.is_file() else None,
               "uv_lock_scope": "workspace lock observed at archive time; not proof of the originally installed environment",
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "limitations": "Archives listed Python code only; model weights, provider runtime, dependencies, data, config, and credentials are not archived."}
    with temporary_sidecar.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(sidecar, indent=2) + "\n")
    temporary_archive.replace(archive_path)
    temporary_sidecar.replace(sidecar_path)
    return verify_archive(workspace, run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Existing run directory inside this workspace's runs/")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-redacted-pins", action="store_true",
                        help="Capture literal [REDACTED] listed-file pins as explicitly unverified current source; verify every available original pin")
    args = parser.parse_args()
    print(json.dumps(archive_run_source(args.workspace_root, args.run_dir, allow_redacted_pins=args.allow_redacted_pins), indent=2))


if __name__ == "__main__":
    main()
