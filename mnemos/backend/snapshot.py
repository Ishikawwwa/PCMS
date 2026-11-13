"""Portable source capture; credentials and state directories are excluded."""
import gzip
import hashlib
import json
from pathlib import Path

from .memoryarena import ROOT
from .store import encoded, now


def archive_source(store, run_id, suffix=""):
    files = {}
    paths = list((ROOT / "mnemos/backend").glob("*.py")) + list((ROOT / "src/constraint_memory/benchmark_adapter").glob("*.py")) + [ROOT / "src/constraint_memory" / name for name in ("polza.py", "streaming.py", "secrets.py", "config.py")]
    for path in paths:
        files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8")
    bundle = {"captured_at": now(), "timing": "Before execution job dispatch", "files": files, "sha256": {p: hashlib.sha256(text.encode()).hexdigest() for p, text in files.items()}}
    folder = Path(store.path).parent / "sources"
    folder.mkdir(exist_ok=True)
    archive = folder / (run_id + ("-" + suffix if suffix else "") + ".json.gz")
    archive.write_bytes(gzip.compress(encoded(bundle).encode()))
    with store.connection() as db:
        store.event(db, run_id, "source.archived", {"archive_path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "timing": bundle["timing"], "file_count": len(files)})


def read_archive(store, run_id):
    store.run(run_id)
    path = Path(store.path).parent / "sources" / (run_id + ".json.gz")
    if not path.exists():
        return None
    bundle = json.loads(gzip.decompress(path.read_bytes()))
    bundle["continuations"] = [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(path.parent.glob(run_id + "-*.json.gz"))]
    return bundle
