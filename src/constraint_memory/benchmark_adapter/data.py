"""Manifest checks and one-at-a-time public request delivery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .upstream import UPSTREAM_SHA


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(root: Path) -> dict:
    path = root / "data/manifests/splits.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    checksum_path = root / "data/manifests/splits.sha256"
    if file_sha256(path) != checksum_path.read_text(encoding="utf-8").split()[0]:
        raise ValueError("Frozen splits checksum changed")
    dev = set(value["families"]["dev"])
    validation = set(value["families"]["validation"])
    holdout = set(value["families"]["holdout"])
    if dev & validation or dev & holdout or validation & holdout:
        raise ValueError("Base itinerary families cross split boundaries")
    if not set(value["partitions"]["smoke"]) <= set(value["partitions"]["dev"]):
        raise ValueError("Smoke groups must be drawn exclusively from development families")
    return value


def selected_group_ids(root: Path, partition: str) -> list[str]:
    splits = load_splits(root)
    if partition not in splits["partitions"]:
        raise ValueError(f"Unknown partition: {partition}")
    return [str(value) for value in splits["partitions"][partition]]


def load_base(root: Path, group_id: str) -> dict:
    if not str(group_id).isdigit():
        raise ValueError("Group IDs must be numeric")
    path = root / "data/visible" / str(group_id) / "base.json"
    metadata = load_splits(root)["groups"][str(group_id)]
    if file_sha256(path) != metadata["base_sha256"]:
        raise ValueError("Public base itinerary checksum changed")
    return json.loads(path.read_text(encoding="utf-8"))


def load_turn(root: Path, group_id: str, index: int) -> dict:
    if not str(group_id).isdigit() or index < 1:
        raise ValueError("Invalid group ID or traveler index")
    path = root / "data/visible" / str(group_id) / f"turn_{index:02d}.json"
    metadata = load_splits(root)["groups"][str(group_id)]
    if index > metadata["traveler_count"]:
        raise ValueError("Traveler index exceeds group length")
    if file_sha256(path) != metadata["turns"][index - 1]["sha256"]:
        raise ValueError("Current public request checksum changed")
    request = json.loads(path.read_text(encoding="utf-8"))
    assert_visible_request(request)
    return request


def assert_visible_request(request: dict) -> None:
    permitted = {"group_id", "base_template_id", "traveler_index", "name", "query"}
    if set(request) - permitted:
        raise ValueError(f"Unpermitted request fields: {sorted(set(request) - permitted)}")
    for key in ["group_id", "traveler_index", "name", "query"]:
        if key not in request:
            raise ValueError(f"Missing visible request field: {key}")
    if not isinstance(request["query"], str) or not isinstance(request["name"], str):
        raise ValueError("Current query and name must be text")
    if not isinstance(request["traveler_index"], int) or request["traveler_index"] < 1:
        raise ValueError("Traveler index must be positive")


def verify_readiness(root: Path, *, check_database: bool = True) -> dict:
    provenance = json.loads((root / "data/manifests/provenance.json").read_text(encoding="utf-8"))
    if provenance["upstream_sha"] != UPSTREAM_SHA:
        raise ValueError("Unrecognized upstream revision")
    upstream = root / "external/MemoryArena"
    for relative, expected in provenance["source_files"].items():
        if file_sha256(upstream / relative) != expected:
            raise ValueError(f"Upstream source changed: {relative}")
    splits = load_splits(root)
    if file_sha256(root / "data/manifests/splits.json") != provenance["splits_sha256"]:
        raise ValueError("Split provenance checksum mismatch")
    database_manifest = root / "data/manifests/tool_db_checksums.json"
    if file_sha256(database_manifest) != provenance["tool_db_manifest_sha256"]:
        raise ValueError("Database manifest changed")
    tools = json.loads(database_manifest.read_text(encoding="utf-8"))
    if check_database:
        if not tools["complete"]:
            raise RuntimeError("Official flight database is missing; run scripts/prepare_memoryarena.py")
        for relative, metadata in tools["files"].items():
            path = root / tools["database_path"] / relative
            if not path.exists() or path.stat().st_size != metadata["bytes"] or file_sha256(path) != metadata["sha256"]:
                raise ValueError(f"Database integrity failure: {relative}")
    return {"provenance": provenance, "splits": splits, "database": tools}


class ChronologicalAdapter:
    """Deliver one current request, requiring its actual commit before advancing.

    Construct a fresh adapter and reset memory for every group. The adapter has
    public paths and expected counts; it never reads hidden answer keys or loads
    the future prompt list into memory.
    """
    def __init__(self, root: Path, group_id: str):
        self.root = Path(root)
        self.group_id = str(group_id)
        self.metadata = load_splits(self.root)["groups"][self.group_id]
        self.base = load_base(self.root, self.group_id)
        self.next_index = 1
        self.inflight = False
        self.past_results: list[dict] = []

    def next_request(self) -> dict:
        if self.inflight:
            raise RuntimeError("Commit the current traveler before reading another request")
        if self.next_index > self.metadata["traveler_count"]:
            raise StopIteration
        request = load_turn(self.root, self.group_id, self.next_index)
        self.inflight = True
        return request

    def commit_result(self, result: dict) -> None:
        if not self.inflight:
            raise RuntimeError("No current traveler is awaiting commitment")
        if str(result.get("group_id")) != self.group_id or result.get("traveler_index") != self.next_index:
            raise ValueError("Result does not match the current group/traveler")
        self.past_results.append(json.loads(json.dumps(result)))
        self.next_index += 1
        self.inflight = False

    def iter_requests(self):
        while self.next_index <= self.metadata["traveler_count"]:
            yield self.next_request()
            if self.inflight:
                raise RuntimeError("Each yielded current request must be committed before advancing")
