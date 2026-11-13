"""Pin and materialize the public MemoryArena travel benchmark.

Downloads are streamed, verified, and recorded. The unredacted dataset is never
used by planner workers: prepare generates separate chronological visible files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

UPSTREAM_SHA = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
DATASET_SHA = "da1a37c8b19280e18627ca01cf368195a5e1d92e"
DATASET_CONTENT_SHA = "2f955d444f6f3ad3c5da2064359ab19f8fc1f90621ff9d00723a450a009c3732"
DATASET_URL = f"https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/{DATASET_SHA}/group_travel_planner/data.jsonl"
FLIGHT_URL = "https://drive.usercontent.google.com/download?id=1dNtxHFv7k0PeMHI0smZk8t-dBlWLA0Gz&export=download&confirm=t"
FLIGHT_SHA = "8dafdb0e3f8b79ce599a1e612a772865295bc226b46e5fb278368f7255b11cee"
FLIGHT_SIZE = 304807007
TOKENIZER_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TOKENIZER_URL = f"https://huggingface.co/Qwen/Qwen3.6-27B/resolve/{TOKENIZER_REVISION}/tokenizer.json"
TOKENIZER_SHA = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TOKENIZER_SIZE = 12807982


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download(url: str, path: Path, expected_sha: str | None = None,
             expected_size: int | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_sha is None or sha256(path) == expected_sha):
        if expected_size is None or path.stat().st_size == expected_size:
            return {"url": url, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        if "text/html" in response.headers.get("content-type", ""):
            raise RuntimeError(f"Download returned HTML instead of benchmark data: {url}")
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual_sha = sha256(temporary)
    size = temporary.stat().st_size
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"Checksum mismatch for {path.name}: {actual_sha}")
    if expected_size and size != expected_size:
        raise ValueError(f"Size mismatch for {path.name}: {size}")
    temporary.replace(path)
    return {"url": url, "path": str(path), "sha256": actual_sha, "bytes": size}


def itinerary_family(base: dict) -> str:
    # Names and wording vary across generated groups; the fixed daily itinerary
    # itself identifies the original base task.
    encoded = json.dumps(base["daily_plans"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prepare(root: Path, *, git: str = "git", download_flights: bool = True) -> dict:
    upstream = root / "external" / "MemoryArena"
    if not upstream.exists():
        subprocess.run([git, "clone", "--depth", "1", "--no-checkout", "https://github.com/ZexueHe/MemoryArena.git", str(upstream)], check=True)
        subprocess.run([git, "-C", str(upstream), "fetch", "--depth", "1", "origin", UPSTREAM_SHA], check=True)
        subprocess.run([git, "-C", str(upstream), "checkout", "--detach", UPSTREAM_SHA], check=True)
    revision = subprocess.check_output([git, "-c", f"safe.directory={upstream.as_posix()}", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_SHA:
        raise RuntimeError(f"Upstream revision is {revision}; required pin is {UPSTREAM_SHA}. Use a clean pinned checkout.")
    raw_path = root / "data" / "hidden" / "memoryarena_travel.jsonl"
    source = download(DATASET_URL, raw_path, DATASET_CONTENT_SHA)
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 270:
        raise ValueError(f"Expected official 270 groups, received {len(rows)}")
    families: dict[str, list[int]] = {}
    for row in rows:
        if len(row["questions"]) != len(row["answers"]):
            raise ValueError(f"Unequal questions and answers in group {row['id']}")
        family = itinerary_family(row["base_person"])
        families.setdefault(family, []).append(row["id"])
    if len(families) != 45:
        raise ValueError(f"Expected 45 base itinerary families, detected {len(families)}; inspect clustering before splitting")
    ordered = sorted(families)
    random.Random(20261013).shuffle(ordered)
    partition_families = {"dev": ordered[:8], "validation": ordered[8:15], "holdout": ordered[15:]}
    partitions = {key: sorted(gid for family in values for gid in families[family]) for key, values in partition_families.items()}
    # Cycle families before taking additional groups, preserving coverage across
    # all eight development templates. Selection uses prompts, never outcomes.
    smoke = [sorted(families[family])[0] for family in partition_families["dev"]]
    smoke += [sorted(families[family])[1] for family in partition_families["dev"][:2]]
    partitions["smoke"] = smoke
    partitions["all"] = sorted(row["id"] for row in rows)
    group_metadata = {}
    for row in rows:
        gid = row["id"]
        family = itinerary_family(row["base_person"])
        questions = row["questions"]
        base = row["base_person"]
        # One file per public request. No answer keys or future request list.
        directory = root / "data" / "visible" / str(gid)
        write_json(directory / "base.json", {"group_id": str(gid), "base_person": base, "base_template_id": family})
        requests = []
        for index, question in enumerate(questions, start=1):
            match = re.match(r"I am (\w+)\.", question)
            request = {"group_id": str(gid), "base_template_id": family, "traveler_index": index,
                       "name": match.group(1) if match else "Person", "query": question}
            write_json(directory / f"turn_{index:02d}.json", request)
            requests.append({"traveler_index": index, "path": str((directory / f"turn_{index:02d}.json").relative_to(root)).replace("\\", "/"),
                             "sha256": sha256(directory / f"turn_{index:02d}.json")})
        group_metadata[str(gid)] = {"base_template_id": family, "traveler_count": len(questions), "turns": requests,
                                   "base_path": str((directory / "base.json").relative_to(root)).replace("\\", "/"),
                                   "base_sha256": sha256(directory / "base.json")}
    manifests = root / "data" / "manifests"
    split_manifest = {"schema_version": 1, "seed": 20261013,
                      "method": "SHA256 of canonical base_person.daily_plans, shuffled family allocation before scores",
                      "family_counts": {k: len(v) for k, v in partition_families.items()},
                      "families": partition_families, "partitions": partitions,
                      "groups": group_metadata, "dataset_sha256": source["sha256"]}
    write_json(manifests / "splits.json", split_manifest)
    (manifests / "splits.sha256").write_text(sha256(manifests / "splits.json") + "  splits.json\n", encoding="utf-8")
    database = upstream / "env" / "env_systems" / "travel_planner_env" / "database"
    if download_flights:
        download(FLIGHT_URL, database / "flights" / "clean_Flights_2022.csv", FLIGHT_SHA, FLIGHT_SIZE)
    checksums = {str(p.relative_to(database)).replace("\\", "/"): {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(database.rglob("*")) if p.is_file() and p.suffix in {".csv", ".txt"}}
    required_flight = "flights/clean_Flights_2022.csv"
    tool_manifest = {"upstream_sha": UPSTREAM_SHA, "database_path": str(database.relative_to(root)).replace("\\", "/"),
                     "files": checksums, "complete": required_flight in checksums,
                     "flight_provenance": {"source": "official MemoryArena setup_travel.md Google Drive folder",
                                           "url": FLIGHT_URL, "expected_sha256": FLIGHT_SHA,
                                           "cross_check": "osunlp/TravelPlannerLeaderboard official Hugging Face space LFS SHA256",
                                           "mirror_revision": "a57daf2e45155a5378dcf75ee9fe374e671ee048"}}
    write_json(manifests / "tool_db_checksums.json", tool_manifest)
    provenance = {"upstream_repo": "https://github.com/ZexueHe/MemoryArena", "upstream_sha": UPSTREAM_SHA,
                  "dataset_config": "group_travel_planner", "dataset_revision": DATASET_SHA, "dataset_sha256": source["sha256"],
                  "dataset": {"name": "ZexueHe/memoryarena", "config": "group_travel_planner", "revision": DATASET_SHA, **source},
                  "group_count": len(rows), "base_family_count": len(families),
                  "splits_sha256": sha256(manifests / "splits.json"),
                  "tool_db_manifest_sha256": sha256(manifests / "tool_db_checksums.json"),
                  "source_files": {str(p.relative_to(upstream)).replace("\\", "/"): sha256(p) for p in [
                      *sorted((upstream / "agent").glob("*.py")),
                      *sorted((upstream / "env" / "env_systems" / "travel_planner_env").rglob("*.py"))]}}
    write_json(manifests / "provenance.json", provenance)
    tokenizer = download(TOKENIZER_URL, root / "external/qwen-tokenizer/tokenizer.json", TOKENIZER_SHA, TOKENIZER_SIZE)
    write_json(manifests / "qwen_tokenizer.json", {"model": "Qwen/Qwen3.6-27B", "revision": TOKENIZER_REVISION,
               "tokenizer_path": "external/qwen-tokenizer/tokenizer.json", "source_url": TOKENIZER_URL,
               "sha256": tokenizer["sha256"], "bytes": tokenizer["bytes"],
               "scope": "tokenizer only; does not pin provider inference revision"})
    return {"groups": len(rows), "base_families": len(families), "partition_groups": {k: len(v) for k, v in partitions.items()},
            "dataset_sha256": source["sha256"], "tool_database_ready": tool_manifest["complete"], "qwen_tokenizer_ready": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--git", default="git")
    parser.add_argument("--without-flights", action="store_true", help="Prepare metadata only; live planning will hard-fail readiness")
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), git=args.git, download_flights=not args.without_flights), indent=2))


if __name__ == "__main__":
    main()
