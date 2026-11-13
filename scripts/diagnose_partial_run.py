"""Offline diagnostic of actual emitted plans in an interrupted development run."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from constraint_memory.benchmark_adapter import score_official
from constraint_memory.orchestration.logging import read_jsonl,write_json

parser=argparse.ArgumentParser()
parser.add_argument("run_directory")
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
directory=Path(args.run_directory).resolve()
manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
if manifest["config"]["experiment"]["partition"] not in {"smoke","dev"}:
    raise SystemExit("Partial-run diagnostics are development-only")
latest={}
for event in read_jsonl(directory/"submission_events.jsonl"):
    for entry in event["accepted"]["accepted"]:
        latest[(event["variant"],str(event["group_id"]),entry["traveler_index"])]=entry
output={"diagnostic_only":True,"completed_group_benchmark":False,"memory_effect_measured":False,"variants":{}}
for variant in sorted({key[0] for key in latest}):
    rows=[entry for key,entry in latest.items() if key[0]==variant]
    groups=sorted({row["group_id"] for row in rows})
    scores=score_official(root/manifest["config"]["experiment"]["upstream_root"],root/"data/hidden/memoryarena_travel.jsonl",
        rows,groups,expected_dataset_sha=manifest["pins"]["dataset_sha256"])
    known={(str(row["group_id"]),row["traveler_index"]) for row in rows}
    # Do not export future traveler annotations from the scorer response.
    output["variants"][variant]=[{"group_id":group["group_id"],"completed_traveler_metrics":[person for person in group["traveler_metrics"]
        if (str(group["group_id"]),person["traveler_index"]) in known]} for group in scores["per_group"]]
write_json(directory/"partial_score_diagnostic.json",output)
print(json.dumps(output,indent=2))
