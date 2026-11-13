"""Run the concept memory against the repository's pinned Polza model and route."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from tokenizers import Tokenizer
import yaml

from constraint_memory.config import verify_tokenizer
from constraint_memory.secrets import load_env, redact

from .lab_trace import LabPolzaClient, LabStopped, LabTrace, atomic_json
from .live_cases import SHOPPING, SHOPPING_QUERY, TRAVEL, TRAVEL_OFFERS, TRAVEL_QUERY
from .live_memory import ConceptMemory, NULLABLE_TEXT, object_schema


ROOT = Path(__file__).resolve().parents[2]
ACTOR = """Choose an offered option that satisfies the user request and all applicable evidence.
Consider only confirmed obligations. Proposals and considered options do not establish current state.
Use only the evidence and offers supplied. Return option_id as null if no option can be verified."""
ACTOR_SCHEMA = object_schema(option_id=NULLABLE_TEXT)
MODES = ("subspaces", "subspaces+table-priority", "hybrid", "flat-table")


def write_json(path, payload):
    atomic_json(path, payload)


def append_json(path, payload):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def polza_json(client, out, case, operation, system, data, schema):
    client.set_scope(case, operation)
    system += "\nReply with one JSON object using this schema: " + json.dumps(schema, separators=(",", ":"))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]
    cap = {"act": 4096, "read": 512}.get(operation, 2048)
    response = client.complete(messages, operation="agent" if operation == "act" else "extraction",
                               max_tokens=cap, json_output=True)
    choice = response["choices"][0]
    content = choice["message"].get("content")
    append_json(out / "model_calls.jsonl", {"case": case, "operation": operation,
                "messages": messages, "content": content, "request_id": response.get("id")})
    if choice.get("finish_reason") == "length":
        raise ValueError(f"{operation} exceeded its {cap}-token output limit")
    return json.loads(content)


def run_case(case, client, out, count, token_limit, shopping_query, trace):
    def complete(operation, system, data, schema):
        return polza_json(client, out, case, operation, system, data, schema)

    trace.emit("run", "Development case started", case=case)
    memory = ConceptMemory(out / f"{case}.jsonl", complete, count, trace)
    events = SHOPPING if case == "shopping" else TRAVEL
    for event in events:
        memory.observe(event)
    checkpoints = (("task24", shopping_query, 5, None),) if case == "shopping" else (
        ("before", TRAVEL_QUERY, 4, "bus"), ("after", TRAVEL_QUERY, 5, "rail"),
    )
    for checkpoint, query, through, expected in checkpoints:
        snapshot, concepts, scope = memory.route(query, through=through)
        for mode in MODES:
            row = memory.render(snapshot, concepts, scope, mode=mode, token_limit=token_limit)
            result = {"case": case, "checkpoint": checkpoint, "mode": mode,
                      "concepts": concepts, "table": scope.table, "rows": scope.required_rows, **row}
            if case == "shopping":
                result["predecessor_present"] = "B071S91WRZ" in row["context"]
                result["remaining_present"] = "Remaining $34.74." in row["context"]
            else:
                actor_input = {"request": query, "memory": row["context"], "offers": TRAVEL_OFFERS}
                action = complete("act", ACTOR, actor_input, ACTOR_SCHEMA)["option_id"]
                result.update(action=action, expected=expected, passed=action == expected)
            trace.before("Record development result", category="score", case=case, mode=mode, checkpoint=checkpoint)
            append_json(out / "results.jsonl", result)
            trace.emit("score", "Development result", **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=("all", "shopping", "travel"), default="all")
    parser.add_argument("--memory-tokens", type=int, default=192)
    parser.add_argument("--shopping-release", type=Path, help="Pinned official bundled_shopping JSONL")
    parser.add_argument("--lab-control", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.memory_tokens <= 0:
        parser.error("--memory-tokens must be positive")
    shopping_query = SHOPPING_QUERY
    shopping_release_sha = None
    if args.shopping_release:
        release = args.shopping_release.read_bytes()
        shopping_release_sha = sha256(release).hexdigest()
        if shopping_release_sha != "4411a2da528a33dc6aca519b49cc225895363f18b2d19b191fddb501200134ef":
            parser.error("Unexpected shopping dataset revision")
        task = next(row for row in map(json.loads, release.splitlines()) if str(row["id"]) == "24")
        shopping_query = task["questions"][3]
    args.out.mkdir(parents=True, exist_ok=False)
    load_env(ROOT / ".env")
    config = yaml.safe_load((ROOT / "configs/travel.yaml").read_text(encoding="utf-8"))
    agent = config["agent"]
    tokenizer_path, tokenizer_manifest = verify_tokenizer(ROOT)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count(value):
        return len(tokenizer.encode(value, add_special_tokens=False).ids)

    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    work = {"write": 0, "read": 0, "act": 0}
    if args.case in {"all", "shopping"}:
        work.update(write=len(SHOPPING), read=1)
    if args.case in {"all", "travel"}:
        work.update(write=work["write"] + len(TRAVEL), read=work["read"] + 2, act=2 * len(MODES))
    manifest = {"model": agent["model"], "route": agent["routing"], "seed": config["experiment"]["seed"],
                "memory_tokens": args.memory_tokens, "tokenizer_sha256": tokenizer_manifest["sha256"],
                "shopping_release_sha256": shopping_release_sha,
                "case": args.case, "planned_model_calls": work,
                "source_sha256": {file.name: sha256(file.read_bytes()).hexdigest()
                                                     for file in source_paths},
                "status": "running"}
    with ZipFile(args.out / "source.zip", "x", ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, path.relative_to(ROOT))
    write_json(args.out / "manifest.json", manifest)
    trace = LabTrace(args.out, args.lab_control)
    trace.emit("run", "Development run ready", manifest=manifest)
    try:
        with LabPolzaClient(trace=trace, model=agent["model"], base_url=agent["base_url"], provider=agent["routing"],
                         seed=manifest["seed"], temperature=agent["temperature"], max_tokens=4096,
                         extraction_max_tokens=2048, reasoning=agent["reasoning"],
                         extraction_reasoning=agent["extraction_reasoning"], stream=agent["stream"],
                         timeout=agent["request_timeout_seconds"], retries=0, max_total_tokens=math.inf,
                         max_cost_rub=None, usage_path=args.out / "usage.jsonl") as client:
            for case in ("shopping", "travel") if args.case == "all" else (args.case,):
                run_case(case, client, args.out, count, args.memory_tokens, shopping_query, trace)
        manifest["status"] = "complete"
    except LabStopped:
        manifest["status"] = "stopped"
    except Exception as error:
        manifest.update(status="failed", error=redact(str(error)))
        raise SystemExit(manifest["error"])
    finally:
        write_json(args.out / "manifest.json", manifest)
        trace.emit("run", "Development run ended", status=manifest["status"], error=manifest.get("error"))


if __name__ == "__main__":
    main()
