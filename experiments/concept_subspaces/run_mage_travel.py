"""Compare released MAGE and standalone PCMS on the same travel environment."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from tokenizers import Tokenizer
import yaml

from constraint_memory.config import verify_tokenizer
from constraint_memory.secrets import load_env, redact

from .lab_trace import LabPolzaClient, LabStopped, LabTrace
from .pcms_travel import run_pcms
from .run_live import ROOT, append_json, write_json


MAGE_COMMIT = "ace693139dc43e8bdb3076da4127483b41a5060f"


class MagePolza:
    """Preserve author prompt and per-action reasoning switches on Polza."""

    def __init__(self, client, directory, condition, actor_thinking):
        self.client = client
        self.directory = directory
        self.condition = condition
        self.enable_thinking = actor_thinking
        self.turn = 0

    def run(self, prompt, system_prompt=None):
        messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
        self.client.set_scope(self.condition, self.turn)
        reasoning = self.client.reasoning
        self.client.reasoning = {"enabled": self.enable_thinking}
        try:
            response = self.client.complete(messages, operation="agent")
        finally:
            self.client.reasoning = reasoning
        action = response["choices"][0]["message"].get("content")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Polza returned no MAGE action or verdict")
        append_json(self.directory / "author_calls.jsonl", {
            "turn": self.turn, "request_id": response.get("id"), "messages": messages, "response": action,
        })
        return action.strip()


def run_mage(module, author, runtime, sample, llm, directory, max_steps, trace):
    condition = "mage"
    base = sample["base_person"]
    state = module.TravelBundleSharedState(sample_id=sample["id"], traveler_queries=sample["questions"],
                                           base_traveler_query=base["query"], base_traveler_plan=base["daily_plans"])

    class InstrumentedAgent(author):
        def build_task(self, *args, **kwargs):
            llm.turn = self.turn_index
            task = super().build_task(*args, **kwargs)
            trace.emit("run", "Traveler task prepared", condition=condition, turn=self.turn_index,
                       task=task, active=self.shared_state.get_active_results())
            return task

        def _trace_action(self, action, execute):
            proposed = trace.emit("action", action.name, condition=condition, turn=self.turn_index, action=action)
            trace.before(f"Execute {action.name}", category="tool", action=proposed, condition=condition, turn=self.turn_index)
            observation = execute()
            trace.emit("observation", f"{action.name} observation", action=proposed, condition=condition,
                       turn=self.turn_index, observation=observation, environment=self._get_env_snapshot())
            return observation

        def forward(self, task, agent_act):
            parent = super().forward
            return self._trace_action(agent_act, lambda: parent(task, agent_act))

        def _do_store(self, task, action):
            parent = super()._do_store
            return self._trace_action(action, lambda: parent(task, action))

        def _do_backtrace(self, task, action):
            parent = super()._do_backtrace
            return self._trace_action(action, lambda: parent(task, action))

        def _on_backtrace_env_restore(self, env_snapshot):
            super()._on_backtrace_env_restore(env_snapshot)
            trace.emit("observation", "Environment restored", condition=condition, turn=self.turn_index,
                       environment=self._get_env_snapshot())

        def __st_memorize__(self, task, action, observation):
            super().__st_memorize__(task, action, observation)
            tree = self.exploration_tree
            nodes = {key: {"node_id": value.node_id, "parent_id": value.parent_id,
                           "children_ids": value.children_ids} for key, value in tree.nodes.items()}
            trace.emit("state", "MAGE execution state", condition=condition, turn=self.turn_index,
                       action=action, observation=observation, active_node=tree.current_node_id,
                       active_summary=tree.current_summary_id, nodes=nodes, summaries=tree.summaries,
                       stored_context=self._stored_context, raw_context=self._raw_context)

    # The author class constructs replay agents through this module symbol.
    module.TravelPlanMAGEAgent = InstrumentedAgent
    try:
        for index, query in enumerate(sample["questions"]):
            llm.turn = index
            agent = InstrumentedAgent(llm=llm, runtime=runtime, turn_index=index, traveler_query=query,
                             shared_state=state, max_steps=max_steps, PROMPT_DEBUG_FLAG=False,
                             log_file=str(directory / f"traveler-{index}.log"),
                             max_verify_retries=1, max_check_retries=1, max_backtrace=5)
            agent.logger.FLAG_PRINT = False
            agent(agent.build_task(query_index=runtime.query_id))
            append_json(directory / "progress.jsonl", {"traveler": index, "active": state.get_active_results()})
            trace.emit("run", "Traveler turn ended", condition=condition, turn=index,
                       active=state.get_active_results())
    finally:
        module.TravelPlanMAGEAgent = author
    return state.get_active_results()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--travel-release", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--turns", type=int, help="Limit to a development prefix")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--memory-tokens", type=int, default=256)
    parser.add_argument("--first-arm", choices=("mage", "pcms"), default="mage")
    parser.add_argument("--pcms-only", action="store_true")
    parser.add_argument("--actor-thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="Disable in both arms for a faster development diagnostic")
    parser.add_argument("--lab-control", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--lab-state", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--partition", choices=("dev", "all"), default="dev")
    args = parser.parse_args()
    if min(args.max_steps, args.memory_tokens) <= 0 or (args.turns is not None and args.turns <= 0):
        parser.error("Step, turn and memory limits must be positive")

    source = ROOT / "external/MAGE"
    head = subprocess.check_output(["git", "-C", str(source), "-c", f"safe.directory={source.as_posix()}",
                                    "rev-parse", "HEAD"], text=True).strip()
    if head != MAGE_COMMIT:
        parser.error("Unexpected MAGE author revision")
    sys.path.extend(map(str, [source, source / "benchmark/travelplan", ROOT / ".venv-mage-author/Lib/site-packages"]))
    import travelplan_mage_agent as mage
    import travelplan_env as env
    from TravelPlanEnv.multitraveler import infer_query_id_for_sample
    from TravelPlanEnv import scoring

    manifest_data = json.loads((ROOT / "data/manifests/provenance.json").read_bytes())
    split = json.loads((ROOT / "data/manifests/splits.json").read_bytes())
    if args.group not in split["partitions"][args.partition]:
        parser.error("Group is outside the selected partition")
    if args.partition == "all" and args.turns is not None:
        parser.error("Benchmark groups must run every traveler")
    raw = args.travel_release.read_bytes()
    if sha256(raw).hexdigest() != manifest_data["dataset_sha256"]:
        parser.error("Unexpected travel dataset revision")
    sample = next(row for row in map(json.loads, raw.splitlines()) if row["id"] == args.group)
    full_turns = len(sample["questions"])
    query_id = infer_query_id_for_sample(sample, csv_path=args.train_csv)
    if args.turns:
        sample = {**sample, "questions": sample["questions"][:args.turns], "answers": sample["answers"][:args.turns]}

    args.out.mkdir(parents=True, exist_ok=False)
    load_env(ROOT / ".env")
    config = yaml.safe_load((ROOT / "configs/travel.yaml").read_text(encoding="utf-8"))
    agent = config["agent"]
    tokenizer_path, tokenizer_manifest = verify_tokenizer(ROOT)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    conditions = ("pcms",) if args.pcms_only else (args.first_arm, "pcms" if args.first_arm == "mage" else "mage")
    stream_retries = int(len(sample["questions"]) == full_turns)
    manifest = {"status": "running", "author_commit": head, "model": agent["model"], "route": agent["routing"],
                "conditions": conditions, "comparison": "pcms_vs_published_mage" if args.pcms_only else "official_mage_vs_standalone_pcms",
                "interrupted_stream_retries_per_arm": stream_retries,
                "budget_scope": "actions per traveler attempt; official MAGE verification and replay retained",
                "group": args.group, "turns": len(sample["questions"]), "full_turns": full_turns,
                "max_steps": args.max_steps, "memory_tokens": args.memory_tokens,
                "tokenizer_sha256": tokenizer_manifest["sha256"],
                "train_csv_sha256": sha256(args.train_csv.read_bytes()).hexdigest(),
                "travel_sha256": manifest_data["dataset_sha256"], "query_id": query_id,
                "seed": config["experiment"]["seed"], "actor_completion_cap": agent["max_completion_tokens"],
                "actor_thinking": args.actor_thinking,
                "source_sha256": {file.name: sha256(file.read_bytes()).hexdigest() for file in source_paths}}
    with ZipFile(args.out / "source.zip", "x", ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, path.relative_to(ROOT))
    write_json(args.out / "manifest.json", manifest)
    trace = LabTrace(args.out, args.lab_control, args.lab_state)
    trace.emit("run", "Paired MAGE run ready", manifest=manifest)
    author = mage.TravelPlanMAGEAgent
    try:
        for condition in conditions:
            directory = args.out / condition
            directory.mkdir()
            trace.emit("run", "Comparison arm started", condition=condition)
            trace.before("Start comparison arm", category="arm", condition=condition)
            with LabPolzaClient(trace=trace, model=agent["model"], base_url=agent["base_url"], provider=agent["routing"],
                             seed=manifest["seed"], temperature=agent["temperature"],
                             max_tokens=agent["max_completion_tokens"], extraction_max_tokens=2048,
                             extraction_reasoning=agent["extraction_reasoning"], stream=agent["stream"],
                             retries=2, interrupted_stream_retries=stream_retries,
                             timeout=agent["request_timeout_seconds"], max_total_tokens=math.inf,
                             max_cost_rub=None, usage_path=directory / "usage.jsonl") as client:
                runtime = env.TravelPlanRuntime(query_id=query_id, csv_path=args.train_csv)
                llm = MagePolza(client, directory, condition, args.actor_thinking)
                if condition == "mage":
                    active = run_mage(mage, author, runtime, sample, llm, directory, args.max_steps, trace)
                else:
                    active = run_pcms(sample, runtime, env, llm, directory,
                                      lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids),
                                      args.max_steps, args.memory_tokens, trace)
            trace.before("Score comparison arm", category="score", condition=condition)
            scores = scoring.score_travelers(sample["base_person"]["daily_plans"], sample["answers"], active)["sub_successes"]
            if len(scores) != len(sample["questions"]):
                raise ValueError("Author scorer did not return one result per evaluated traveler")
            result = {"condition": condition, "sub_successes": scores, "active": active}
            success_key = "task_success" if len(sample["questions"]) == full_turns else "prefix_success"
            result[success_key] = all(score >= 1 for score in result["sub_successes"])
            write_json(directory / "result.json", result)
            trace.emit("score", "Comparison arm scored", **result)
        manifest["status"] = "complete"
    except LabStopped:
        manifest["status"] = "stopped"
    except Exception as error:
        manifest.update(status="failed", error=redact(str(error)))
        raise SystemExit(manifest["error"])
    finally:
        write_json(args.out / "manifest.json", manifest)
        trace.emit("run", "Paired MAGE run ended", status=manifest["status"], error=manifest.get("error"))


if __name__ == "__main__":
    main()
