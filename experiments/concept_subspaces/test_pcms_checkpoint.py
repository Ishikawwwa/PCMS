import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from .lab_trace import LabTrace
from .pcms_travel import run_pcms


def worker(directory, resume):
    runtime = SimpleNamespace(query_id=1, itinerary=[], registry={})
    runtime.env = SimpleNamespace(get_itinerary=lambda _: list(runtime.itinerary))
    runtime.prepare_session = lambda *args, **kwargs: None
    runtime.snapshot_full = lambda: {"sessions": {"session": {}}, "itinerary": runtime.itinerary,
                                     "registry": runtime.registry}

    def restore(state):
        runtime.itinerary = state["itinerary"]
        runtime.registry = state["registry"]

    runtime.restore_full = restore
    domain = SimpleNamespace(derive_traveler_name=lambda _: "Traveler", build_session_id=lambda *_: "session")

    class Tool:
        def __init__(self, name, callback):
            self.action_name = self.action_desc = name
            self.params_doc = {}
            self.callback = callback

        def __call__(self, **params):
            return self.callback(**params)

    def search():
        runtime.registry["option-7"] = "Plan"
        return "option-7"

    def select(option_id):
        runtime.itinerary.append(runtime.registry[option_id])
        return "Selected"

    runtime_modules = {
        "agentcore.actions": {"ThinkAct": Tool("Think", lambda: None)},
        "agentcore.agents.agent_utils": {"parse_action": lambda raw: (*json.loads(raw), True)},
        "travelplan_actions": {"build_travel_actions": lambda *_: [
            Tool("Search", search), Tool("Select", select), Tool("Finish", lambda: "Done")
        ]},
    }
    for name, attrs in runtime_modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module

    class InterruptedTrace(LabTrace):
        def before(self, operation, **details):
            if operation == "Execute Select" and not resume:
                # Exit without any cleanup to ensure the next process relies on disk.
                os._exit(23)

    replies = iter([["Finish", {}]] if resume else [["Search", {}], ["Select", {"option_id": "option-7"}]])
    llm = SimpleNamespace(run=lambda *args, **kwargs: json.dumps(next(replies)))
    trace = InterruptedTrace(directory)
    if resume:
        trace.resume()
    sample = {"base_person": {"query": "Base", "daily_plans": []}, "questions": ["Plan a trip"]}
    results = run_pcms(sample, runtime, domain, llm, directory, len, 3, 256, trace, resume=resume)
    (directory / "test-results.json").write_text(json.dumps(results), encoding="utf-8")


def test_pcms_resumes_pending_action_after_worker_exit(tmp_path):
    code = ("from pathlib import Path; from experiments.concept_subspaces.test_pcms_checkpoint import worker; "
            "import sys; worker(Path(sys.argv[1]), sys.argv[2] == 'resume')")
    command = [sys.executable, "-c", code, str(tmp_path)]
    assert subprocess.run([*command, "first"], check=False).returncode == 23
    subprocess.run([*command, "resume"], check=True)
    results = json.loads((tmp_path / "test-results.json").read_bytes())
    assert results[0]["plan_json"] == ["Plan"]
    assert results[0]["completed"]
    actions = [json.loads(line) for line in (tmp_path / "actions.jsonl").read_bytes().splitlines()]
    assert [(row["step"], row["action"]) for row in actions] == [(0, "Search"), (1, "Select"), (2, "Finish")]
