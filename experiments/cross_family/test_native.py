"""Archive/replay integration with actual sessions and scripted outputs, no score."""
import json
import os
from pathlib import Path
import uuid

from constraint_memory.config import verify_tokenizer, sha256
from constraint_memory.orchestration.group_session import GroupSession
from constraint_memory.orchestration.replay import replay_run
from constraint_memory.orchestration.runner import format_base
from experiments.cross_family import run

ROOT = Path(__file__).resolve().parents[2]


def test_native_sessions_archive_and_tokenizer_replay(monkeypatch):
    """Actual causal adapter/memory/session, scripted writer/actor, no gold.

    This tests integration, not efficacy. It reads the first selected group's
    public requests chronologically; the actor copies the public base plan.
    """
    from tokenizers import Tokenizer
    protocol = run.build_protocol(ROOT)
    shadow = ROOT / "tmp/cross-family-native" / uuid.uuid4().hex
    destination = shadow / "runs/offline-adapter-check"
    destination.mkdir(parents=True)
    tokenizer_path, _ = verify_tokenizer(ROOT)
    copied_tokenizer = shadow / "external/qwen-tokenizer/tokenizer.json"
    copied_tokenizer.parent.mkdir(parents=True)
    os.link(tokenizer_path, copied_tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_pins = run.source_map(ROOT, "src")
    for relative in source_pins:
        target = shadow / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    (destination / "manifest.json").write_text(json.dumps({"run_id": destination.name,
        "source_sha256": source_pins, "tokenizer_sha256": sha256(tokenizer_path),
        "mode": "scripted_offline_adapter_check_no_model_or_scores"}), encoding="utf-8")
    archive = run.archive_run_source(shadow, destination)
    assert archive["original_source_pin_verification"] == "full"

    class ScriptedClient:
        def __init__(self):
            self.records = []
            self.writer_scopes = []

        def set_scope(self, group, traveler=None, operation="agent"):
            self.scope = group, traveler

        def complete(self, messages, **kwargs):
            assert kwargs["operation"] == "extraction"
            self.writer_scopes.append(self.scope)
            return {"choices": [{"message": {"content": json.dumps({"constraints": [], "definitions": [],
                "decisions": [], "unresolved": []})}, "finish_reason": "stop"}]}

    class ScriptedPlanner:
        def __init__(self, *args, **kwargs):
            assert kwargs["max_steps"] == 30

        def run_turn(self, packet, context, base):
            text = format_base({**base, "name": packet["name"]})
            return {"name": packet["name"], "plan": base["daily_plans"], "submitted_text": text.split("\n", 1)[1].strip(),
                "final_plan": text, "scratchpad": [], "structured_tool_results": [],
                "total_steps": 1, "parent_model_requests": 1, "success": True}

    # Actual sessions must not open hidden/annotation data; no scorer is called.
    original_open = Path.open

    def guard(path, *args, **kwargs):
        normalized = str(path).replace("\\", "/").lower()
        assert "/data/hidden/" not in normalized and "/data/audit_annotations/" not in normalized
        assert path.name != ".env"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    client = ScriptedClient()
    group = protocol["group_ids"][0]
    readiness = {"database": {"database_path": "unused-no-tool-access"}}
    for variant in ("E1", "E2"):
        predictions, row = GroupSession(ROOT, destination, variant, group, protocol["config"], readiness,
            tokenizer, client, ScriptedPlanner, format_base).run()
        assert len(predictions) == row["traveler_count"] == 7
        assert row["variant"] == variant
    assert len(client.writer_scopes) == 14
    assert all(scope in {f"E1:{group}", f"E2:{group}"} and traveler for scope, traveler in client.writer_scopes)
    replay = replay_run(destination)
    assert len(replay) == 2 and all(item["match"] for item in replay)
    assert not (destination / "results.jsonl").exists()
