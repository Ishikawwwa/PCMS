"""The real official worker receives the same declared rule in every condition."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from constraint_memory.benchmark_adapter.planner import PlannerWorker
from constraint_memory.config import load_config
from constraint_memory.orchestration.contracts import INHERITANCE_CONTRACT_TEXT


ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/"external/MemoryArena"
DB=UPSTREAM/"env/env_systems/travel_planner_env/database"


@pytest.mark.skipif(not (UPSTREAM/"agent/travel_planner.py").exists(),reason="official source not prepared")
def test_real_worker_shared_contract_keeps_raw_current_and_public_base_queries():
    seen=[]
    class ScriptedClient:
        def chat_with_tools(self,messages,tools):
            seen.append(messages)
            return SimpleNamespace(content="=== Current's Plan ===\nDay 1:\nDinner: Current dinner\n",
                                   tool_calls=[],raw_response={})
    request={"group_id":"g","traveler_index":1,"name":"Current","query":"RAW_CURRENT_QUERY"}
    base={"name":"Base","query":"RAW_BASE_QUERY","daily_plans":[{"day":1,"dinner":"Base dinner"}]}
    planner=PlannerWorker(UPSTREAM,DB,ScriptedClient(),max_steps=1,timeout_seconds=60,
                          task_contract_text=INHERITANCE_CONTRACT_TEXT)
    result=planner.run_turn(request,"VISIBLE_MEMORY",base)
    joined=json.dumps(seen[0])
    assert "RAW_CURRENT_QUERY" in joined and "RAW_BASE_QUERY" in joined
    assert INHERITANCE_CONTRACT_TEXT in joined and "VISIBLE_MEMORY" in joined
    assert result["worker_audit"]["shared_declared_task_contract"]==INHERITANCE_CONTRACT_TEXT
    assert not result["worker_audit"]["hidden_modules_loaded"]
    assert base["query"]=="RAW_BASE_QUERY" and request["query"]=="RAW_CURRENT_QUERY"


def test_inheritance_factor_is_explicit_and_required_for_candidate():
    config=load_config(ROOT/"configs/travel.yaml",ROOT/"configs/variants/E3I.yaml")
    assert config["experiment"]["declared_inheritance_contract"] is True
    assert config["experiment"]["primary_contrast"]==["E0","E3I"]
    config["experiment"]["declared_inheritance_contract"]=False
    from constraint_memory.config import validate_config
    with pytest.raises(ValueError,match="E3I requires"):
        validate_config(config)
