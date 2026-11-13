"""A shared tool adapter changes one wire argument, never calls another tool."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from constraint_memory.benchmark_adapter.planner import PlannerWorker
from constraint_memory.benchmark_adapter.tool_arguments import normalize_city_arguments,TOOL_ARGUMENT_CONTRACT_TEXT


ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/"external/MemoryArena"
DB=UPSTREAM/"env/env_systems/travel_planner_env/database"


@pytest.mark.parametrize("city,effective",[("San Antonio(Texas)","San Antonio"),("Houston(TX)","Houston"),
    ("Boston (Massachusetts)","Boston"),("Washington(DC)","Washington")])
def test_known_us_state_annotation_has_one_declared_wire_mapping(city,effective):
    original={"city":city,"other":"literal"}
    result,changes=normalize_city_arguments("RestaurantSearch",original,True)
    assert original=={"city":city,"other":"literal"}
    assert result=={"city":effective,"other":"literal"}
    assert changes["city"]=={"original":city,"effective":effective,"rule":"recognized_us_state_annotation"}


@pytest.mark.parametrize("city",["Houston","Boston(airport)","Paris(France)","New York(Manhattan)",
    "Houston(Texas)(TX)","(Texas)",None,17])
def test_unknown_or_noncanonical_suffix_is_never_guessed(city):
    assert normalize_city_arguments("AccommodationSearch",{"city":city},True)==({"city":city},{})


@pytest.mark.parametrize("tool",["FlightSearch","GoogleSearch","UnknownTool"])
def test_unrelated_tools_and_disabled_adapter_remain_literal(tool):
    assert normalize_city_arguments(tool,{"city":"Houston(Texas)"},True)==({"city":"Houston(Texas)"},{})
    assert normalize_city_arguments("RestaurantSearch",{"city":"Houston(Texas)"},False)==({"city":"Houston(Texas)"},{})


@pytest.mark.skipif(not (DB/"restaurants/clean_restaurant.csv").exists() and not DB.exists(),reason="official database not prepared")
@pytest.mark.parametrize("enabled",[False,True])
def test_real_official_worker_retains_requested_args_and_executes_one_search(enabled):
    seen=[]
    class ScriptedClient:
        def chat_with_tools(self,messages,tools):
            seen.append(messages)
            if len(seen)==1:
                call=SimpleNamespace(id="search",name="RestaurantSearch",arguments={"city":"Houston(Texas)"})
                return SimpleNamespace(content=None,tool_calls=[call],raw_response={})
            return SimpleNamespace(content="=== Current's Plan ===\nDay 1:\nLunch: scripted fixture\n",
                                   tool_calls=[],raw_response={})
    planner=PlannerWorker(UPSTREAM,DB,ScriptedClient(),max_steps=2,timeout_seconds=60,
                          tool_city_state_annotation_normalization=enabled)
    result=planner.run_turn({"group_id":"g","name":"Current","traveler_index":1,"query":"CURRENT"},
        "MEMORY",{"name":"Base","query":"BASE","daily_plans":[{"day":1,"current_city":"Houston(Texas)"}]})
    assert result["parent_model_requests"]==2 and len(result["structured_tool_results"])==1
    observation=result["structured_tool_results"][0]
    assert observation["arguments"]=={"city":"Houston(Texas)"}
    assert result["scratchpad"][0]["tool_calls"][0]["args"]==observation["arguments"]
    assert result["worker_audit"]["tool_city_state_annotation_normalization"]==enabled
    if enabled:
        assert observation["effective_arguments"]=={"city":"Houston"}
        assert len(observation["displayed_rows"])>0
        assert all(row["City"]=="Houston" for row in observation["displayed_rows"])
        assert TOOL_ARGUMENT_CONTRACT_TEXT in json.dumps(seen[0])
    else:
        assert observation["displayed_rows"]==[]
        assert "effective_arguments" not in observation
