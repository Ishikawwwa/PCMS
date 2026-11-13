"""Private stdio worker for the unchanged official TravelPlannerAgent."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from .data import assert_visible_request
from .upstream import import_official, mount_upstream, TRAVEL_PACKAGE
from .submissions import named_sections
from .tool_arguments import normalize_city_arguments,TOOL_ARGUMENT_CONTRACT_TEXT
from .property_schema import PRICE_SCHEMA_CONTRACT_TEXT


def deny_hidden_reads(paths: list[str]) -> None:
    forbidden = [os.path.normcase(os.path.abspath(p)) for p in paths]

    def audit(event: str, args: tuple):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            candidate = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            if any(candidate == p or candidate.startswith(p + os.sep) for p in forbidden):
                raise PermissionError("Planner access to hidden/future dataset paths is prohibited")
        if event == "import" and isinstance(args[0], str):
            module = args[0]
            if module == "datasets" or module.startswith("datasets.") or module.endswith("travel_planner_env.eval") or module.endswith("travel_planner_env.data_loader"):
                raise ImportError("Planner cannot import the hidden scorer or dataset loader")

    sys.addaudithook(audit)


def format_person_plan(name: str, daily_plans: list[dict]) -> str:
    lines = [f"=== {name}'s Plan ==="]
    fields = ["current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]
    for day in daily_plans:
        lines.append(f"Day {day.get('days') or day.get('day')}:")
        for field in fields:
            title = " ".join(word.capitalize() for word in field.split("_"))
            lines.append(f"{title}: {day.get(field, '-')}")
        lines.append("")
    return "\n".join(lines)


def record_visible_tool_results(executor,city_adapter=False):
    """Capture official return rows without issuing extra database queries.

    ToolExecutor formats DataFrames with to_string(index=False), which includes
    all returned rows and columns. Each structured record also stores the exact
    agent-visible string. Only the columns already selected by each official
    tool are captured; no raw database objects are exposed.
    """
    observations: list[dict] = []
    original_execute = executor.execute
    original_format = executor._format_result
    current: list[dict] = []

    def format_result(result, tool_name):
        formatted = original_format(result, tool_name)
        observation = current[-1]
        if hasattr(result, "to_dict") and hasattr(result, "columns"):
            observation["displayed_columns"] = list(result.columns)
            observation["displayed_rows"] = result.to_dict(orient="records")
            observation["return_type"] = "dataframe"
        elif isinstance(result, dict):
            observation["displayed_columns"] = list(result)
            observation["displayed_rows"] = [dict(result)]
            observation["return_type"] = "dict"
        elif isinstance(result, list):
            observation["displayed_values"] = list(result)
            observation["return_type"] = "list"
        else:
            observation["return_type"] = "text"
        return formatted

    def execute(tool_name, arguments):
        effective,changes=normalize_city_arguments(tool_name,arguments,city_adapter)
        observation = {"execution_index": len(observations), "tool_name": tool_name,
                       "arguments": dict(arguments), "displayed_columns": [], "displayed_rows": [],
                       "source": "official_tool_return_before_formatting"}
        if city_adapter:
            observation.update(effective_arguments=effective,argument_normalization=changes)
        current.append(observation)
        start = time.perf_counter()
        try:
            formatted = original_execute(tool_name, effective)
            observation["latency_seconds"] = time.perf_counter() - start
            observation["formatted_result"] = formatted
            observation["formatted_result_sha256"] = hashlib.sha256(formatted.encode("utf-8")).hexdigest()
            observation.setdefault("return_type", "text")
            observations.append(observation)
            return formatted
        finally:
            current.pop()

    executor.execute = execute
    executor._format_result = format_result
    return observations


def run(payload: dict, protocol_output) -> dict:
    assert_visible_request(payload["request"])
    deny_hidden_reads(payload["denied_paths"])
    upstream_root = Path(payload["upstream_root"])
    mount_upstream(upstream_root)
    from agent.travel_planner import TravelPlannerAgent
    from env.env_systems.travel_planner_env.clients.base_client import ModelResponse, ToolCall
    parser = import_official(upstream_root, "combination").parse_plan_text

    class RemoteClient:
        def chat_with_tools(self, messages, tools, **kwargs):
            protocol_output.write(json.dumps({"kind": "model_request", "messages": messages, "tools": tools}, ensure_ascii=False) + "\n")
            protocol_output.flush()
            response = json.loads(sys.stdin.readline())
            return ModelResponse(response.get("content"), [ToolCall(**call) for call in response.get("tool_calls", [])], response.get("raw_response"))

        def format_assistant_tool_calls(self, tool_calls):
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                for call in tool_calls]}

        def format_tool_result(self, tool_call_id, result, name=None):
            message = {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}
            if name:
                message["name"] = name
            return message

    class InjectedAgent(TravelPlannerAgent):
        def _create_client(self, model_name):
            return RemoteClient()

    agent = InjectedAgent(model_name=payload["model"], temperature=payload["temperature"], max_steps=payload["max_steps"], db_path=payload["db_path"])
    city_adapter=payload.get("tool_city_state_annotation_normalization",False)
    if not isinstance(city_adapter,bool):
        raise ValueError("Tool city adapter must be an explicit shared boolean")
    structured_observations = record_visible_tool_results(agent.executor,city_adapter)
    agent.reset()
    base = payload["base_person"]
    base_query=base["query"]
    contract=payload.get("task_contract_text")
    if contract is not None:
        from constraint_memory.orchestration.contracts import INHERITANCE_CONTRACT_TEXT
        if contract != INHERITANCE_CONTRACT_TEXT:
            raise ValueError("Unsupported declared task contract")
        base_query += "\n\n" + contract
    if city_adapter:
        base_query += "\n\n" + TOOL_ARGUMENT_CONTRACT_TEXT
    price_unit=payload.get("tool_price_unit")
    if price_unit not in {None,"USD"}:
        raise ValueError("Unknown declared travel price unit")
    if price_unit:
        base_query += "\n\n" + PRICE_SCHEMA_CONTRACT_TEXT
    agent.set_base_person(base["name"], base_query, format_person_plan(base["name"], base["daily_plans"]))
    request = payload["request"]
    result = agent.run_single_person(query=request["query"], name=request["name"], round_idx=request["traveler_index"],
                                     include_previous_plans=False, memory_context=payload["memory_context"] or None, memory_system=None)
    # The upstream cumulative parser keeps the LAST FULL section for each
    # current/past added traveler. Extra sections never revise the fixed base.
    sections=named_sections(result.final_plan)
    visible=payload.get("visible_travelers",{request["name"]:request["traveler_index"]})
    if any(not isinstance(index,int) or index<1 or index>payload.get("causal_round",request["traveler_index"]) for index in visible.values()):
        raise ValueError("Submission name map contains a future traveler")
    accepted=[{"name":name,"traveler_index":visible[name],"submitted_text":text,"plan":parser(text)}
              for name,text in sections.items() if name in visible and name!=base["name"]]
    named_plan=sections.get(request["name"],"")
    tool_results = [tool for step in result.scratchpad for tool in step.tool_results or []]
    if len(tool_results) != len(structured_observations):
        raise RuntimeError("Tool observations are not aligned with the official scratchpad")
    for structured, tool in zip(structured_observations, tool_results):
        if structured["formatted_result"] != tool["result"] or structured["tool_name"] != tool["name"]:
            raise RuntimeError("Structured tool observation differs from the official agent-visible result")
        structured["tool_call_id"] = tool["tool_call_id"]
    return {"group_id": request["group_id"], "traveler_index": request["traveler_index"], "name": request["name"],
            "final_plan": result.final_plan, "submitted_text": named_plan, "plan": parser(named_plan),
            "accepted_submissions":accepted,
            "scratchpad": agent.get_scratchpad_dict(), "total_steps": result.total_steps,
            "structured_tool_results": structured_observations,
            "success": result.success, "error_message": result.error_message,
            "worker_audit": {"fresh_process_per_turn": True, "current_query_only": True, "previous_plans_via_memory_only": True,
                             "judge_feedback": "none", "hidden_modules_loaded": [name for name in sys.modules if name.endswith("travel_planner_env.eval") or name.endswith("travel_planner_env.data_loader")],
                             "model_credentials_present": any("API_KEY" in key for key in os.environ),
                             "shared_declared_task_contract":contract,
                             "tool_city_state_annotation_normalization":city_adapter,
                             "shared_tool_argument_contract":TOOL_ARGUMENT_CONTRACT_TEXT if city_adapter else None,
                             "tool_price_unit":price_unit,
                             "shared_price_schema_contract":PRICE_SCHEMA_CONTRACT_TEXT if price_unit else None,
                             "official_agent": "agent.travel_planner.TravelPlannerAgent", "official_parser": TRAVEL_PACKAGE + ".combination.parse_plan_text"}}


def main() -> None:
    output = sys.stdout
    try:
        payload = json.loads(sys.stdin.readline())
        with contextlib.redirect_stdout(sys.stderr):
            result = run(payload, output)
        output.write(json.dumps({"kind": "result", "result": result}, ensure_ascii=False) + "\n")
        output.flush()
    except Exception as exc:
        output.write(json.dumps({"kind": "error", "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
