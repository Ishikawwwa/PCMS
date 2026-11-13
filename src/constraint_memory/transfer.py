"""Model transfer diagnostic over domain-neutral observed facts and contracts.

Cases supply actual task inputs, never expected answers. Evaluation happens
after the model commits an option. This measures structured-contract memory;
it does not certify free-form natural-language extraction or MAGE superiority.
"""
from __future__ import annotations

import copy
import json

from .memory.portable import ConstraintMemory


def encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)


def build_memory(case):
    required={"case_id","domain","events","active_constraint_refs","options","query","format"}
    if set(case)!=required or case["format"] not in {"tables","json","messages"}:
        raise ValueError("Transfer cases must contain only declared task input fields")
    memory=ConstraintMemory(case["case_id"])
    last=-1
    for event in case["events"]:
        if event["step"]<last: raise ValueError("Transfer history must be chronological")
        last=event["step"]
        memory.observe(event["event_id"],last,event["content"],kind=event["kind"])
        for fact in event.get("facts",[]):
            memory.write_fact(fact["ref"],event["event_id"],fact["pointer"],last,
                unit=fact.get("unit"),entity_ref=fact.get("entity_ref"))
        for rule in event.get("constraints",[]):
            memory.write_constraint(rule["ref"],event["event_id"],rule["pointer"],last)
    for reference in case["active_constraint_refs"]:
        if reference not in memory.constraints: raise ValueError("Current request references an unknown contract")
    ids=[option["option_id"] for option in case["options"]]
    if not ids or any(not isinstance(value,str) or not value for value in ids) or len(set(ids))!=len(ids):
        raise ValueError("Options require unique opaque identities")
    for option in case["options"]:
        if set(option)!={"option_id","event_id","pointer","properties"}: raise ValueError("Invalid observed option declaration")
        event=memory.events.get(option["event_id"])
        if event is None or event["kind"]!="observation": raise ValueError("Options must cite already visible observations")
        row,_=memory._source(option["event_id"],option["pointer"],last)
        if not isinstance(row,dict) or row.get("option_id")!=option["option_id"]:
            raise ValueError("Option identity differs from exact observed row")
        for field in option["properties"]:
            if not field["pointer"].startswith(option["pointer"]+"/"):
                raise ValueError("Option properties must belong to its exact observed row")
            memory._source(option["event_id"],field["pointer"],last)
    return memory,last


def context_for(case,condition,count,budget):
    memory,step=build_memory(case)
    if condition=="CONSTRAINT":
        frontier=memory.frontier(case["active_constraint_refs"],step,count,10**9)
        sections=frontier["sections"]
        if case["format"]=="tables": context=frontier["context"]
        elif case["format"]=="json": context=encoded(sections)
        else: context="\n".join(f"{name}: {encoded(rows)}" for name,rows in sections.items())
        tokens=count(context)
        return {"context":context if tokens<=budget else "BUDGET_OVERFLOW",
                "status":"OK" if tokens<=budget else "BUDGET_OVERFLOW",
                "required_tokens":tokens,"omitted_events":[]}
    if condition!="HISTORY": raise ValueError("Unknown transfer condition")
    events=copy.deepcopy(case["events"])
    omitted=[]
    def render():
        if case["format"]=="messages":
            return "\n".join(f"{row['kind']} {row['event_id']} at {row['step']}: {encoded(row['content'])}" for row in events)
        return encoded(events)
    context=render()
    while events and count(context)>budget:
        omitted.append(events.pop(0)["event_id"])
        context=render()
    if count(context)>budget: context=""
    return {"context":context,"status":"PARTIAL_CONTEXT" if omitted else "OK",
            "required_tokens":count(context),"omitted_events":omitted}


def messages_for(case,context):
    memory,step=build_memory(case)
    rows=[memory._source(option["event_id"],option["pointer"],step)[0] for option in case["options"]]
    return [{"role":"system","content":(
        "Choose one currently offered option under the prior project requirements. "
        "Use the latest causal facts; unresolved evidence requires abstention. "
        "Return JSON with exactly option_id: an offered identity, or null if no "
        "option can be verified. Do not invent values or change past facts.")},
        {"role":"user","content":encoded({"request":case["query"],
            "required_contract_refs":case["active_constraint_refs"],"current_options":rows,
            "option_property_bindings":case["options"],
            "memory":context["context"]})}]


def evaluate_selection(case,response):
    """Called only after a committed model response; never planner feedback."""
    def check_option(option):
        memory,step=build_memory(case)
        step+=1
        memory.observe("transfer-committed-action",step,{"option_id":option["option_id"]},kind="action")
        selected=memory.write_fact("transfer.selected","transfer-committed-action","/option_id",step)
        for field in option["properties"]:
            memory.write_fact(field["ref"],option["event_id"],field["pointer"],step,
                unit=field.get("unit"),entity_ref={"ref":"transfer.selected","version":selected["version"]})
        checks=[memory.check(ref,step) for ref in case["active_constraint_refs"]]
        return checks
    if not isinstance(response,dict) or set(response)!={"option_id"}:
        return {"status":"INVALID_SUBMISSION","passed":False,"checks":[]}
    option=next((row for row in case["options"] if row["option_id"]==response["option_id"]),None)
    if response["option_id"] is None:
        feasible=any(bool(checks) and all(row["status"]=="SATISFIED" for row in checks)
                     for option in case["options"] if (checks:=check_option(option)) is not None)
        return {"status":"ABSTAINED","passed":not feasible,"checks":[],"feasible_option_exists":feasible}
    if option is None: return {"status":"UNKNOWN_OPTION","passed":False,"checks":[]}
    checks=check_option(option)
    return {"status":"EVALUATED","passed":bool(checks) and all(row["status"]=="SATISFIED" for row in checks),"checks":checks}
