"""Check truncated chosen names against already observed complete tool rows.

This reports a repair hint only for a unique literal title before a colon. It
never fuzzy-matches, materializes properties for that abbreviation, rewrites a
submission, queries a database, or treats an unobserved choice as impossible.
"""
from __future__ import annotations

import copy
import re

from .backends import canonical,reject_privileged
from .grounding import observed_rows,matching_entity_rows,city_match_rank


SEARCH_SLOTS={"breakfast","lunch","dinner","accommodation"}


def validate_choice_identity(request,actual_records,visible_events,requested_slots,step):
    reject_privileged(request)
    group,traveler=str(request["group_id"]),request["traveler"]
    if type(step) is not int or step<request["step"]:
        raise ValueError("Choice validation needs the actual causal commitment step")
    result={"status":"UNVERIFIED","task_success":None,"source_kind":"OBSERVED_TOOL_NAME",
            "violations":[],"checked_slot_count":0,"unverified_slots":[]}
    if not isinstance(requested_slots,list): raise ValueError("Requested slots must be explicit scoped exceptions")
    allowed={(row["day"],row["slot"]) for row in requested_slots if row.get("slot") in SEARCH_SLOTS}
    observations=[]
    for event in visible_events:
        reject_privileged(event)
        if (str(event.get("group_id"))!=group or type(event.get("step")) is not int or event["step"]>step
                or event.get("type")!="tool_result"):
            continue
        for row in observed_rows(event["content"]):
            folded={str(key).casefold():value for key,value in row.items()}
            if isinstance(folded.get("name"),str) and isinstance(folded.get("city"),str):
                observations.append({"row":row,"name":folded["name"],"city":folded["city"],
                                     "source_ref":"event:"+event["event_id"]})
    for record in actual_records:
        reject_privileged(record)
        if (record.get("traveler")!=traveler or record.get("attribute","value")!="value"
                or (record.get("day"),record.get("slot")) not in allowed): continue
        if str(record.get("group_id",record.get("scope",{}).get("group",group)))!=group:
            raise ValueError("Actual choices cross the request group")
        if record.get("status")!="COMMITTED" or type(record.get("created_at")) is not int or record["created_at"]>step:
            continue
        value=record.get("value")
        if not isinstance(value,str) or "," not in value:
            result["unverified_slots"].append({"day":record["day"],"slot":record["slot"]})
            continue
        name,city=[part.strip(" \t") for part in value.rsplit(",",1)]
        indices,_,_=matching_entity_rows(name,city,[item["row"] for item in observations])
        if indices:
            result["checked_slot_count"]+=1
            continue
        # Only whitespace around the colon/title can differ. Every title
        # character and case remains exact; no arbitrary-prefix alias exists.
        title=re.sub(r"[ \t]+"," ",name).strip(" \t")
        candidates={}
        for item in observations:
            full=item["name"]
            if ":" not in full or not city_match_rank(city,item["city"]): continue
            prefix=re.sub(r"[ \t]+"," ",full.split(":",1)[0]).strip(" \t")
            if prefix!=title or not full.split(":",1)[1].strip(): continue
            key=canonical(item["row"])
            candidates.setdefault(key,{**item,"refs":[]})["refs"].append(item["source_ref"])
        if len(candidates)!=1:
            result["unverified_slots"].append({"day":record["day"],"slot":record["slot"],
                                              "reason":"NO_UNIQUE_OBSERVED_FULL_NAME"})
            continue
        candidate=next(iter(candidates.values()))
        result["checked_slot_count"]+=1
        result["violations"].append({"code":"TRUNCATED_OBSERVED_ENTITY_NAME",
            "subject":{"traveler":traveler,"day":record["day"],"slot":record["slot"]},
            "actual":value,"observed_full_name":candidate["name"],"selected_city":city,
            "actual_ref":f"{record['decision_id']}@v{record['version']}",
            "actual_source_refs":copy.deepcopy(record["source_refs"]),
            "observed_source_refs":list(dict.fromkeys(candidate["refs"]))})
    result["status"]="VIOLATED" if result["violations"] else "PASSED" if result["checked_slot_count"] and not result["unverified_slots"] else "UNVERIFIED"
    return result
