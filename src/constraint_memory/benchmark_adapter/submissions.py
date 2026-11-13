"""Latest full named sections, matching the pinned public runner's semantics."""
from __future__ import annotations
import copy
import re


def named_sections(text):
    parts=re.split(r"===\s*([^=]+?)'s Plan\s*===",text or "")
    sections={}
    for index in range(1,len(parts),2):
        if index+1<len(parts):
            sections[parts[index].strip()]=parts[index+1].strip()
    return sections


class SubmissionLedger:
    """Full-section replacement; future names and the fixed base cannot be revised."""
    def __init__(self,group_id,base_name):
        self.group_id=str(group_id)
        self.base_name=base_name
        self.requests={}
        self.latest={}
        self.history=[]

    def register_request(self,packet):
        if str(packet["group_id"])!=self.group_id:
            raise ValueError("Submission request crosses group boundary")
        index=packet["traveler_index"]
        if index!=len(self.requests)+1:
            raise ValueError("Requests must be registered in strict chronological order")
        self.requests[index]=copy.deepcopy(packet)

    @property
    def visible_travelers(self):
        return {packet["name"]:index for index,packet in self.requests.items()}

    def accept(self,result):
        if str(result["group_id"])!=self.group_id or result["traveler_index"] not in self.requests:
            raise ValueError("Submission is outside the visible current/past request boundary")
        source_round=result.get("source_round",result["traveler_index"])
        if source_round!=len(self.requests):
            raise ValueError("Submission source round must be the current chronological boundary")
        raw_sections=named_sections(result.get("final_plan",""))
        proposals={item["name"]:item for item in result.get("accepted_submissions",[])}
        # Legacy scripted clients provide a current parsed section only. They
        # still cannot introduce unknown or unparsed prior named sections.
        current=self.requests[result["traveler_index"]]
        if not proposals and current["name"] in raw_sections and result.get("submitted_text") is not None:
            proposals[current["name"]]={"name":current["name"],"submitted_text":result["submitted_text"],"plan":result.get("plan")}
        accepted=[]
        ignored=[]
        for name,text in raw_sections.items():
            if name==self.base_name:
                ignored.append({"name":name,"reason":"FIXED_BASE_REVISION"})
                continue
            indices=[index for index,packet in self.requests.items() if packet["name"]==name]
            if not indices:
                ignored.append({"name":name,"reason":"FUTURE_OR_UNKNOWN_TRAVELER"})
                continue
            proposal=proposals.get(name)
            if proposal is None or proposal.get("submitted_text")!=text:
                raise ValueError("Accepted parsed section differs from the actual named submission")
            for index in indices:
                previous=self.latest.get(index)
                entry={"group_id":self.group_id,"traveler_index":index,"name":name,
                       "submitted_text":text,"final_plan":f"=== {name}'s Plan ===\n{text}",
                       "plan":copy.deepcopy(proposal.get("plan")),"source_round":source_round,
                       "source_attempt":result.get("source_attempt",1),
                       "submission_version":previous["submission_version"]+1 if previous else 1,
                       "total_steps":result.get("total_steps"),"error_message":result.get("error_message")}
                self.latest[index]=entry
                self.history.append(copy.deepcopy(entry))
                accepted.append(copy.deepcopy(entry))
        # A missing current section does not erase an already submitted plan
        # during repair; an explicit empty/malformed section does replace it.
        return {"accepted":accepted,"ignored":ignored,"source_round":source_round}

    def predictions(self,variant=None,family=None):
        result=[]
        for index,packet in sorted(self.requests.items()):
            entry=copy.deepcopy(self.latest.get(index,{"group_id":self.group_id,"traveler_index":index,
                "name":packet["name"],"plan":None,"final_plan":"","source_round":index,
                "submission_version":0,"error_message":"MISSING_NAMED_SUBMISSION: "+packet["name"]}))
            if variant is not None:
                entry["variant"]=variant
            if family is not None:
                entry["base_template_id"]=family
            result.append(entry)
        return result
