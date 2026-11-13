"""Portable constraint memory with opaque references and structured provenance.

This module consumes any domain's observed JSON facts and explicit JSON task
contracts. Natural-language writer output is retained as advisory unless a
separate domain adapter verifies it; valid JSON alone never establishes meaning.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .binding_tables import serialize_sections
from .predicate_kernel import compare,number,transform,OPERATORS


def _json(value):
    def validate(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item): raise ValueError("JSON object keys must be strings")
            for child in item.values(): validate(child)
        elif type(item) is list:
            for child in item: validate(child)
        elif item is not None and type(item) not in {str,int,float,bool}:
            raise ValueError("Evidence must contain only JSON values")
    validate(value)
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)


def _pointer(content,pointer):
    if not isinstance(pointer,str) or pointer and not pointer.startswith("/"):
        raise ValueError("Evidence pointer must be an RFC 6901 JSON pointer")
    value=content
    for token in pointer.split("/")[1:] if pointer else []:
        # Only the two specified JSON pointer escapes are accepted.
        import re
        if re.search(r"~(?![01])",token): raise ValueError("Invalid evidence pointer escape")
        key=token.replace("~1","/").replace("~0","~")
        if isinstance(value,list):
            if not key.isdigit() or str(int(key))!=key: raise ValueError("Invalid array evidence index")
            value=value[int(key)]
        elif isinstance(value,dict): value=value[key]
        else: raise ValueError("Evidence pointer does not select a structured value")
    return copy.deepcopy(value)


class ConstraintMemory:
    """Causal versioned memory for scheduling, code/config, planning or procurement."""
    def __init__(self,scope_id):
        if not isinstance(scope_id,str) or not scope_id: raise ValueError("An isolated scope is required")
        self.scope_id=scope_id
        self.events={}
        self.facts={}
        self.constraints={}
        self.check_history=[]

    def observe(self,event_id,step,content,*,kind="observation",scope_id=None):
        if scope_id is not None and scope_id!=self.scope_id: raise ValueError("Observation crosses memory scope")
        if not isinstance(event_id,str) or not event_id or type(step) is not int or step<0:
            raise ValueError("Observation identity and causal step required")
        event={"event_id":event_id,"scope_id":self.scope_id,"step":step,"kind":kind,
               "content":copy.deepcopy(content),"sha256":hashlib.sha256(_json(content).encode()).hexdigest()}
        if event_id in self.events and _json(event)!=_json(self.events[event_id]):
            raise ValueError("Observed evidence is immutable")
        self.events[event_id]=event

    def _source(self,event_id,pointer,step):
        if type(step) is not int or step<0: raise ValueError("Causal source step required")
        event=self.events.get(event_id)
        if event is None or event["step"]>step: raise ValueError("Missing or future source evidence")
        try: value=_pointer(event["content"],pointer)
        except (KeyError,IndexError,TypeError): raise ValueError("Invalid evidence pointer") from None
        return value,{"event_id":event_id,"pointer":pointer,"event_sha256":event["sha256"]}

    @staticmethod
    def _append(bucket,reference,record):
        if not isinstance(reference,str) or not reference: raise ValueError("Opaque record reference required")
        history=bucket.setdefault(reference,[])
        if history and record["created_at"]<history[-1]["created_at"]:
            raise ValueError("Record revisions must be chronological")
        record={**record,"reference":reference,"version":len(history)+1}
        history.append(copy.deepcopy(record))
        return copy.deepcopy(record)

    def write_fact(self,reference,event_id,pointer,step,*,unit=None,entity_ref=None):
        if type(step) is not int or step<0: raise ValueError("Causal fact step required")
        if reference in self.constraints: raise ValueError("Fact and constraint references cannot collide")
        if unit is not None and (not isinstance(unit,str) or not unit): raise ValueError("Unit must be an explicit nonempty label")
        if entity_ref is not None:
            if (not isinstance(entity_ref,dict) or not isinstance(entity_ref.get("ref"),str)
                    or type(entity_ref.get("version")) is not int or entity_ref["version"]<1):
                raise ValueError("Entity bindings require an opaque ref and exact version")
            entity=self.resolve(entity_ref["ref"],step,entity_ref["version"])
            if entity is None: raise ValueError("Entity binding lacks causal evidence")
        value,source=self._source(event_id,pointer,step)
        return self._append(self.facts,reference,{"value":value,"unit":unit,"entity_ref":entity_ref,
            "created_at":step,"status":"KNOWN","source":source})

    def write_constraint(self,reference,event_id,pointer,step):
        if type(step) is not int or step<0: raise ValueError("Causal constraint step required")
        if reference in self.facts: raise ValueError("Fact and constraint references cannot collide")
        predicate,source=self._source(event_id,pointer,step)
        event=self.events[event_id]
        if not isinstance(predicate,dict) or not isinstance(predicate.get("subject_ref"),str) or not predicate["subject_ref"] or not isinstance(predicate.get("target"),dict):
            raise ValueError("Constraint needs an opaque subject and typed target")
        if predicate.get("operator") not in OPERATORS: raise ValueError("Unsupported predicate operator")
        target=predicate["target"]
        if ("ref" in target)==("value" in target): raise ValueError("Target must have exactly one reference or literal")
        for key in ("scale","offset"):
            if key in target and number(target[key]) is None: raise ValueError("Numeric transform metadata required")
        for unit in (predicate.get("unit"),target.get("unit")):
            if unit is not None and (not isinstance(unit,str) or not unit): raise ValueError("Explicit nonempty unit label required")
        if predicate.get("unit") and target.get("unit") and predicate["unit"]!=target["unit"]:
            raise ValueError("Constraint declares conflicting units")
        if "ref" in target:
            if not isinstance(target["ref"],str) or not target["ref"]: raise ValueError("Target reference must be nonempty")
            if target.get("binding_mode","LIVE") not in {"LIVE","SNAPSHOT"}: raise ValueError("Unknown target binding mode")
            if "version" in target and (type(target["version"]) is not int or target["version"]<1):
                raise ValueError("Target versions must be positive integers")
            if target.get("binding_mode","LIVE")=="LIVE" and "version" in target:
                raise ValueError("LIVE references cannot pin a version; declare SNAPSHOT")
        # Only explicitly supplied structured contracts are semantically accepted.
        # An observation containing model JSON remains advisory.
        verified=event["kind"]=="task_contract"
        return self._append(self.constraints,reference,{"predicate":predicate,"created_at":step,
            "source":source,"semantic_status":"EXPLICIT_CONTRACT" if verified else "UNVERIFIED"})

    def write_advisory_constraint(self,reference,model_event_id,pointer,user_event_id,start,end,step):
        """Retain source-quoted natural-language interpretation without hard checks."""
        model=self.events.get(model_event_id)
        user=self.events.get(user_event_id)
        if (not model or model["kind"]!="model_output" or not user or user["step"]>step
                or not isinstance(user["content"],str) or type(start) is not int or type(end) is not int
                or not 0<=start<end<=len(user["content"])):
            raise ValueError("Advisory interpretation requires exact causal model/user evidence")
        record=self.write_constraint(reference,model_event_id,pointer,step)
        record.update(source={"event_id":user_event_id,"span":{"start":start,"end":end},
            "quote":user["content"][start:end],"event_sha256":user["sha256"]},
            interpretation_source=record["source"],semantic_status="UNVERIFIED")
        self.constraints[reference][-1]=copy.deepcopy(record)
        return copy.deepcopy(record)

    def resolve(self,reference,step,version=None,*,_seen=None):
        if type(step) is not int or step<0: raise ValueError("Causal read step required")
        if version is not None and (type(version) is not int or version<1): raise ValueError("Exact version must be a positive integer")
        seen=set(_seen or ())
        if reference in seen: return None
        seen.add(reference)
        history=self.facts.get(reference,self.constraints.get(reference,[]))
        visible=[record for record in history if record["created_at"]<=step]
        # Simultaneous, incompatible observations have no causal winner.
        # A later explicit revision can resolve the conflict; insertion order
        # cannot. Exact version reads also retain that uncertainty.
        if visible:
            selected=[record for record in visible if version is None or record["version"]==version]
            if selected:
                selected_step=selected[-1]["created_at"]
                peers=[record for record in visible if record["created_at"]==selected_step]
                signatures={_json({key:record.get(key) for key in
                    ("value","unit","entity_ref","status","predicate","semantic_status")}) for record in peers}
                if len(signatures)>1:
                    return None
        if version is not None: visible=[record for record in visible if record["version"]==version]
        if not visible: return None
        record=visible[-1]
        if record.get("status")=="WITHDRAWN": return None
        entity=record.get("entity_ref")
        if entity:
            # Resolve the full entity ancestry with cycle detection. A stale
            # catalog selection invalidates its properties transitively.
            latest=self.resolve(entity["ref"],step,_seen=seen)
            if not latest or latest.get("status")!="KNOWN" or latest["version"]!=entity["version"]:
                return None
        return copy.deepcopy(record)

    def withdraw_fact(self,reference,event_id,pointer,step):
        if reference not in self.facts: raise ValueError("Cannot withdraw an unknown fact")
        value,source=self._source(event_id,pointer,step)
        if value!={"withdraw":reference}: raise ValueError("Withdrawal requires an explicit observed record")
        return self._append(self.facts,reference,{"created_at":step,"status":"WITHDRAWN","source":source})

    def check(self,reference,step):
        constraint=self.resolve(reference,step)
        result={"constraint_ref":reference,"status":"UNVERIFIED","task_success":None,"step":step,"bindings":[]}
        if not constraint or "predicate" not in constraint or constraint["semantic_status"]!="EXPLICIT_CONTRACT":
            result["reason"]="SEMANTICS_NOT_VERIFIED"
            return result
        predicate=constraint["predicate"]
        subject=self.resolve(predicate["subject_ref"],step)
        if not subject or "value" not in subject:
            result["reason"]="SUBJECT_UNKNOWN"
            return result
        records=[subject]
        target=predicate["target"]
        expected=copy.deepcopy(target.get("value"))
        if "ref" in target:
            prior=self.resolve(target["ref"],step,target.get("version"))
            if not prior or "value" not in prior:
                result["reason"]="TARGET_UNKNOWN"
                return result
            expected=copy.deepcopy(prior["value"])
            records.append(prior)
            if target.get("binding_mode","LIVE")=="SNAPSHOT" and target.get("version") is None:
                result["reason"]="SNAPSHOT_VERSION_REQUIRED"
                return result
        unit=predicate.get("unit",target.get("unit"))
        if unit and any(record.get("unit")!=unit for record in records):
            result["reason"]="UNIT_UNKNOWN_OR_MISMATCH"
            return result
        if len(records)>1 and any(record.get("unit") for record in records) and len({record.get("unit") for record in records})>1:
            result["reason"]="UNIT_UNKNOWN_OR_MISMATCH"
            return result
        if target.get("scale",1)!=1 or target.get("offset",0)!=0:
            base,scale,offset=number(expected),number(target.get("scale",1)),number(target.get("offset",0))
            if base is None or scale is None or offset is None:
                result["reason"]="NONNUMERIC_TRANSFORM"
                return result
            expected=transform(base,scale,offset)
            if expected is None:
                result["reason"]="ARITHMETIC_RANGE_UNVERIFIED"
                return result
        compared={"value":expected,"tolerance":target.get("tolerance")} if predicate["operator"]=="ABS_DIFF_LTE" else expected
        verdict=compare(subject["value"],predicate["operator"],compared)
        result.update(status="SATISFIED" if verdict is True else "VIOLATED" if verdict is False else "UNVERIFIED",
            operator=predicate["operator"],actual=copy.deepcopy(subject["value"]),expected=str(expected) if hasattr(expected,"as_tuple") else expected,
            constraint_version=constraint["version"],source=copy.deepcopy(constraint["source"]),
            bindings=[{"reference":record["reference"],"version":record["version"],"source":record["source"]} for record in records])
        self.check_history.append(copy.deepcopy(result))
        return result

    def frontier(self,references,step,token_counter,token_budget):
        """Render current constraints and their exact consumed fact versions."""
        if not callable(token_counter) or type(token_budget) is not int or token_budget<1:
            raise ValueError("Explicit tokenizer and positive budget required")
        def count(value):
            result=token_counter(value)
            if type(result) is not int or result<0: raise ValueError("Tokenizer must return a nonnegative integer")
            return result
        constraints,facts,unknown=[],{},[]
        def include_fact(reference,version=None):
            record=self.resolve(reference,step,version)
            if record is None or "value" not in record:
                unknown.append({"reference":reference,"version":version,"reason":"UNKNOWN"})
                return
            key=(reference,record["version"])
            if key in facts: return
            facts[key]=record
            entity=record.get("entity_ref")
            if entity: include_fact(entity["ref"],entity["version"])
        for reference in references:
            record=self.resolve(reference,step)
            if record is None: unknown.append({"reference":reference,"reason":"UNKNOWN"}); continue
            if "predicate" not in record:
                include_fact(reference,record["version"])
                continue
            constraints.append(record)
            if record["semantic_status"]!="EXPLICIT_CONTRACT":
                unknown.append({"reference":reference,"reason":"SEMANTICS_NOT_VERIFIED"})
            predicate=record["predicate"]
            targets=[(predicate["subject_ref"],None)]
            if "ref" in predicate["target"]: targets.append((predicate["target"]["ref"],predicate["target"].get("version")))
            for ref,version in targets:
                include_fact(ref,version)
        sections={"FACTS":list(facts.values()),"CONSTRAINTS":constraints,"UNKNOWN":unknown}
        text=serialize_sections(sections,reference_values=[*self.facts,*self.constraints,*self.events])
        tokens=count(text)
        if tokens>token_budget:
            notice="BUDGET_OVERFLOW"
            while notice and count(notice)>token_budget: notice=notice[:-1]
            return {"status":"BUDGET_OVERFLOW","context":notice,"required_tokens":tokens,"required_sections":sections}
        return {"status":"UNVERIFIED" if unknown else "OK","context":text,"token_count":tokens,"sections":sections}

    def snapshot(self):
        return copy.deepcopy({"scope_id":self.scope_id,"events":self.events,"facts":self.facts,
                              "constraints":self.constraints,"check_history":self.check_history})
