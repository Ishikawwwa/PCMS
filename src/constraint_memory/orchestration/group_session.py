"""Causal group execution, latest full submissions, and budgeted scoped repair."""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path

from constraint_memory.benchmark_adapter import PlannerWorker
from constraint_memory.benchmark_adapter.data import ChronologicalAdapter
from constraint_memory.benchmark_adapter.submissions import SubmissionLedger
from constraint_memory.extraction import extract,parse_submission,grounded_attributes
from constraint_memory.memory import create_backend
from constraint_memory.benchmark_adapter.property_schema import declared_price_schema,annotate_price_facts
from constraint_memory.polza import BudgetExceeded
from .logging import append_jsonl,write_json
from .contracts import INHERITANCE_CONTRACT_TEXT,inheritance_contract


class GroupSession:
    def __init__(self,root,run_dir,variant,group_id,config,readiness,tokenizer,client,planner_factory,base_formatter):
        self.root,self.run_dir=Path(root),Path(run_dir)
        self.variant,self.group_id=variant,str(group_id)
        self.config,self.readiness,self.tokenizer,self.client=config,readiness,tokenizer,client
        self.planner_factory=planner_factory or PlannerWorker
        self.base_formatter=base_formatter
        self.agent=config["agent"]
        self.scope=f"{variant}:{group_id}"
        self.client.set_scope(self.scope)
        self.record_start=len(client.records)
        self.started=time.perf_counter()
        self.usage=[]
        self.step=0
        self.observed_rows=[]
        self.travelers=[]
        self.budget_exhausted=None
        self.contract=None
        self.property_schema_event=None
        self.request_events={}
        self.memory_config={**config["memory"],"backend":variant,"token_budget":self.agent["max_agent_visible_memory_tokens"]}
        self.backend=create_backend(variant,self.memory_config)
        started=time.perf_counter()
        self.backend.reset(self.group_id)
        self.record_operation("memory_reset",time.perf_counter()-started)
        self.backend.set_token_counter(self.count_tokens,name="Qwen3.6-27B pinned tokenizer")
        self.log_memory("reset",self.group_id,config=self.memory_config)
        self.causal=ChronologicalAdapter(self.root,self.group_id)
        self.base=self.causal.base["base_person"]
        self.family=self.causal.base["base_template_id"]
        self.submissions=SubmissionLedger(self.group_id,self.base["name"])

    def count_tokens(self,text):
        return len(self.tokenizer.encode(text).ids)

    def log(self,name,value):
        append_jsonl(self.run_dir/name,{"variant":self.variant,"group_id":self.group_id,**value})

    def record_operation(self,operation,elapsed,category="maintenance",traveler=None):
        row={"group_id":self.scope,"traveler":traveler,"operation":operation,"category":category,
             "model_call":False,"input_tokens":0,"output_tokens":0,"total_tokens":0,"cost_rub":0,
             "currency":"RUB","latency_seconds":elapsed,"external_tool":category=="tool"}
        self.usage.append(row)
        append_jsonl(self.run_dir/"usage.jsonl",row)

    def log_memory(self,operation,arg=None,**extra):
        self.log("memory_ops.jsonl",{"op":operation,"arg":arg,**extra})

    def observe(self,kind,content):
        event={"event_id":f"{self.group_id}:{self.step}","group_id":self.group_id,"step":self.step,"type":kind,"content":content}
        self.step+=1
        started=time.perf_counter()
        self.backend.observe(event)
        self.record_operation("memory_observe",time.perf_counter()-started)
        self.log("events.jsonl",event)
        self.log_memory("observe",event)
        return event

    def begin(self,request):
        if self.config["memory"].get("public_base_outside_memory",True):
            request={**request,"public_anchor_travelers":[self.base["name"]]}
        started=time.perf_counter()
        context=self.backend.begin_subtask(request)
        self.record_operation("memory_read",time.perf_counter()-started,traveler=request["traveler"])
        self.log_memory("begin_subtask",request)
        self.log("retrievals.jsonl",{"traveler":request["traveler"],**context})
        return context

    def scoped_request(self,packet,step):
        request={"group_id":self.group_id,"traveler":packet["name"],
                 "round_idx":packet["traveler_index"],"step":step,"query":packet["query"]}
        if self.contract:
            request["source_refs"]=["event:"+self.request_events[packet["traveler_index"]]["event_id"]]
            request["prior_travelers"]=[self.base["name"],*[p["name"] for index,p in self.submissions.requests.items()
                if index<packet["traveler_index"]]]
        return request

    def commit(self,result):
        started=time.perf_counter()
        outcome=self.backend.commit_subtask(result)
        self.record_operation("memory_commit",time.perf_counter()-started,traveler=result["traveler"])
        self.log_memory("commit_subtask",result,result=outcome)
        self.step=max(self.step,self.backend.current_step+1)
        return outcome

    def initialize_base(self):
        self.client.set_scope(self.scope,self.base["name"])
        if self.variant in {"B0","B1","B2"} and self.config["memory"].get("public_base_outside_memory",True):
            return
        event=self.observe("request",self.base["query"])
        request={"group_id":self.group_id,"traveler":self.base["name"],"round_idx":0,"step":event["step"],"query":self.base["query"]}
        # The immutable base query and exact itinerary are already provided to
        # every planner. Store that public anchor deterministically rather than
        # asking a model to reinterpret its whole-trip requirements as slots.
        self.begin(request)
        action=self.base_formatter(self.base)
        self.commit({**request,"action":action,"decisions":parse_submission(action)})

    def collect_observations(self,result):
        by_id={}
        for trace in result.get("scratchpad",[]):
            if trace.get("thought"):
                self.observe("agent_action",trace["thought"])
            for call in trace.get("tool_calls") or []:
                self.observe("tool_call",json.dumps(call,ensure_ascii=False))
            for output in trace.get("tool_results") or []:
                by_id[output.get("tool_call_id")]=self.observe("tool_result",str(output.get("result","")))
        for output in result.get("structured_tool_results",[]):
            if output.get("latency_seconds") is not None:
                self.record_operation("tool_"+output["tool_name"],output["latency_seconds"],"tool",result["name"])
            event=by_id.get(output.get("tool_call_id"))
            if not event or event["content"]!=output.get("formatted_result"):
                raise ValueError("Structured observation differs from the official visible tool result")
            self.observed_rows.extend({"event_id":event["event_id"],"tool_name":output["tool_name"],"row":row,
                                       "source_content":event["content"]} for row in output.get("displayed_rows",[]))

    def pending(self):
        return self.backend.pending_recoveries(self.step) if hasattr(self.backend,"pending_recoveries") else []

    def acknowledge(self,pending,committed,steps_used,charged_round,attempt):
        current={}
        for record in self.backend.snapshot().get("decisions",[]):
            previous=current.get(record["decision_id"])
            if previous is None or record["version"]>previous["version"]:
                current[record["decision_id"]]=record
        for recovery in pending:
            affected=set(recovery["affected_decisions"])
            refs=[f"{did}@v{current[did]['version']}" for did in committed.get("stored_ids",{}).get("decisions",[])
                  if did in affected and did in current and current[did]["status"]=="COMMITTED"]
            ack={"group_id":self.group_id,"step":self.step,"repaired_action_event_ids":[committed["action_event_id"]],
                 "repaired_decision_refs":refs,"revision_action_cost":steps_used,
                 "acknowledgment_id":f"round:{charged_round}:attempt:{attempt}:{recovery['recovery_id']}"}
            started=time.perf_counter()
            result=self.backend.acknowledge_recovery(recovery["recovery_id"],ack)
            self.record_operation("memory_recovery_ack",time.perf_counter()-started)
            self.log_memory("acknowledge_recovery",args=[recovery["recovery_id"],ack],result=result)
            self.log("recoveries.jsonl",{"charged_round":charged_round,"attempt":attempt,"acknowledgment":ack,"result":result})

    def attempt(self,packet,request,context,charged_round,attempt,remaining):
        self.client.set_scope(self.scope,packet["name"])
        pending_before=self.pending()
        denied=context["status"] in {"BUDGET_OVERFLOW","VALIDATION_FAILED"} or self.count_tokens(context["context"])>self.agent["max_agent_visible_memory_tokens"]
        if context["status"]=="RECOVERY_REQUIRED" and attempt==1:
            denied=True
        if denied or self.budget_exhausted:
            result={"final_plan":"","plan":None,"scratchpad":[],"total_steps":0,"success":False,
                    "error_message":self.budget_exhausted or "MEMORY_BOUNDARY_FAILURE: "+context["status"]}
        else:
            planner=self.planner_factory(self.root/self.config["experiment"]["upstream_root"],
                self.root/self.readiness["database"]["database_path"],self.client,model=self.agent["model"],
                temperature=self.agent["temperature"],max_steps=remaining,timeout_seconds=3600,
                denied_paths=[self.root/"runs",self.root/"reports",self.root/"tests",self.run_dir],
                visible_travelers=self.submissions.visible_travelers,causal_round=charged_round,
                task_contract_text=self.contract["text"] if self.contract else None,
                tool_city_state_annotation_normalization=self.config["experiment"].get("tool_city_state_annotation_normalization",False),
                tool_price_unit=self.config["experiment"].get("tool_price_unit"))
            try:
                result=planner.run_turn(packet,context["context"],self.base)
            except BudgetExceeded as error:
                self.budget_exhausted=str(error)
                result={"final_plan":"","plan":None,"scratchpad":[],"total_steps":getattr(planner,"model_requests_sent",0),
                        "parent_model_requests":getattr(planner,"model_requests_sent",0),"success":False,"error_message":str(error)}
        used=result.get("parent_model_requests",result.get("total_steps",0))
        if not isinstance(used,int) or isinstance(used,bool) or not 0<=used<=remaining:
            raise ValueError("Planner dispatch count exceeded its current-arrival step allowance")
        result={**result,"group_id":self.group_id,"traveler_index":packet["traveler_index"],"name":packet["name"],
                "source_round":charged_round,"source_attempt":attempt}
        if not result.get("submitted_text") or not result.get("plan"):
            result={**result,"success":False,"error_message":result.get("error_message") or "MISSING_NAMED_SUBMISSION: "+packet["name"]}
        self.collect_observations(result)
        accepted=self.submissions.accept(result)
        action="\n\n".join(item["final_plan"] for item in accepted["accepted"])
        decisions=parse_submission(action)
        replaced={item["name"] for item in accepted["accepted"]}
        previous=self.backend.committed_decisions(self.step) if hasattr(self.backend,"committed_decisions") else []
        selected=[r for r in previous if r.get("attribute","value")=="value" and r["traveler"] not in replaced]+decisions
        attributes=grounded_attributes(selected,self.observed_rows) if self.variant.startswith("E") else {}
        attributes=annotate_price_facts(attributes,self.property_schema_event)
        previous_facts={(r["traveler"],r["day"],r["slot"],r.get("attribute","value")):r for r in previous}
        attributes["decisions"]=[r for r in attributes.get("decisions",[]) if r["traveler"] in replaced or
            previous_facts.get((r["traveler"],r["day"],r["slot"],r["attribute"]),{}).get("value")!=r["value"]]
        decisions+=attributes.get("decisions",[])
        committed=self.commit({**request,"step":self.step,"action":action,"decisions":decisions,
            "replace_travelers":[item["name"] for item in accepted["accepted"]],
            "extraction":{"definitions":attributes.get("definitions",[])},"remaining_action_budget":remaining-used,
            "status":"UNCERTAIN" if result["success"] else "FAILED"})
        self.log("submission_events.jsonl",{"charged_round":charged_round,"subject_traveler_index":packet["traveler_index"],
            "attempt":attempt,"accepted":accepted,"steps_used":used,"steps_remaining":remaining-used})
        self.log("traces.jsonl",{"traveler_index":packet["traveler_index"],"charged_round":charged_round,
                                 "attempt":attempt,"result":result,"commit":committed})
        if attempt>1 and pending_before:
            self.acknowledge(pending_before,committed,used,charged_round,attempt)
        return used,result,committed

    def repair_target(self):
        if not hasattr(self.backend,"required_repair_travelers"):
            return None
        names=self.backend.required_repair_travelers()
        candidates=[packet for packet in self.submissions.requests.values() if packet["name"] in names]
        return min(candidates,key=lambda packet:packet["traveler_index"]) if candidates else None

    def run_arrival(self,packet):
        index=packet["traveler_index"]
        self.submissions.register_request(packet)
        self.client.set_scope(self.scope,packet["name"])
        event=self.observe("request",packet["query"])
        self.request_events[index]=event
        request=self.scoped_request(packet,event["step"])
        try:
            extracted=extract(self.client,request,[event],self.config["memory"]["parser"]) if self.variant.startswith("E") and not self.budget_exhausted else {}
        except BudgetExceeded as error:
            self.budget_exhausted=str(error)
            extracted={"unresolved":[{"reason":"GROUP_RESOURCE_BUDGET_EXHAUSTED"}]}
        self.log("extractions.jsonl",{"traveler":packet["name"],"extraction":extracted})
        if self.config["experiment"].get("stop_on_extraction_failure") and extracted.get("parse_success") is False:
            raise RuntimeError("DEVELOPMENT_WRITER_READINESS_GATE: extraction failed; no benchmark comparison is valid")
        context=self.begin({**request,"extraction":extracted})
        initial_status=context["status"]
        remaining=self.agent["max_steps_per_traveler"]
        attempts=0
        limit=int(self.config["memory"].get("recovery_max_attempts",2))
        target=packet
        terminal="no_recovery_required"
        previous_failed_scope=None
        previous_inheritance_failure=None
        while True:
            attempts+=1
            used,result,committed=self.attempt(target,request,context,index,attempts,remaining)
            remaining-=used
            inheritance_check=committed.get("inheritance_check")
            if inheritance_check is not None:
                self.log("inheritance_checks.jsonl",{"charged_round":index,"attempt":attempts,"check":inheritance_check})
            request_check=committed.get("request_check")
            if request_check is not None:
                self.log("request_checks.jsonl",{"charged_round":index,"attempt":attempts,"check":request_check})
            failed_checks=[(name,check) for name,check in (("inheritance",inheritance_check),("request",request_check))
                           if check and check["status"]=="VIOLATED"]
            if failed_checks:
                # Only source-certified current-person predicates can request
                # this repair. Policy/cuisine completeness remains unverified.
                # Proof versions do not make an otherwise unchanged bad action
                # look like progress; compare the actual failing predicates.
                failed=json.dumps([(name,[{key:row.get(key) for key in (
                    "code","subject","operator","expected","actual","observed_full_name")}
                    for row in check["violations"]]) for name,check in failed_checks],sort_keys=True,ensure_ascii=False)
                if self.budget_exhausted or remaining<=0 or attempts>=limit+1:
                    terminal="resource_budget_exhausted" if self.budget_exhausted else "step_budget_exhausted" if remaining<=0 else "constraint_repair_attempt_cap"
                    break
                if attempts>1 and failed==previous_inheritance_failure:
                    terminal="unchanged_inheritance_failure" if all(name=="inheritance" for name,_ in failed_checks) else "unchanged_request_failure"
                    break
                previous_inheritance_failure=failed
                target=packet
                request=self.scoped_request(packet,self.step)
                context=self.begin(request)
                if context["status"] in {"BUDGET_OVERFLOW","VALIDATION_FAILED"}:
                    terminal="nonrepairable_memory_boundary: "+context["status"]
                    break
                continue
            pending=self.pending()
            if not pending:
                current=self.submissions.latest.get(index)
                if attempts>1 and (not current or not current.get("plan")) and remaining>0 and attempts<limit+1 and target["traveler_index"]!=index:
                    target=packet
                    request=self.scoped_request(packet,self.step)
                    context=self.begin(request)
                    continue
                terminal="resolved_unverified" if attempts>1 else "no_recovery_required"
                break
            if not self.backend.features.get("targeted_recovery",False):
                terminal="recovery_disabled"
                break
            failed_scope=(tuple(sorted({did for recovery in pending for did in recovery["affected_decisions"]})),
                          tuple((prediction["traveler_index"],json.dumps(prediction.get("plan"),sort_keys=True))
                                for prediction in self.submissions.predictions()))
            if attempts>1 and remaining>0 and not self.budget_exhausted and failed_scope==previous_failed_scope:
                terminal="unchanged_failed_scope"
                break
            previous_failed_scope=failed_scope
            if self.budget_exhausted or remaining<=0 or attempts>=limit+1:
                terminal="resource_budget_exhausted" if self.budget_exhausted else "step_budget_exhausted" if remaining<=0 else "repair_attempt_cap"
                break
            target=self.repair_target()
            if target is None:
                terminal="no_visible_repair_target"
                break
            request=self.scoped_request(target,self.step)
            context=self.begin(request)
            if context["status"] in {"BUDGET_OVERFLOW","VALIDATION_FAILED"}:
                terminal="nonrepairable_memory_boundary: "+context["status"]
                break
        current=next(p for p in self.submissions.predictions() if p["traveler_index"]==index)
        self.causal.commit_result(current)
        self.travelers.append({"traveler_index":index-1,"name":packet["name"],"status":initial_status,
            "steps":self.agent["max_steps_per_traveler"]-remaining,"repair_attempts":max(0,attempts-1),"recovery_terminal":terminal})
        self.log("recoveries.jsonl",{"charged_round":index,"terminal":terminal,"steps_used":self.agent["max_steps_per_traveler"]-remaining,
            "steps_remaining":remaining,"repair_attempts":max(0,attempts-1),"pending_recovery_ids":[r["recovery_id"] for r in self.pending()]})
        write_json(self.run_dir/f"snapshots/{self.variant}/{self.group_id}.json",self.backend.snapshot())

    def run(self):
        unit=self.config["experiment"].get("tool_price_unit")
        if unit is not None:
            self.property_schema_event=self.observe("property_schema",declared_price_schema(unit))
        if self.config["experiment"].get("declared_inheritance_contract",False):
            event=self.observe("task_contract",INHERITANCE_CONTRACT_TEXT)
            self.contract=inheritance_contract(self.group_id,event)
        self.initialize_base()
        if hasattr(self.backend,"configure_inheritance"):
            if self.contract is None:
                raise ValueError("Candidate needs the declared shared inheritance contract")
            self.backend.configure_inheritance(self.contract,self.base["name"])
            self.log_memory("configure_inheritance",args=[self.contract,self.base["name"]])
        for packet in self.causal.iter_requests():
            self.run_arrival(packet)
        predictions=self.submissions.predictions(self.variant,self.family)
        for prediction in predictions:
            append_jsonl(self.run_dir/"predictions.jsonl",prediction)
        stats=self.backend.stats()
        snapshot=self.backend.snapshot()
        count=self.causal.metadata["traveler_count"]
        row={"group_id":self.group_id,"base_template_id":self.family,"variant":self.variant,
            "replicate_id":self.config["experiment"]["seed"],"traveler_count":count,"person_count":count,
            "usage":copy.deepcopy(self.client.records[self.record_start:])+self.usage,"travelers":self.travelers,
            "wall_clock_seconds":time.perf_counter()-self.started,"memory_stats":stats,"storage_bytes":stats.get("storage_bytes"),
            "index_bytes":len(json.dumps(snapshot["retrieval_index"],ensure_ascii=False).encode()) if "retrieval_index" in snapshot else None,
            "record_count":sum(len(snapshot.get(key,[])) for key in ("evidence","tasks","constraints","decisions","definitions","outcomes"))}
        return predictions,row
