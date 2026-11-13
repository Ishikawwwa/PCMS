"""E3R: source-certified request predicates and observed-name repair.

The model remains the planner. Unsupported semantics and unavailable evidence
stay unverified, and all actual submissions remain in the versioned archive.
"""
from __future__ import annotations

import copy
import hashlib

from .inherited_frontier import InheritedFrontierMemory
from .choice_identity import validate_choice_identity
from .request_checks import compile_request_checks,validate_request_checks


class VerifiedFrontierMemory(InheritedFrontierMemory):
    def __init__(self,config=None):
        super().__init__(config)
        self.variant="E3R"
        self.features.update(independent_request_predicates=True,observed_full_name_check=True)

    def reset(self,group_id):
        super().reset(group_id)
        self._request_predicates={}
        self._request_predicate_archive=[]
        self._request_checks={}
        self._request_check_history=[]
        self._current_request_repair=None
        self._current_repair_choices=[]

    def begin_subtask(self,request):
        self._current_request_repair=None
        self._current_repair_choices=[]
        context=super().begin_subtask(request)
        compiled=self._request_predicates.get(request["traveler"])
        if compiled is not None:
            context["request_predicates_certified"]=len(compiled.get("predicates",[]))
        self._last_context=copy.deepcopy(context)
        return context

    def _compile_current(self,request,step):
        inherited=super()._compile_current(request,step)
        traveler=request["traveler"]
        pool=[{**row,"group_id":self.group_id} for row in self.committed_decisions(step)]
        if self._inheritance_contract is not None and traveler!=self._base_traveler:
            if inherited.get("causal_request_verified"):
                compiled=compile_request_checks(request,pool)
            else:
                compiled={"status":"UNVERIFIED","group_id":self.group_id,"traveler":traveler,
                          "step":request["step"],"predicates":[],"diagnostics":[{"code":"UNVERIFIED_REQUEST_SOURCE"}]}
            self._request_predicates[traveler]=copy.deepcopy(compiled)
            self._request_predicate_archive.append(copy.deepcopy(compiled))
            previous=self._request_checks.get(traveler)
            query_hash=hashlib.sha256(request.get("query","").encode("utf-8")).hexdigest()
            if (previous and previous["step"]<=step and previous["request_sha256"]==query_hash
                    and previous["check"]["status"]=="VIOLATED"):
                # Repair evidence is re-evaluated against the current causal
                # bindings before it enters the planner's bounded context.
                check=self._evaluate_request(request,inherited,compiled,pool,step)
                if check["status"]=="VIOLATED":
                    self._current_request_repair=check
                    slots={(row["day"],row["slot"]) for row in inherited.get("exceptions",[])}
                    self._current_repair_choices=[{
                        "record_ref":f"{row['decision_id']}@v{row['version']}","value":copy.deepcopy(row["value"]),
                        "source_refs":copy.deepcopy(row["source_refs"])} for row in self._current_values(traveler,step)
                        if (row["day"],row["slot"]) in slots]
        return inherited

    def _evaluate_request(self,request,inherited,compiled,pool,step):
        numeric=(validate_request_checks(compiled,pool,step=step) if compiled is not None else
                 {"status":"UNVERIFIED","violations":[],"task_success":None})
        identity=(validate_choice_identity(request,pool,list(self.events.values()),
                    inherited.get("exceptions",[]),step) if inherited.get("causal_request_verified") else
                  {"status":"UNVERIFIED","violations":[],"task_success":None})
        violations=[{**copy.deepcopy(row),"check_kind":kind}
                    for kind,check in (("request_predicate",numeric),("observed_full_name",identity))
                    for row in check["violations"]]
        return {"status":"VIOLATED" if violations else "CHECKED_UNVERIFIED_TASK",
                "task_success":None,"violations":violations,
                "predicate_check":numeric,"identity_check":identity}

    def _sections(self,records,definitions,outcomes,compact=False):
        sections=super()._sections(records,definitions,outcomes,compact)
        if self._current_request_repair:
            sections["VERIFIED REQUEST REPAIR"]=copy.deepcopy(self._current_request_repair["violations"])
            sections["CURRENT REQUEST CHOICES"]=copy.deepcopy(self._current_repair_choices)
        return sections

    def commit_subtask(self,result):
        committed=super().commit_subtask(result)
        traveler=self._request.get("traveler")
        if self._inheritance_contract is None or traveler==self._base_traveler:
            return committed
        step=self.events[committed["action_event_id"]]["step"]
        pool=[{**row,"group_id":self.group_id} for row in self.committed_decisions(step)]
        inherited=self._inheritance_compiled.get(traveler,{})
        compiled=self._request_predicates.get(traveler)
        check=self._evaluate_request(self._request,inherited,compiled,pool,step)
        saved={"traveler":traveler,"step":step,"action_event_id":committed["action_event_id"],
               "request_sha256":hashlib.sha256(self._request.get("query","").encode("utf-8")).hexdigest(),
               "check":copy.deepcopy(check)}
        self._request_checks[traveler]=copy.deepcopy(saved)
        self._request_check_history.append(saved)
        committed["request_check"]=copy.deepcopy(check)
        return committed

    def snapshot(self):
        result=super().snapshot()
        result["request_checks"]={"compiled":copy.deepcopy(self._request_predicates),
            "compiler_archive":copy.deepcopy(self._request_predicate_archive),
            "latest_checks":copy.deepcopy(self._request_checks),
            "check_history":copy.deepcopy(self._request_check_history)}
        return result
