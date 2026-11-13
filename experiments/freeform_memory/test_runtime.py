"""Mechanical runtime checks, never a claim of language or model performance."""
import copy
import json

import pytest

from experiments.freeform_memory.runtime import (
    build_observed,writer_messages,ingest_interpretation,prepare_context,
    planner_messages,commit_option,
)


def task():
    return {"case_id":"unfamiliar-project","domain":"purchase","format":"email",
        "query":"Choose one current offer under the messages above.",
        "events":[{"event_id":"initial","kind":"observation","step":0,
            "content":{"authorization":{"amount":120}},
            "facts":[{"ref":"authorized-limit","pointer":"/authorization/amount","unit":"EUR"}]},
            {"event_id":"message","kind":"user","step":1,
             "content":"Please keep the total at or below our current authorization."},
            {"event_id":"revision","kind":"observation","step":2,
             "content":{"limit":90},"facts":[{"ref":"authorized-limit","pointer":"/limit","unit":"EUR"}]},
            {"event_id":"catalog","kind":"observation","step":3,
             "content":{"offers":[{"option_id":"a7","total":80},{"option_id":"b9","total":110}]}}],
        "options":[{"option_id":identity,"event_id":"catalog","pointer":f"/offers/{index}",
            "properties":[{"ref":"proposed-total","pointer":f"/offers/{index}/total","unit":"EUR"}]}
            for index,identity in enumerate(("a7","b9"))]}


def interpretation():
    return {"constraints":[{"source_event_id":"message",
        "evidence_quote":"keep the total at or below our current authorization",
        "subject_ref":"proposed-total","operator":"LTE","target":{"ref":"authorized-limit"},"unit":"EUR"}],
        "unresolved":[]}


def test_stronger_ledger_and_ir_receive_identical_raw_requirements_and_factual_state():
    case=task()
    before=copy.deepcopy(case)
    ledger=prepare_context(case,"FACT_LEDGER",len,100000)
    ir=prepare_context(case,"ADVISORY_IR",len,100000,interpretation())
    for field in ("RAW_USER_REQUIREMENTS","CURRENT_FACTS","UNKNOWN_FACTS","FACT_REVISIONS"):
        assert ledger["sections"][field]==ir["sections"][field]
    assert ledger["sections"]["CURRENT_FACTS"][0]["value"]==90
    assert [row["value"] for row in ledger["sections"]["FACT_REVISIONS"]["authorized-limit"]]==[120,90]
    assert ir["hard_semantic_checks"] is False and ir["task_success"] is None
    assert case==before


def test_supported_looking_model_predicate_remains_advisory_with_original_quote():
    case=task()
    memory,step,_=build_observed(case)
    result=ingest_interpretation(memory,case,interpretation(),step+1)
    assert not result["rejected"]
    record=memory.resolve(result["references"][0],step+1)
    assert record["semantic_status"]=="UNVERIFIED"
    assert record["source"]["quote"]==interpretation()["constraints"][0]["evidence_quote"]
    assert memory.check(result["references"][0],step+1)["reason"]=="SEMANTICS_NOT_VERIFIED"


@pytest.mark.parametrize("change",[
    {"source_event_id":"catalog"},{"source_event_id":"future"},{"evidence_quote":"invented requirement"},
    {"subject_ref":"hidden-output"},{"operator":"SOLVE_TRAVEL"},{"verified":True},
    {"target":{"ref":"hidden-gold"}},{"target":{"ref":"authorized-limit","scale":True}},
    {"target":{"ref":"authorized-limit","binding_mode":"LIVE","version":1}},
    {"unit":"USD"},
])
def test_rejected_or_advisory_interpretations_never_make_false_semantic_claims(change):
    case=task()
    response=interpretation()
    response["constraints"][0].update(change)
    context=prepare_context(case,"ADVISORY_IR",len,100000,response)
    assert context["status"]=="OK" and context["hard_semantic_checks"] is False
    for row in context["sections"].get("ADVISORY_CONSTRAINTS",[]):
        assert row["semantic_status"]=="UNVERIFIED"
    assert context["sections"]["RAW_USER_REQUIREMENTS"]
    # Even a semantically wrong but structurally valid unit is not certified.
    if change!={"unit":"USD"}: assert context["interpretation"]["rejected"]


@pytest.mark.parametrize("raw",[None,{},[],{"constraints":[],"unresolved":[],"task_success":True}])
def test_malformed_or_missing_writer_falls_back_to_same_exact_ledger(raw):
    case=task()
    plain=prepare_context(case,"FACT_LEDGER",len,100000)
    ir=prepare_context(case,"ADVISORY_IR",len,100000,raw)
    assert ir["context"]==plain["context"]
    assert "fallback" in ir["writer_status"]


@pytest.mark.parametrize("condition",["FACT_LEDGER","ADVISORY_IR"])
def test_context_cap_never_drops_user_rules_or_fact_versions_silently(condition):
    case=task()
    full=prepare_context(case,condition,len,100000,interpretation())
    small=prepare_context(case,condition,len,10,interpretation())
    assert small["status"]=="BUDGET_OVERFLOW" and len(small["context"])<=10
    assert small["sections"]==full["sections"] and small["required_tokens"]==full["required_tokens"]


@pytest.mark.parametrize("text",[
    "Keep at least ten minutes between meetings unless the organizer explicitly waives it.",
    "Release note:\nDo not enable tracing on the legacy host.\nSigned, ops",
    "Prefer the old approved cap, not yesterday's proposed increase.",
    "If the returned quantity is unknown, wait for the count before allocating stock.",
])
def test_free_form_requests_survive_without_a_domain_parser_or_semantic_oracle(text):
    case=task()
    case["events"][1]["content"]=text
    case["query"]=text
    context=prepare_context(case,"FACT_LEDGER",len,100000)
    assert context["sections"]["RAW_USER_REQUIREMENTS"][0]["content"]==text
    assert context["sections"]["RAW_USER_REQUIREMENTS"][-1]["content"]==text
    assert context["task_success"] is None
    messages=writer_messages(case)
    assert json.loads(messages[1]["content"])["raw_user_messages"][-1]["content"]==text


def test_writer_sees_raw_facts_and_symbols_without_offline_expected_answers():
    messages=writer_messages(task())
    data=json.loads(messages[1]["content"])
    assert data["output_fields"]==["proposed-total"]
    assert data["fact_histories"]["authorized-limit"][-1]["value"]==90
    assert not {"semantic_predicates","expected_options","expected_abstention","annotations"}.intersection(data)
    assert "offers" not in data


def test_actual_option_is_grounded_from_its_row_and_does_not_become_task_success():
    case=task()
    selected=commit_option(case,{"option_id":"b9"})
    assert selected["status"]=="COMMITTED_UNVERIFIED" and selected["task_success"] is None
    cost=selected["selected_facts"]["proposed-total"]
    assert cost["value"]==110 and cost["unit"]=="EUR"
    assert cost["source"]["pointer"]=="/offers/1/total"
    assert cost["entity_ref"]=={"ref":"actual-selected-option","version":1}
    assert case==task()


@pytest.mark.parametrize("response,status",[({"option_id":None},"ABSTAINED"),
    ({"option_id":"unknown"},"UNKNOWN_OPTION"),({"option_id":"a7","passed":True},"INVALID_SUBMISSION"),
    (None,"INVALID_SUBMISSION")])
def test_invalid_or_abstained_choices_are_actual_events_not_filled_in(response,status):
    result=commit_option(task(),response)
    assert result["status"]==status and not result.get("selected_facts")


def test_offers_cannot_borrow_other_row_properties_or_overwrite_antecedents():
    case=task()
    case["options"][0]["properties"][0]["pointer"]="/offers/1/total"
    with pytest.raises(ValueError,match="inside"): build_observed(case)
    case=task()
    case["options"][0]["properties"][0]["ref"]="authorized-limit"
    with pytest.raises(ValueError,match="shadow"): build_observed(case)


def test_structured_oracle_fields_and_task_contract_events_are_not_runtime_inputs():
    case=task()
    case["semantic_predicates"]=[]
    with pytest.raises(ValueError,match="inputs only"): build_observed(case)
    case=task()
    case["events"][1]["kind"]="task_contract"
    with pytest.raises(ValueError,match="interpreted"): build_observed(case)


def test_planner_receives_only_current_offers_and_audited_memory():
    case=task()
    context=prepare_context(case,"FACT_LEDGER",len,100000)
    messages=planner_messages(case,context)
    data=json.loads(messages[1]["content"])
    assert set(data)=={"request","offers","option_property_bindings","memory"}
    assert [row["option_id"] for row in data["offers"]]==["a7","b9"]
