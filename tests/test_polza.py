import json
import pytest
from constraint_memory.polza import PolzaClient, ProviderError, BudgetExceeded
from constraint_memory.extraction import parse_submission, extract
from constraint_memory.secrets import redact
from constraint_memory.config import load_config


def response(cost=0.3):
    return {"id":"r1", "model":"qwen/qwen3.6-27b", "provider":"fixed",
            "choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost_rub":cost}}


def test_provider_rub_and_separate_extraction_usage(tmp_path):
    client=PolzaClient(api_key="test",transport=lambda *args: response(),usage_path=tmp_path/"usage.jsonl")
    client.set_scope("g1","Bob")
    client.complete([{"role":"user","content":"hello"}],operation="extraction")
    row=json.loads((tmp_path/"usage.jsonl").read_text())
    assert row["category"]=="maintenance" and row["cost_rub"]==0.3
    assert client.get_usage_stats()["total_cost_rub"]==0.3


def test_missing_usage_and_unknown_cost_fail_closed():
    client=PolzaClient(api_key="test",transport=lambda *args: {"choices":[{"message":{}}]})
    with pytest.raises(ProviderError): client.complete([])
    client=PolzaClient(api_key="test",transport=lambda *args: response(None))
    client.complete([])
    with pytest.raises(BudgetExceeded): client.complete([])


def test_token_budget_prevents_call():
    client=PolzaClient(api_key="test",max_total_tokens=5,transport=lambda *args: pytest.fail("must not call"))
    with pytest.raises(BudgetExceeded): client.complete([])


def test_key_redaction():
    assert redact({"api_key":"secret","message":"pza_fake123"})=={"api_key":"[REDACTED]","message":"[REDACTED]"}


def test_exact_submitted_plan_and_revisions():
    decisions=parse_submission("=== Alice's Plan ===\nDay 2:\nDinner: Actual choice, City\n=== Bob's Plan ===\nDay 1:\nAccommodation: Hotel A, City")
    assert decisions[0]["value"]=="Actual choice, City"
    assert decisions[1]["traveler"]=="Bob" and decisions[1]["slot"]=="accommodation"


def test_extractor_rejects_fabricated_evidence():
    raw=response(); raw["choices"][0]["message"]["content"]=json.dumps({"constraints":[{"source_refs":["event:future"],"evidence_quote":"secret"}]})
    client=PolzaClient(api_key="test",transport=lambda *args: raw)
    result=extract(client,{"traveler":"Bob"},[{"event_id":"request","content":"Bob joins Alice"}])
    assert not result["constraints"] and result["unresolved"]


def test_fixed_target_model_config():
    assert load_config("configs/travel.yaml")["agent"]["model"]=="qwen/qwen3.6-27b"


def test_writer_reasoning_setting_does_not_change_the_backbone_or_planner_setting():
    payloads=[]
    def transport(method,path,payload):
        payloads.append(payload)
        return response()
    client=PolzaClient(api_key="offline",transport=transport,reasoning={"enabled":True},extraction_reasoning={"enabled":False})
    client.complete([],operation="extraction")
    client.complete([],operation="agent")
    assert [p["model"] for p in payloads]==["qwen/qwen3.6-27b"]*2
    assert [p["reasoning"] for p in payloads]==[{"enabled":False},{"enabled":True}]


def test_separate_writer_output_cap_is_charged_without_changing_planner_cap(tmp_path):
    payloads=[]
    def transport(method,path,payload): payloads.append(payload); return response()
    client=PolzaClient(api_key="offline",transport=transport,max_tokens=8192,extraction_max_tokens=16384,usage_path=tmp_path/"usage.jsonl")
    client.complete([],operation="extraction")
    client.complete([],operation="agent")
    assert [p["max_tokens"] for p in payloads]==[16384,8192]
    assert (tmp_path/"writer_responses.jsonl").exists()
    assert len(client.records)==2


def test_development_writer_gate_is_forbidden_on_frozen_holdout():
    config=load_config("configs/travel.yaml")
    config["experiment"].update(partition="holdout",final_holdout_frozen=True,stop_on_extraction_failure=True)
    from constraint_memory.config import validate_config
    with pytest.raises(ValueError,match="development-only"): validate_config(config)
