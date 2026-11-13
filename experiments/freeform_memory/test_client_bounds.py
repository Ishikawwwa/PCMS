"""Exercise the real client locally; large receipts are invented test data."""
import json
import math

import pytest

from constraint_memory.polza import PolzaClient,BudgetExceeded


def client(cost,cap=None):
    payloads=[]
    def transport(method,path,payload):
        payloads.append(payload)
        return {"id":"offline-"+str(len(payloads)),"model":"qwen/qwen3.6-27b","provider":"offline-test",
            "choices":[{"message":{"content":"{}"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":600000,"completion_tokens":1,"total_tokens":600001,"cost_rub":cost}}
    result=PolzaClient(api_key="offline-only",transport=transport,max_total_tokens=math.inf,
                       max_cost_rub=cap,max_tokens=8192)
    result.set_scope("fixed-work-test")
    return result,payloads


def test_real_client_has_no_implicit_money_or_aggregate_token_stop_when_disabled():
    actual,payloads=client(10000)
    with actual:
        actual.complete([{"role":"user","content":"local test"}])
        actual.complete([{"role":"user","content":"second local test"}])
        assert actual.get_usage_stats()["total_cost_rub"]==20000
        assert actual.get_usage_stats()["calls"]==2
    assert len(payloads)==2 and all(row["max_tokens"]==8192 for row in payloads)
    # The disabled internal threshold is never sent as invalid API JSON.
    for row in payloads: json.dumps(row,allow_nan=False)


def test_an_explicit_opt_in_money_stop_still_applies():
    actual,payloads=client(10000,cap=10)
    with actual:
        actual.complete([])
        with pytest.raises(BudgetExceeded,match="RUB"): actual.complete([])
    assert len(payloads)==1


def test_unknown_cost_is_reported_as_unknown_in_an_uncapped_fixed_work_run():
    actual,payloads=client(None)
    with actual:
        actual.complete([])
        actual.complete([])
        assert actual.get_usage_stats()["total_cost_rub"] is None
        assert actual.get_usage_stats()["calls"]==2
    assert len(payloads)==2
