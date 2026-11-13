import pytest
from constraint_memory.polza import PolzaClient,ProviderError,BudgetExceeded


def test_ambiguous_paid_failure_remains_unknown_and_blocks_further_group_usage():
    def failed(*args): raise ProviderError("ambiguous response timeout",request_sent=True)
    client=PolzaClient(api_key="offline",transport=failed)
    client.set_scope("E3:g")
    with pytest.raises(ProviderError): client.complete([])
    assert client.records[0]["cost_rub"] is None and client.records[0]["total_tokens"] is None
    assert client.get_usage_stats()["total_cost_rub"] is None
    with pytest.raises(BudgetExceeded): client.complete([])


def test_proven_pre_request_failure_has_zero_model_usage_and_no_secret_echo():
    def failed(*args): raise ProviderError("connect failed before send",request_sent=False)
    client=PolzaClient(api_key="offline",transport=failed)
    with pytest.raises(ProviderError): client.complete([])
    assert client.records[0]["cost_rub"]==0 and client.records[0]["total_tokens"]==0
