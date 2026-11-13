"""Network-free tests of accounting and reservation failures found in review."""

import pytest

from constraint_memory.polza import BudgetExceeded, PolzaClient, ProviderError


def completion(usage=None):
    return {"id": "offline-fixture", "model": "qwen/qwen3.6-27b",
            "choices": [{"message": {"content": "scripted"}, "finish_reason": "stop"}],
            "usage": usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 5,
                                                       "total_tokens": 15, "cost_rub": .1}}


def test_tool_schemas_are_reserved_before_a_paid_request():
    client = PolzaClient(api_key="offline", max_tokens=10, max_total_tokens=200,
                        transport=lambda *args: pytest.fail("Over-budget tool schemas must prevent the request"))
    tools = [{"type": "function", "function": {"name": "search",
        "description": "documented arguments " * 80, "parameters": {"type": "object"}}}]
    with pytest.raises(BudgetExceeded):
        client.complete([], tools=tools)


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": -10, "completion_tokens": 5, "total_tokens": -5, "cost_rub": .1},
    {"prompt_tokens": 10.9, "completion_tokens": 5, "total_tokens": 15.9, "cost_rub": .1},
    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 1, "cost_rub": .1},
    {"prompt_tokens": None, "completion_tokens": 5, "cost_rub": .1},
    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_rub": -1},
    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_rub": float("nan")},
])
def test_invalid_provider_usage_cannot_silently_understate_spending(usage):
    client = PolzaClient(api_key="offline", transport=lambda *args: completion(usage))
    with pytest.raises(ProviderError):
        client.complete([])


def test_tool_chat_respects_explicit_smaller_completion_limit():
    sent = []
    def transport(method, path, payload):
        sent.append(payload)
        return completion()
    client = PolzaClient(api_key="offline", max_tokens=500, transport=transport)
    client.chat_with_tools([], [], max_tokens=50)
    assert sent[0]["max_tokens"] == 50


def test_accounting_includes_extraction_and_does_not_carry_other_groups_into_cap():
    client = PolzaClient(api_key="offline", max_tokens=5, max_total_tokens=150,
                         transport=lambda *args: completion())
    client.set_scope("E0:one")
    client.complete([], operation="extraction")
    client.complete([], operation="agent")
    with pytest.raises(BudgetExceeded):
        client.complete([])
    client.set_scope("E0:two")
    client.complete([])
    assert sum(record["total_tokens"] for record in client.records if record["group_id"] == "E0:one") == 30
    assert client.records[-1]["group_id"] == "E0:two"
