"""Every accepted inference uses the exact configured returned model identity."""
import copy
import io
import json

import pytest

from constraint_memory.polza import BudgetExceeded, PolzaClient, ProviderError
from constraint_memory.streaming import read_completion_stream


MODEL = "qwen/qwen3.6-27b"
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_rub": 0.3}


def completion(**changes):
    return {"id": "gen_123456", "model": MODEL, "provider": "Chutes",
            "choices": [{"message": {"content": "SYNTHETIC_REJECTED_CONTENT"}, "finish_reason": "stop"}],
            "usage": copy.deepcopy(USAGE), **changes}


@pytest.mark.parametrize("operation", ["agent", "extraction", "preflight"])
def test_exact_returned_model_is_accepted_for_every_operation(tmp_path, operation):
    raw = completion()
    with PolzaClient(api_key="offline", transport=lambda *args: raw, usage_path=tmp_path / "usage.jsonl") as client:
        client.set_scope("E3F:6", "Benjamin")
        assert client.complete([], operation=operation) is raw
        assert len(client.records) == 1
        row = json.loads((tmp_path / "usage.jsonl").read_text())
        assert row["request_status"] == "completed" and row["response_accepted"] is True
        assert row["expected_model"] == row["model"] == MODEL
        assert row["model_identity_validation"] == "EXACT_MATCH"
        assert row["cost_rub"] == 0.3


@pytest.mark.parametrize("operation", ["agent", "extraction", "preflight"])
def test_wrong_model_is_rejected_after_actual_identity_and_billing_are_saved(tmp_path, operation):
    calls = []
    raw = completion(model="other/model")
    before = copy.deepcopy(raw)
    def transport(*args):
        calls.append(args)
        return raw
    with PolzaClient(api_key="offline", transport=transport, retries=3, usage_path=tmp_path / "usage.jsonl") as client:
        client.set_scope("E3F:6", "Benjamin")
        with pytest.raises(ProviderError, match="different model") as error:
            client.complete([], operation=operation)
        assert len(calls) == len(client.records) == 1
        assert error.value.request_sent is True
        assert error.value.request_id == "gen_123456" and error.value.usage == USAGE
        row = json.loads((tmp_path / "usage.jsonl").read_text())
        assert row["request_status"] == "completed" and row["response_accepted"] is False
        assert row["model_identity_validation"] == "MISMATCH"
        assert row["model"] == "other/model" and row["provider"] == "Chutes"
        assert row["request_id"] == "gen_123456" and row["expected_model"] == MODEL
        assert (row["input_tokens"], row["output_tokens"], row["total_tokens"], row["cost_rub"]) == (10, 5, 15, 0.3)
        assert client.get_usage_stats()["total_cost_rub"] == 0.3
        assert "SYNTHETIC_REJECTED_CONTENT" not in str(error.value)
        assert not (tmp_path / "writer_responses.jsonl").exists()
    assert raw == before


@pytest.mark.parametrize("model", [None, "", " ", 123, True, {"model": MODEL}])
def test_missing_or_invalid_model_identity_preserves_known_cost_before_rejection(tmp_path, model):
    raw = completion(model=model)
    with PolzaClient(api_key="offline", transport=lambda *args: raw, usage_path=tmp_path / "usage.jsonl") as client:
        with pytest.raises(ProviderError, match="omitted a valid returned model"):
            client.complete([])
        rows = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["cost_rub"] == 0.3
        assert rows[0]["request_id"] == "gen_123456" and rows[0]["provider"] == "Chutes"
        assert rows[0]["model_identity_validation"] == "MISSING_OR_INVALID"
        assert rows[0]["response_accepted"] is False


def test_absent_model_field_cannot_inherit_the_requested_identity(tmp_path):
    raw = completion()
    del raw["model"]
    with PolzaClient(api_key="offline", transport=lambda *args: raw, usage_path=tmp_path / "usage.jsonl") as client:
        with pytest.raises(ProviderError):
            client.complete([])
        row = json.loads((tmp_path / "usage.jsonl").read_text())
        assert row["model"] is None and row["expected_model"] == MODEL
        assert row["cost_rub"] == 0.3


@pytest.mark.parametrize("model", ["qwen/qwen3.6-27B", "qwen/qwen3.6-27b ", "Qwen3.6-27B", "qwen/qwen3.6-27b:fp8"])
def test_model_aliases_case_or_spacing_are_not_accepted_as_exact_backbone(model):
    with PolzaClient(api_key="offline", transport=lambda *args: completion(model=model)) as client:
        with pytest.raises(ProviderError, match="different model"):
            client.complete([])
        assert client.records[0]["model"] == model and client.records[0]["response_accepted"] is False


def test_received_cost_is_charged_once_even_when_content_is_rejected(tmp_path):
    calls = []
    def transport(*args):
        calls.append(args)
        return completion(model="other/model")
    with PolzaClient(api_key="offline", transport=transport, max_cost_rub=0.3, usage_path=tmp_path / "usage.jsonl") as client:
        with pytest.raises(ProviderError):
            client.complete([])
        with pytest.raises(BudgetExceeded, match="spending cap"):
            client.complete([])
        assert len(calls) == len(client.records) == 1
        assert client.get_usage_stats()["total_cost_rub"] == 0.3


def test_missing_provider_cost_remains_unknown_for_rejected_identity(tmp_path):
    raw = completion(model="other/model", usage={**USAGE, "cost_rub": None})
    with PolzaClient(api_key="offline", transport=lambda *args: raw, usage_path=tmp_path / "usage.jsonl") as client:
        with pytest.raises(ProviderError):
            client.complete([])
        assert client.records[0]["cost_rub"] is None
        assert client.get_usage_stats()["total_cost_rub"] is None
        with pytest.raises(BudgetExceeded, match="omitted cost"):
            client.complete([])


def test_valid_same_request_id_is_not_confused_with_a_model_mismatch():
    # Model enforcement concerns the returned model, not a new request-ID policy.
    # Separate offline accounting audits can flag repeated provider IDs.
    with PolzaClient(api_key="offline", transport=lambda *args: completion()) as client:
        client.complete([])
        client.complete([])
        assert len(client.records) == 2 and all(row["response_accepted"] for row in client.records)


def test_streamed_wrong_model_final_usage_is_preserved_without_paid_retry(tmp_path):
    events = [
        {"id": "gen_123456", "model": "other/model", "provider": "Chutes",
         "choices": [{"index": 0, "delta": {"content": "wrong completion"}, "finish_reason": "stop"}]},
        {"id": "gen_123456", "model": "other/model", "provider": "Chutes", "choices": [], "usage": USAGE},
    ]
    stream = ("".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n").encode()
    calls = []
    def transport(*args):
        calls.append(args)
        return read_completion_stream(io.BytesIO(stream))
    with PolzaClient(api_key="offline", transport=transport, stream=True, retries=3, usage_path=tmp_path / "usage.jsonl") as client:
        with pytest.raises(ProviderError, match="different model"):
            client.complete([])
        assert len(calls) == len(client.records) == 1
        assert client.records[0]["cost_rub"] == USAGE["cost_rub"]
        assert client.records[0]["model"] == "other/model"


def test_tool_chat_cannot_expose_wrong_model_tool_calls(tmp_path):
    raw = completion(model="other/model", choices=[{"message": {"tool_calls": [
        {"id": "call-1", "function": {"name": "Search", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}])
    with PolzaClient(api_key="offline", transport=lambda *args: raw, usage_path=tmp_path / "usage.jsonl") as client:
        with pytest.raises(ProviderError, match="different model"):
            client.chat_with_tools([], [])
        assert len(client.records) == 1 and client.records[0]["cost_rub"] == 0.3
