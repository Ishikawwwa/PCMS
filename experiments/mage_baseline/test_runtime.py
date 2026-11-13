"""Independent offline checks of the paper-inspired MAGE runtime bridge.

Scripted clients and reversible in-memory tools exercise the actual controller
loop. These tests make no inference, network, dataset, credential, author-fidelity,
or research-performance claim.
"""
from __future__ import annotations

import copy
import json

import pytest

from experiments.mage_baseline.controller import MageController
from experiments.mage_baseline.kernel import MageKernel


def clone(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


class ReversibleEnvironment:
    """Actual state is reversible; historical execution/restore receipts are not."""

    def __init__(self):
        self.state = {"selected": None, "cost": 0, "valid_option_ids": ["a", "b", "c"]}
        self.executed = []
        self.restored = []
        self.checkpoint_reads = 0
        self.restore_error = None
        self.restore_corruption = None
        self.next_observation = None
        self.execute_error = None

    def checkpoint(self):
        self.checkpoint_reads += 1
        # Returning an alias deliberately tests whether the controller owns its
        # savepoint rather than retaining our subsequently mutated state object.
        return self.state

    def execute(self, action):
        self.executed.append(clone(action))
        if self.execute_error:
            raise self.execute_error
        if action.get("op") == "select":
            option = action["id"]
            if option not in self.state["valid_option_ids"]:
                return {"error": "invalid option", "id": option}
            self.state["selected"] = option
            self.state["cost"] = action.get("cost", 1)
        elif action.get("op") == "invalidate":
            self.state["valid_option_ids"].remove(action["id"])
        elif action.get("op") == "append":
            self.state.setdefault("notes", []).append(action["text"])
        if self.next_observation is not None:
            response, self.next_observation = self.next_observation, None
            return clone(response)
        return {"selected": self.state["selected"], "cost": self.state["cost"],
                "valid_option_ids": list(self.state["valid_option_ids"])}

    def restore(self, checkpoint):
        self.restored.append(clone(checkpoint))
        if self.restore_error:
            raise self.restore_error
        self.state = clone(checkpoint)
        if self.restore_corruption is not None:
            self.state = clone(self.restore_corruption)


class ScriptedClient:
    """Raw native completion envelopes and fake accounting for every attempt."""

    def __init__(self, outputs=(), hook=None):
        self.outputs = list(outputs)
        self.calls = []
        self.records = []
        self.hook = hook

    def complete(self, messages, *, json_output, operation, max_tokens):
        request = {"messages": clone(messages), "json_output": json_output,
                   "operation": operation, "max_tokens": max_tokens}
        self.calls.append(request)
        if self.hook:
            self.hook(request)
        self.records.append({"operation": operation, "total_tokens": 7,
                             "cost_rub": 0.125, "billing_status": "offline_fake"})
        if not self.outputs:
            raise AssertionError("Unexpected maintenance request")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        content = output if isinstance(output, str) else encoded(output)
        return {"id": f"fake-{len(self.calls)}", "model": "offline-scripted",
                "choices": [{"message": {"content": content}}]}


PASS = {"passed": True, "feedback": None}
FAIL = {"passed": False, "feedback": {"reason": "wrong branch"}}


def controller(outputs=(), *, environment=None, hook=None, logger=None, **overrides):
    env = environment or ReversibleEnvironment()
    client = ScriptedClient(outputs, hook)
    kwargs = {"task_instruction": "Choose from actual current offers.",
              "environment": env, "client": client, "token_counter": len,
              "raw_threshold_tokens": 1_000_000, "memory_token_limit": 1_000_000,
              "maintenance_max_tokens": 251, "logger": logger}
    kwargs.update(overrides)
    return MageController(**kwargs), env, client


def active_summary_ids(runtime):
    return [row["id"] for row in runtime.kernel.state()["C"]]


def assert_failed_closed(runtime, env):
    assert runtime.status == "FAILED"
    before = clone(env.executed)
    with pytest.raises((RuntimeError, ValueError)):
        runtime.read()
    with pytest.raises((RuntimeError, ValueError)):
        runtime.execute({"op": "select", "id": "c"})
    assert env.executed == before


@pytest.mark.parametrize("left,right", [
    (True, 1), (False, 0), ({"nested": [True]}, {"nested": [1]}),
    ([{"x": False}], [{"x": 0}]), (1, 1.0),
])
def test_kernel_uses_type_preserving_sibling_identity(left, right):
    kernel = MageKernel()
    first = kernel.grow({"argument": left}, {"answer": left})
    summary = kernel.compress("first branch")
    kernel.maintain("task", validator=lambda *args: (True, None))
    kernel.revise("alternative", summary["id"])
    second = kernel.grow({"argument": right}, {"answer": right})
    assert first["id"] != second["id"]
    assert encoded(second["content"]) == encoded({"action": {"argument": right},
                                                  "observation": {"answer": right}})


def test_kernel_identical_siblings_reuse_but_same_content_other_parent_does_not():
    kernel = MageKernel()
    first = kernel.grow({"x": 1}, {"observed": "same"})
    summary = kernel.compress("first")
    kernel.maintain("task", validator=lambda *args: (True, None))
    kernel.revise("repeat", summary["id"])
    repeat = kernel.grow({"x": 1}, {"observed": "same"})
    later = kernel.grow({"x": 1}, {"observed": "same"})
    assert repeat["id"] == first["id"]
    assert later["id"] != first["id"]
    assert len(kernel.snapshot()["raw_nodes"]) == 3


def test_execute_records_actual_tool_result_and_root_checkpoint_is_owned():
    runtime, env, client = controller([FAIL])
    env.next_observation = {"actual": "returned independently", "typed": True}
    runtime.execute({"op": "select", "id": "a", "cost": 19})
    grown = runtime.kernel.snapshot()["raw_nodes"][runtime.kernel.pb]["content"]
    assert grown["action"] == {"op": "select", "id": "a", "cost": 19}
    assert grown["observation"] == {"actual": "returned independently", "typed": True}
    assert len([row for row in runtime.kernel.snapshot()["history"] if row["operation"] == "Grow"]) == 1
    runtime.compress("candidate selected a")
    assert env.state == {"selected": None, "cost": 0, "valid_option_ids": ["a", "b", "c"]}
    assert runtime.kernel.state()["C"] == []
    assert runtime.status == "READY"
    assert runtime.actions_used == 1 and runtime.revisions_used == 1
    assert len(client.records) == 1


def test_automatic_fallback_has_two_immediate_callbacks_and_no_reentrant_actor():
    holder, callback_order, blocked = {}, [], []

    def during_model_call(request):
        runtime, env = holder["runtime"], holder["env"]
        callback_order.append(request["operation"])
        assert runtime.status == "PENDING_VALIDATION"
        before = clone(env.executed)
        for attempt in [lambda: runtime.read(),
                        lambda: runtime.execute({"op": "select", "id": "b"})]:
            with pytest.raises((RuntimeError, ValueError)):
                attempt()
            blocked.append(True)
        assert env.executed == before

    runtime, env, client = controller([{"summary": {"choice": "a"}}, PASS],
                                       hook=during_model_call, raw_threshold_tokens=1)
    holder.update(runtime=runtime, env=env)
    runtime.execute({"op": "select", "id": "a"})
    assert callback_order == ["mage_summary", "mage_validation"]
    assert len(blocked) == 4
    assert [row["operation"] for row in client.records] == callback_order
    assert all(row["json_output"] is True and row["max_tokens"] == 251 for row in client.calls)
    assert runtime.maintenance_calls == 2 and runtime.status == "READY"
    assert env.executed == [{"op": "select", "id": "a"}]
    assert runtime.kernel.state()["R"] == []
    assert runtime.kernel.state()["C"][0]["validation"] == "VALID"


def test_failed_suffix_restores_prefix_and_failed_branch_is_only_inactive_hint():
    runtime, env, client = controller([PASS, FAIL])
    runtime.execute({"op": "select", "id": "a", "cost": 10})
    runtime.compress("valid prefix")
    valid_id = active_summary_ids(runtime)[0]
    prefix_checkpoint = clone(env.state)
    runtime.execute({"op": "select", "id": "b", "cost": 99})
    runtime.compress("bad suffix")
    assert env.state == prefix_checkpoint
    assert active_summary_ids(runtime) == [valid_id]
    state = runtime.read()["state"]
    assert [row["content"] for row in state["C"]] == ["valid prefix"]
    assert state["R"] == []
    assert all(hint["active"] is False for hint in state["H"])
    assert any(hint.get("content") == "bad suffix" for hint in state["H"])
    assert len(env.executed) == 2 and len(env.restored) == 1
    assert runtime.actions_used == 2 and runtime.revisions_used == 1
    assert len(client.records) == 2


def test_proactive_revision_restores_option_validity_and_excludes_later_suffix():
    runtime, env, _ = controller([PASS, PASS, PASS])
    runtime.execute({"op": "select", "id": "a", "cost": 10})
    runtime.compress("prefix")
    prefix = clone(env.state)
    runtime.execute({"op": "invalidate", "id": "b"})
    runtime.compress("middle invalidated offer b")
    target = active_summary_ids(runtime)[1]
    runtime.execute({"op": "select", "id": "c", "cost": 90})
    runtime.compress("later suffix")
    runtime.revise({"reason": "redo middle"}, target)
    assert env.state == prefix and "b" in env.state["valid_option_ids"]
    assert [row["content"] for row in runtime.read()["state"]["C"]] == ["prefix"]
    assert runtime.actions_used == 3 and runtime.revisions_used == 1
    assert len(runtime.kernel.snapshot()["summary_nodes"]) == 4


def test_repeated_action_after_restore_still_executes_and_reuses_raw_node():
    runtime, env, _ = controller([FAIL, PASS])
    action = {"op": "select", "id": "a", "cost": 3}
    runtime.execute(action)
    raw_id = runtime.kernel.pb
    runtime.compress("failed interpretation")
    runtime.execute(action)
    assert runtime.kernel.pb == raw_id
    assert len(env.executed) == 2 and runtime.actions_used == 2
    runtime.compress("corrected interpretation")
    assert len(runtime.kernel.snapshot()["raw_nodes"]) == 2
    assert len(runtime.kernel.snapshot()["summary_nodes"]) == 2
    assert runtime.kernel.state()["C"][0]["version"] == 2


def test_changed_actual_observation_after_restore_creates_new_sibling():
    runtime, env, _ = controller([FAIL])
    action = {"op": "select", "id": "a"}
    runtime.execute(action)
    first = runtime.kernel.pb
    runtime.compress("failed")
    env.next_observation = {"selected": "a", "cost": 1, "new_session_option_id": "changed"}
    runtime.execute(action)
    assert runtime.kernel.pb != first
    assert len(env.executed) == 2
    assert len(runtime.kernel.snapshot()["raw_nodes"]) == 3


@pytest.mark.parametrize("target", ["s0", "r1", "s999", None, 1])
def test_invalid_revision_target_cannot_mutate_environment(target):
    runtime, env, _ = controller([PASS])
    runtime.execute({"op": "select", "id": "a"})
    runtime.compress("valid")
    before = encoded(env.state)
    pointers = runtime.kernel.pb, runtime.kernel.pt
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        runtime.revise("invalid", target)
    assert encoded(env.state) == before and env.restored == []
    assert (runtime.kernel.pb, runtime.kernel.pt) == pointers
    assert runtime.revisions_used == 0


def test_inactive_branch_target_cannot_restore_an_unrelated_checkpoint():
    runtime, env, _ = controller([PASS, PASS])
    runtime.execute({"op": "select", "id": "a"})
    runtime.compress("first")
    old_target = active_summary_ids(runtime)[0]
    runtime.revise("alternative", old_target)
    runtime.execute({"op": "select", "id": "b"})
    runtime.compress("second")
    before, restores = encoded(env.state), len(env.restored)
    with pytest.raises((RuntimeError, ValueError)):
        runtime.revise("stale branch", old_target)
    assert encoded(env.state) == before and len(env.restored) == restores
    assert runtime.revisions_used == 1


def test_restore_exception_fails_closed_before_any_further_task_action():
    runtime, env, _ = controller([FAIL])
    runtime.execute({"op": "select", "id": "b"})
    env.restore_error = RuntimeError("offline restoration failure")
    with pytest.raises((RuntimeError, ValueError)):
        runtime.compress("bad")
    assert_failed_closed(runtime, env)


def test_restore_with_wrong_state_is_detected_even_when_restore_returns_normally():
    runtime, env, _ = controller([FAIL])
    runtime.execute({"op": "select", "id": "b"})
    env.restore_corruption = {"selected": "b", "cost": 777, "valid_option_ids": []}
    with pytest.raises((RuntimeError, ValueError)):
        runtime.compress("bad")
    assert_failed_closed(runtime, env)


def test_restore_verification_distinguishes_json_boolean_from_number():
    env = ReversibleEnvironment()
    env.state["feature"] = True
    runtime, env, _ = controller([FAIL], environment=env)
    runtime.execute({"op": "select", "id": "a"})
    env.restore_corruption = {"selected": None, "cost": 0,
                              "valid_option_ids": ["a", "b", "c"], "feature": 1}
    with pytest.raises((RuntimeError, ValueError)):
        runtime.compress("bad")
    assert_failed_closed(runtime, env)


@pytest.mark.parametrize("bad", [
    "not JSON", {"passed": "false", "feedback": "bad"},
    {"passed": 1, "feedback": None}, {"passed": True},
    {"passed": True, "feedback": None, "extra": "not allowed"},
    [True, None], RuntimeError("judge offline error"),
])
def test_invalid_or_failed_judge_is_never_treated_as_pass_and_is_not_retried(bad):
    runtime, env, client = controller([bad])
    runtime.execute({"op": "select", "id": "a"})
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        runtime.compress("candidate")
    assert_failed_closed(runtime, env)
    assert len(client.calls) == len(client.records) == 1
    assert runtime.kernel.state()["C"][0]["validation"] != "VALID"


@pytest.mark.parametrize("bad", [
    "invalid JSON", {"other": "missing summary"},
    {"summary": "candidate", "extra": True}, RuntimeError("summarizer offline error"),
])
def test_invalid_or_failed_summarizer_fails_closed_without_a_judge_retry(bad):
    runtime, env, client = controller([bad])
    runtime.execute({"op": "select", "id": "a"})
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        runtime.compress()
    assert_failed_closed(runtime, env)
    assert [row["operation"] for row in client.calls] == ["mage_summary"]
    assert len(client.records) == 1


def test_action_allowance_is_not_refunded_by_revision():
    runtime, env, _ = controller([FAIL], max_actions=2)
    runtime.execute({"op": "select", "id": "a"})
    runtime.compress("failed")
    runtime.execute({"op": "select", "id": "b"})
    before = clone(env.executed)
    with pytest.raises((RuntimeError, ValueError)):
        runtime.execute({"op": "select", "id": "c"})
    assert env.executed == before and runtime.actions_used == 2
    assert runtime.revisions_used == 1


def test_revision_allowance_is_monotonic_and_exhaustion_cannot_continue():
    runtime, env, client = controller([FAIL, FAIL], max_revisions=1)
    runtime.execute({"op": "select", "id": "a"})
    runtime.compress("first failed branch")
    runtime.execute({"op": "select", "id": "b"})
    with pytest.raises((RuntimeError, ValueError)):
        runtime.compress("second failed branch")
    assert_failed_closed(runtime, env)
    assert runtime.revisions_used == 1 and runtime.actions_used == 2
    assert len(client.records) == 2 and len(env.restored) == 1


def test_actual_model_usage_and_environment_receipts_survive_restore():
    runtime, env, client = controller([{"summary": "first"}, FAIL,
                                       {"summary": "second"}, PASS])
    runtime.execute({"op": "select", "id": "a"})
    runtime.compress()
    runtime.execute({"op": "select", "id": "b"})
    runtime.compress()
    assert len(client.records) == runtime.maintenance_calls == 4
    assert sum(row["total_tokens"] for row in client.records) == 28
    assert sum(row["cost_rub"] for row in client.records) == 0.5
    assert len(env.executed) == runtime.actions_used == 2
    assert runtime.revisions_used == 1 and env.state["selected"] == "b"


def test_read_serializes_only_active_crh_not_archived_full_raw_history():
    runtime, env, _ = controller([PASS])
    marker = "ONLY_IN_COVERED_RAW_TRACE_5864"
    runtime.execute({"op": "append", "text": marker})
    runtime.compress({"retained": "compact result"})
    delivered = runtime.read()
    assert set(delivered) == {"state", "context", "tokens"}
    assert json.loads(delivered["context"]) == delivered["state"]
    assert delivered["tokens"] == len(delivered["context"])
    assert marker not in delivered["context"]
    assert marker in encoded(runtime.kernel.snapshot()["raw_nodes"])
    assert delivered["state"]["R"] == []
    assert env.state["notes"] == [marker]


def test_memory_overflow_cannot_return_a_truncated_success_context():
    runtime, _, _ = controller(memory_token_limit=300)
    runtime.execute({"op": "append", "text": "unique-visible-content-" * 200})
    before = encoded(runtime.kernel.snapshot())
    with pytest.raises((RuntimeError, ValueError)):
        runtime.read()
    assert encoded(runtime.kernel.snapshot()) == before


def test_logger_receives_json_copies_and_cannot_rewrite_actual_history():
    seen = []

    def logger(event):
        json.dumps(event, allow_nan=False)
        seen.append(clone(event))
        event.clear()

    runtime, env, _ = controller([PASS], logger=logger)
    action = {"op": "select", "id": "a", "cost": 4}
    runtime.execute(action)
    action["id"] = "mutated caller action"
    runtime.compress("selected a")
    assert seen
    raw = runtime.kernel.snapshot()["raw_nodes"]
    assert any(node["content"]["action"] == {"op": "select", "id": "a", "cost": 4}
               for node in raw.values())
    assert env.executed == [{"op": "select", "id": "a", "cost": 4}]


@pytest.mark.parametrize("parameter", ["raw_threshold_tokens", "memory_token_limit",
                                      "max_actions", "maintenance_max_tokens"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_nonpositive_or_nonstrict_numeric_limits_are_rejected(parameter, bad):
    with pytest.raises((ValueError, TypeError)):
        controller(**{parameter: bad})


def test_read_results_are_detached_from_mutable_controller_state():
    runtime, _, _ = controller([PASS])
    runtime.execute({"op": "select", "id": "a"})
    runtime.compress({"choice": {"id": "a"}})
    delivered = runtime.read()
    delivered["state"]["C"][0]["content"]["choice"]["id"] = "tampered"
    assert runtime.read()["state"]["C"][0]["content"]["choice"]["id"] == "a"
