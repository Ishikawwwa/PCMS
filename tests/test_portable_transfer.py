"""Independent end-to-end transfer fixtures; no credentials, corpus, or models.

Observed events and actions are scripted, explicit task contracts are supplied
by the caller, and natural-language interpretations stay advisory. No domain
schema is added to ConstraintMemory and no task-success oracle is used.
"""
import copy
import hashlib
import json

import pytest

from constraint_memory.memory.portable import ConstraintMemory
from constraint_memory.memory.binding_tables import TABLE_GUIDANCE, unpack_sections


class Flow:
    def __init__(self, domain, *, opaque=False, layout="mapping", numbers="integer"):
        self.domain, self.opaque, self.layout, self.numbers = domain, opaque, layout, numbers
        self.memory = ConstraintMemory(self.ref("isolated-namespace"))
        self.serial = 0
        self.outcomes = []

    def ref(self, label):
        if self.opaque:
            return "~opaque/" + hashlib.sha256((self.domain + ":" + label).encode()).hexdigest()[:24]
        return self.domain + "/" + label

    def value(self, label):
        return self.ref("entity-value/" + label)

    def number(self, value):
        return value if self.numbers == "integer" else float(value) if self.numbers == "float" else f"{value:.3f}"

    def event(self, content, step, kind="observation"):
        self.serial += 1
        event = self.ref("event/" + str(self.serial))
        self.memory.observe(event, step, content, kind=kind)
        return event

    def fact(self, label, value, step, *, unit=None, entity=None):
        reference = self.ref(label)
        anchor = None
        if entity:
            known = self.memory.resolve(self.ref(entity), step)
            assert known is not None
            anchor = {"ref": self.ref(entity), "version": known["version"]}
        if self.layout == "mapping":
            content = {"values": {reference: value}, "units": {reference: unit},
                       "entity_bindings": {reference: anchor}, "metadata": {"sequence": self.serial + 1}}
            pointer = "/values/" + reference.replace("~", "~0").replace("/", "~1")
        else:
            content, pointer = {"records": [{"key": reference, "value": value, "unit": unit, "entity_ref": anchor}]}, "/records/0/value"
        event = self.event(content, step)
        record = self.memory.write_fact(reference, event, pointer, step, unit=unit, entity_ref=anchor)
        assert record["value"] == value
        return record

    def contract(self, label, subject, operator, step, *, target=None, literal=None, unit=None, **target_options):
        goal = {"ref": self.ref(target), **target_options} if target else {"value": literal, **target_options}
        predicate = {"subject_ref": self.ref(subject), "operator": operator, "target": goal}
        if unit:
            predicate["unit"] = unit
        event = self.event({"requirements": [predicate]}, step, kind="task_contract")
        return self.memory.write_constraint(self.ref(label), event, "/requirements/0", step)

    def advisory(self, label, wording, subject, operator, literal, step=0):
        user = self.event(wording, step, kind="request")
        # A writer's self-reported success is data, not semantic authority.
        predicate = {"subject_ref": self.ref(subject), "operator": operator,
                     "target": {"value": literal}, "status": "SATISFIED", "task_success": True}
        writer = self.event({"interpretations": [predicate]}, step + 1, kind="model_output")
        record = self.memory.write_advisory_constraint(self.ref(label), writer, "/interpretations/0", user, 0, len(wording), step + 1)
        assert record["source"]["quote"] == wording
        assert record["semantic_status"] == "UNVERIFIED"
        return record

    def check(self, label, step, expected):
        result = self.memory.check(self.ref(label), step)
        assert result["status"] == expected
        assert result["task_success"] is None
        self.outcomes.append((label, step, result["status"]))
        return result

    def frontier(self, labels, step, expected="OK"):
        before = self.memory.snapshot()
        # This deterministic test counter measures characters, not model tokens.
        result = self.memory.frontier([self.ref(label) for label in labels], step, len, 100000)
        assert result["status"] == expected
        assert unpack_sections(json.loads(result["context"][len(TABLE_GUIDANCE):])) == result["sections"]
        assert self.memory.snapshot() == before
        return result

    def withdraw(self, label, step):
        event = self.event({"withdraw": self.ref(label)}, step, kind="action")
        return self.memory.withdraw_fact(self.ref(label), event, "", step)


def scheduling(flow):
    flow.fact("resource.selected", flow.value("space-A"), 0)
    flow.fact("resource.capacity", flow.number(8), 0, unit="people", entity="resource.selected")
    flow.fact("attendance", flow.number(6), 0, unit="people")
    flow.fact("preceding.finish", flow.number(600), 0, unit="minute")
    flow.fact("current.start", flow.number(615), 0, unit="minute")
    flow.advisory("accessibility-note", "Please choose a space with live captions.\nKeep a transition buffer between sessions.",
                  "resource.features", "SUPERSET_OF", ["captions"])
    flow.contract("transition-live", "current.start", "GTE", 2, target="preceding.finish", offset=15, unit="minute")
    flow.contract("transition-approved", "current.start", "GTE", 2, target="preceding.finish", offset=15,
                  binding_mode="SNAPSHOT", version=1, unit="minute")
    flow.contract("capacity", "resource.capacity", "GTE", 2, target="attendance", unit="people")
    flow.check("transition-live", 2, "SATISFIED")
    flow.check("capacity", 2, "SATISFIED")
    flow.check("accessibility-note", 2, "UNVERIFIED")
    flow.frontier(["transition-live", "capacity", "resource.selected"], 2)
    flow.fact("resource.selected", flow.value("space-B"), 3)
    flow.check("capacity", 3, "UNVERIFIED")
    flow.frontier(["capacity"], 3, "UNVERIFIED")
    flow.fact("resource.capacity", flow.number(12), 4, unit="people", entity="resource.selected")
    flow.check("capacity", 4, "SATISFIED")
    flow.fact("preceding.finish", flow.number(650), 5, unit="minute")
    flow.check("transition-live", 5, "VIOLATED")
    saved = flow.check("transition-approved", 5, "SATISFIED")
    assert saved["bindings"][1]["version"] == 1
    flow.fact("current.start", flow.number(675), 6, unit="minute")
    flow.check("transition-live", 6, "SATISFIED")
    assert flow.memory.resolve(flow.ref("resource.capacity"), 2)["value"] == flow.number(8)
    flow.withdraw("resource.selected", 7)
    flow.check("capacity", 7, "UNVERIFIED")
    flow.frontier(["accessibility-note", "capacity"], 7, "UNVERIFIED")


def procurement(flow):
    flow.fact("supplier.selected", flow.value("vendor-A"), 0)
    flow.fact("quote.total", flow.number(2200), 0, unit="USD", entity="supplier.selected")
    flow.fact("purchase.limit", flow.number(2500), 0, unit="USD")
    flow.advisory("packaging-note", "Use recycled packaging where practicable; confirm the supplier's certification first.",
                  "quote.certifications", "SUPERSET_OF", ["recycled"])
    flow.contract("budget-live", "quote.total", "LTE", 2, target="purchase.limit", unit="USD")
    flow.contract("budget-approved", "quote.total", "LTE", 2, target="purchase.limit", unit="USD", binding_mode="SNAPSHOT", version=1)
    flow.check("budget-live", 2, "SATISFIED")
    flow.fact("supplier.selected", flow.value("vendor-B"), 3)
    flow.check("budget-live", 3, "UNVERIFIED")
    flow.fact("quote.total", flow.number(2600), 4, unit="USD", entity="supplier.selected")
    flow.check("budget-live", 4, "VIOLATED")
    assert flow.memory.resolve(flow.ref("quote.total"), 4)["value"] == flow.number(2600)
    flow.fact("quote.total", flow.number(2350), 5, unit="USD", entity="supplier.selected")
    flow.check("budget-live", 5, "SATISFIED")
    flow.fact("purchase.limit", flow.number(2300), 6, unit="USD")
    flow.check("budget-live", 6, "VIOLATED")
    flow.check("budget-approved", 6, "SATISFIED")
    flow.fact("quote.total", flow.number(2000), 7, unit="EUR", entity="supplier.selected")
    unit_check = flow.check("budget-live", 7, "UNVERIFIED")
    assert unit_check["reason"] == "UNIT_UNKNOWN_OR_MISMATCH"
    flow.fact("quote.certifications", ["recycled"], 7, entity="supplier.selected")
    flow.check("packaging-note", 7, "UNVERIFIED")
    flow.frontier(["budget-live", "packaging-note", "supplier.selected"], 7, "UNVERIFIED")


def deployment(flow):
    flow.fact("artifact.selected", flow.value("build-A"), 0)
    flow.fact("artifact.digest", flow.value("digest-A"), 0, entity="artifact.selected")
    flow.fact("approval.digest", flow.value("digest-A"), 0)
    flow.fact("artifact.flags", ["signed", "trace"], 0, entity="artifact.selected")
    flow.fact("policy.flags", ["signed", "trace"], 0)
    flow.advisory("retention-note", "Enable audit trails and preserve operational logs for the agreed retention period.",
                  "artifact.flags", "SUPERSET_OF", ["trace"])
    flow.contract("identity-live", "artifact.digest", "EQUALS", 2, target="approval.digest")
    flow.contract("identity-approved", "artifact.digest", "EQUALS", 2, target="approval.digest", binding_mode="SNAPSHOT", version=1)
    flow.contract("mandatory-flags", "artifact.flags", "SUPERSET_OF", 2, target="policy.flags")
    flow.check("identity-live", 2, "SATISFIED")
    flow.check("mandatory-flags", 2, "SATISFIED")
    flow.fact("artifact.selected", flow.value("build-B"), 3)
    flow.check("identity-live", 3, "UNVERIFIED")
    flow.check("mandatory-flags", 3, "UNVERIFIED")
    flow.fact("artifact.digest", flow.value("digest-B"), 4, entity="artifact.selected")
    flow.fact("artifact.flags", ["signed"], 4, entity="artifact.selected")
    flow.check("identity-live", 4, "VIOLATED")
    flow.check("mandatory-flags", 4, "VIOLATED")
    assert flow.memory.resolve(flow.ref("artifact.flags"), 4)["value"] == ["signed"]
    flow.fact("approval.digest", flow.value("digest-B"), 5)
    flow.check("identity-live", 5, "SATISFIED")
    flow.check("identity-approved", 5, "VIOLATED")
    flow.fact("artifact.flags", ["signed", "trace"], 6, entity="artifact.selected")
    flow.check("mandatory-flags", 6, "SATISFIED")
    flow.check("retention-note", 6, "UNVERIFIED")
    frontier = flow.frontier(["identity-live", "mandatory-flags", "artifact.selected"], 6)
    linked = [row for row in frontier["sections"]["FACTS"] if row.get("entity_ref")]
    assert linked and all(row["entity_ref"]["version"] == 2 for row in linked)


def inventory(flow):
    flow.fact("batch.selected", flow.value("lot-A"), 0)
    flow.fact("count.available", flow.number(28), 0, unit="items", entity="batch.selected")
    flow.fact("reorder.floor", flow.number(20), 0, unit="items")
    flow.advisory("handling-note", "Keep the selected batch protected from humidity and confirm its expiry before dispatch.",
                  "batch.flags", "SUPERSET_OF", ["dry"])
    flow.contract("coverage", "count.available", "GTE", 2, target="reorder.floor", unit="items")
    flow.check("coverage", 2, "SATISFIED")
    # Independent counters disagree in the same reconciliation window. Their
    # insertion order must not determine a verified inventory decision.
    flow.fact("count.available", flow.number(8), 3, unit="items", entity="batch.selected")
    flow.fact("count.available", flow.number(29), 3, unit="items", entity="batch.selected")
    flow.check("coverage", 3, "UNVERIFIED")
    flow.frontier(["coverage"], 3, "UNVERIFIED")
    flow.fact("count.available", flow.number(24), 4, unit="items", entity="batch.selected")
    flow.check("coverage", 4, "SATISFIED")
    flow.check("coverage", 3, "UNVERIFIED")
    flow.fact("batch.selected", flow.value("lot-B"), 5)
    flow.check("coverage", 5, "UNVERIFIED")
    flow.fact("count.available", flow.number(12), 6, unit="items", entity="batch.selected")
    flow.check("coverage", 6, "VIOLATED")
    flow.fact("count.available", flow.number(22), 7, unit="items", entity="batch.selected")
    flow.check("coverage", 7, "SATISFIED")
    flow.check("handling-note", 7, "UNVERIFIED")
    flow.withdraw("count.available", 8)
    flow.check("coverage", 8, "UNVERIFIED")
    assert len(flow.memory.snapshot()["facts"][flow.ref("count.available")]) == 7


WORKFLOWS = {"scheduling": scheduling, "procurement": procurement, "deployment": deployment, "inventory": inventory}


@pytest.mark.parametrize("domain", WORKFLOWS)
@pytest.mark.parametrize("opaque", [False, True])
@pytest.mark.parametrize("layout", ["mapping", "records"])
@pytest.mark.parametrize("numbers", ["integer", "float", "decimal-text"])
def test_end_to_end_workflows_have_same_outcomes_under_renaming_and_formats(domain, opaque, layout, numbers):
    flow = Flow(domain, opaque=opaque, layout=layout, numbers=numbers)
    WORKFLOWS[domain](flow)
    # Every consumed proof remains tied to an immutable event and valid pointer.
    for histories in flow.memory.snapshot()["facts"].values():
        for record in histories:
            source = record["source"]
            assert source["event_sha256"] == flow.memory.events[source["event_id"]]["sha256"]
    assert flow.outcomes and all(outcome[2] in {"SATISFIED", "VIOLATED", "UNVERIFIED"} for outcome in flow.outcomes)


@pytest.mark.parametrize("domain", WORKFLOWS)
@pytest.mark.parametrize("reverse", [False, True])
def test_unreconciled_conflicting_sources_are_unknown_in_every_domain_and_order(domain, reverse):
    flow = Flow(domain, opaque=True, layout="records")
    flow.contract("limit", "measured.quantity", "GTE", 0, literal=20, unit="units")
    values = [8, 29]
    if reverse:
        values.reverse()
    for value in values:
        flow.fact("measured.quantity", value, 1, unit="units")
    flow.check("limit", 1, "UNVERIFIED")
    assert flow.memory.resolve(flow.ref("measured.quantity"), 1, version=1) is None
    assert flow.memory.resolve(flow.ref("measured.quantity"), 1, version=2) is None
    flow.frontier(["limit"], 1, "UNVERIFIED")
    flow.fact("measured.quantity", 23, 2, unit="units")
    flow.check("limit", 2, "SATISFIED")
    flow.check("limit", 1, "UNVERIFIED")


@pytest.mark.parametrize("domain", WORKFLOWS)
def test_missing_facts_late_evidence_and_scope_boundaries_do_not_become_success(domain):
    flow = Flow(domain, opaque=True)
    flow.fact("actual.quantity", 21, 0, unit="units")
    flow.contract("policy", "actual.quantity", "GTE", 1, target="required.quantity", unit="units")
    result = flow.check("policy", 1, "UNVERIFIED")
    assert result["reason"] == "TARGET_UNKNOWN"
    flow.fact("required.quantity", 20, 5, unit="units")
    flow.check("policy", 1, "UNVERIFIED")
    flow.check("policy", 5, "SATISFIED")
    with pytest.raises(ValueError, match="scope"):
        flow.memory.observe("foreign", 6, {}, scope_id="unrelated-namespace")
    flow.withdraw("actual.quantity", 6)
    flow.check("policy", 6, "UNVERIFIED")
    other = ConstraintMemory("new-isolated-namespace")
    assert other.resolve(flow.ref("actual.quantity"), 6) is None


@pytest.mark.parametrize("domain", WORKFLOWS)
@pytest.mark.parametrize("wording", ["Please keep the selected resource within the approved limit.",
    "REQUEST\n- selected resource must satisfy the approved limit\n- ask when evidence is missing",
    "Constraint proposal: selected <= approved. This is an interpretation, not a machine contract."])
def test_natural_language_and_model_self_certification_stay_advisory(domain, wording):
    flow = Flow(domain, opaque=True)
    flow.advisory("free-form", wording, "quantity", "LTE", 20)
    flow.fact("quantity", 10, 2)
    checked = flow.check("free-form", 2, "UNVERIFIED")
    assert checked["reason"] == "SEMANTICS_NOT_VERIFIED"
    context = flow.frontier(["free-form"], 2, "UNVERIFIED")
    record = context["sections"]["CONSTRAINTS"][0]
    assert record["source"]["quote"] == wording and record["semantic_status"] == "UNVERIFIED"
    assert record["predicate"]["task_success"] is True and checked["task_success"] is None


@pytest.mark.parametrize("domain", WORKFLOWS)
def test_rendering_budget_overflow_never_deletes_protected_evidence(domain):
    flow = Flow(domain, opaque=True)
    WORKFLOWS[domain](flow)
    before = copy.deepcopy(flow.memory.snapshot())
    references = list(flow.memory.constraints)
    result = flow.memory.frontier(references, 8, len, 2)
    assert result["status"] == "BUDGET_OVERFLOW" and len(result["context"]) <= 2
    assert result["required_sections"]["CONSTRAINTS"]
    assert flow.memory.snapshot() == before


@pytest.mark.parametrize("value,target", [("2027-04-15T08:00:00Z", "2027-04-13T08:00:00Z"),
                                          ("USD 2,200.00", 2500), ("v1.10.0", "v1.2.0")])
def test_unimplemented_date_currency_and_semver_interpretations_are_not_guessed(value, target):
    flow = Flow("unsupported-presentations")
    flow.fact("value", value, 0)
    flow.contract("comparison", "value", "GTE", 1, literal=target)
    flow.check("comparison", 1, "UNVERIFIED")
