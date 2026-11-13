"""Versioned command and extraction contracts for entity-grounded memory."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

DESIGN = "mnemos/entity-grounded-v1"
PROMPT = "mnemos/entity-grounded-prompts-v1"
RELATIONS = {"MENTIONS", "PARTICIPATES_IN", "HAS_FACT", "HAS_STATE", "DEPENDS_ON", "DERIVED_FROM", "SUPPORTS", "CONTRADICTS", "MEMBER_OF", "APPLIES_TO", "REQUIRES_INPUT", "SUPERSEDES", "SPECIALIZES"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Span(Strict):
    trace_id: str
    event_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class Episode(Strict):
    title: str
    task: str | None
    goal: str | None
    summary: str
    events: list[str]
    decisions: list[str]
    outcome: Literal["success", "failure", "unknown"]
    uncertainty: list[str]
    scope: str
    provenance: list[Span] = Field(min_length=1)


class Entity(Strict):
    title: str
    entity_type: str
    canonical_name: str
    aliases: list[str]
    scope: str
    identity_candidates: list[str]
    resolution: Literal["resolved", "ambiguous"]
    confidence: float = Field(ge=0, le=1)
    provenance: list[Span] = Field(min_length=1)


class Fact(Strict):
    title: str
    subject_id: str
    predicate: str
    value: str | float | bool | None
    value_type: Literal["string", "number", "boolean", "unknown", "entity"]
    target_id: str | None
    units: str | None
    modality: Literal["required", "preferred", "observed", "unknown"]
    scope: str
    valid_from: str | None
    valid_to: str | None
    assertion_time: str | None
    supersedes_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    provenance: list[Span] = Field(min_length=1)


class StateRecord(Strict):
    title: str
    task_id: str
    branch_id: str
    state_type: Literal["goal", "decision", "constraint", "pending", "outcome"]
    execution_status: Literal["unknown", "pending", "committed", "success", "failure"]
    content: str
    dependencies: list[str]
    entity_ids: list[str]
    scope: str
    valid_from: str | None
    valid_to: str | None
    uncertainty: list[str]
    provenance: list[Span] = Field(min_length=1)


class Concept(Strict):
    title: str
    proposition: str
    applicability: list[str]
    exceptions: list[str]
    supporting_ids: list[str] = Field(min_length=1)
    contradicting_ids: list[str]
    scope: str
    provenance: list[Span] = Field(min_length=1)


class Slot(Strict):
    name: str
    object_type: Literal["fact", "state", "entity"]
    predicate: str | None
    scope: str
    entity_type: str | None


class Rule(Strict):
    title: str
    instruction: str
    applicability: list[str]
    target_entity_types: list[str]
    task_types: list[str]
    required_inputs: list[Slot] = Field(min_length=1)
    exceptions: list[str] = Field(min_length=1)
    supporting_ids: list[str] = Field(min_length=1)
    contradicting_ids: list[str]
    scope: str
    provenance: list[Span] = Field(min_length=1)


SCHEMAS = {"propose_episodes": Episode, "resolve_entities": Entity, "propose_facts": Fact,
           "propose_task_state": StateRecord, "propose_concepts": Concept, "propose_rule": Rule}
TYPES = {"propose_episodes": "episode", "resolve_entities": "entity", "propose_facts": "fact",
         "propose_task_state": "state", "propose_concepts": "concept", "propose_rule": "rule"}


class Command(Strict):
    operation_id: str = Field(min_length=1, max_length=200)
    graph_version: int = Field(ge=0)
    config_id: str
    type: Literal["import_traces", "propose_episodes", "resolve_entities", "decide_identity", "revise_proposal", "propose_facts", "propose_task_state", "propose_concepts", "embed", "project", "propose_hyperedges", "propose_rule", "propose_direct_rule", "validate_rule", "challenge_source", "revalidate"]
    input_ids: list[str] = Field(default_factory=list, max_length=500)
    parameters: dict = Field(default_factory=dict)


class Commit(Strict):
    graph_version: int
    accepted_ids: list[str]
    reviewer: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    identity_decisions: dict[str, str | None] = Field(default_factory=dict)


class ContextQuery(Strict):
    query: str = Field(min_length=1, max_length=16000)
    scope: str | None = None
    task_id: str | None = None
    branch_id: str = "main"
    as_of: str | None = None
    variant: Literal["A", "B", "C", "D"] = "D"
    token_budget: int = Field(default=4096, ge=16, le=32768)
    max_hops: int = Field(default=3, ge=0, le=6)
    max_nodes: int = Field(default=100, ge=1, le=1000)
    allow_hypotheses: bool = False


INSTRUCTIONS = {
    "propose_episodes": "Group ordered observations into episodes, not individual attributes. Preserve chronology, decisions and outcome. Outcome is unknown unless actually observed. Can abstain.",
    "resolve_entities": "Identify scoped entities from evidence. Reuse supplied resolved canonical entities: do not emit a duplicate record for an already represented identity unless the new evidence makes identity uncertain. Show existing identity candidates for uncertain matches. Same names across scopes are distinct. Ambiguous identities remain ambiguous; never silently merge.",
    "propose_facts": "Extract atomic entity-grounded facts with literal values, units, modality, validity and scope. subject_id and target_id must be supplied accepted entity IDs. Missing values remain null/unknown. Required is different from preferred. Narration is not observed success. Never infer a requirement from a proposed itinerary.",
    "propose_task_state": "Extract source-supported goals, decisions, constraints, pending actions and outcomes, with task/branch/dependencies. Use supplied entity IDs. Unobserved completion/status is unknown. Never invent a task result.",
    "propose_concepts": "Propose scoped conditional concepts from selected accepted episodes, facts and states. Keep concrete values in facts. Include exceptions and contradictions; can abstain. Supporting IDs must be supplied graph objects.",
    "propose_rule": "Propose a nontrivial procedural hypothesis relating all accepted hyperedge members. Include applicability, typed scoped required_inputs, exceptions and counterevidence. Similarity is not evidence; paraphrase of one member is not a rule. Can abstain. Never invent missing inputs.",
}
SCHEMAS["propose_direct_rule"] = Rule
TYPES["propose_direct_rule"] = "rule"
INSTRUCTIONS["propose_direct_rule"] = "Propose a scoped procedural hypothesis directly from supplied accepted experiences, facts and states, WITHOUT semantic hyperedges. Preserve required input slots and exceptions. This is experimental variant C, not hypergraph-induced. Abstain unless supported."
