from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_design: Literal["entity_grounded_v1", "legacy"] = "entity_grounded_v1"
    prompt_version: str = "mnemos/entity-grounded-prompts-v1"
    representation_version: str = "mnemos/scoped-canonical-v1"
    retrieval_version: str = "mnemos/typed-context-v1"
    acceptance_policy: Literal["human_review"] = "human_review"
    mean_pairwise: float = Field(default=.3, ge=-1, le=1)
    candidate_cap: int = Field(default=100, ge=1, le=1000)
    retrieval_hops: int = Field(default=3, ge=0, le=6)
    retrieval_nodes: int = Field(default=100, ge=1, le=1000)
    context_tokens: int = Field(default=4096, ge=32, le=32768)
    agent_enabled: bool = False
    mode: Literal["demo", "polza"] = "polza"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    llm_model: str = "qwen/qwen3.6-27b"
    llm_revision: str = "provider-managed"
    llm_url: Literal["https://polza.ai/api/v1"] = "https://polza.ai/api/v1"
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int = 42
    max_tokens: int = Field(default=8192, ge=128, le=16384)
    max_model_calls: int = Field(default=100, ge=1, le=10000)
    dataset_id: str = ""
    top_k: int = Field(default=8, ge=1, le=100)
    threshold: float = Field(default=.30, ge=-1, le=1)
    min_pairwise: float = Field(default=.20, ge=-1, le=1)
    min_size: int = Field(default=2, ge=2, le=6)
    max_size: int = Field(default=6, ge=2, le=6)
    jaccard: float = Field(default=.85, gt=0, le=1)
    max_depth: int = Field(default=5, ge=3, le=12)
    held_out_queries: list[str] = Field(default_factory=lambda: [
        "How should an agent respond when a downstream service is overwhelmed?",
        "What evidence should be retained before retrying a failed operation?",
        "When should a belief be reconsidered after a source changes?",
    ])

    @model_validator(mode="after")
    def valid_limits(self):
        if self.min_size > self.max_size:
            raise ValueError("min_size must not exceed max_size")
        if not self.embedding_revision.strip() or self.embedding_revision in {"main", "latest"}:
            raise ValueError("Freeze an embedding model revision")
        return self


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Untitled experiment", min_length=1, max_length=100)
    settings: Settings = Field(default_factory=Settings)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["import", "episode", "concept", "edit", "review", "validate", "embed", "project", "associate", "crystallize", "challenge", "revalidate", "benchmark", "attribute"]
    targets: list[str] = Field(default_factory=list, max_length=500)
    parameters: dict = Field(default_factory=dict)
    expected_graph_version: int = Field(ge=0)
    idempotency_key: str = Field(min_length=1, max_length=200)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8000)
    variants: list[Literal["A", "B", "C", "D", "E"]] = Field(default_factory=lambda: ["A", "C", "E"], min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class BenchmarkInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_id: str = Field(pattern=r"^[0-9]{1,3}$")
    memory_mode: Literal["mnemos", "history", "none", "A", "B", "C", "D"] = "mnemos"
    max_steps_per_traveler: int = Field(default=30, ge=1, le=30)
    memory_token_budget: int = Field(default=4096, ge=512, le=32768)
    start_paused: bool = False
    auto_crystallize: bool = True
    settings: Settings = Field(default_factory=lambda: Settings(memory_design="legacy"))
