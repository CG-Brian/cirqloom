"""Versioned bundle schema (AGENTS.md §8). Single source of truth on the Python side."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SchemaVersion = Literal["1.1.0"]

# Identifies the version of the data schema.
SCHEMA_VERSION: SchemaVersion = "1.1.0"

# Indicates the validation or publication status of evidence.
EvidenceStatus = Literal["published",
                         "measured",
                         "candidate",
                         "experimental",
                         "unavailable",]

# Identifies the method or source used to obtain evidence.
EvidenceType = Literal["heuristic_attention",
                       "logit_attribution",
                       "published_annotation",
                       "structural",
                       "causal_intervention",]


class StrictModel(BaseModel):
    """Provides strict validation rules shared by all schema models."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelInfo(StrictModel):
    """Stores metadata about the pretrained model being analyzed."""
    id: str
    hf_id: str
    revision: str | None # None only when no weights were loaded

    layers: int = Field(gt=0)
    heads_per_layer: int = Field(gt=0)
    tokenizer_id: str


class AttentionAsset(StrictModel):
    """Raw attention of one example prompt, all heads: little-endian float16, C order,
    shape [layers, heads_per_layer, sequence_length, sequence_length], row = destination.
    float16 keeps about 3 significant digits; rows sum to 1 only within ~1e-3."""
    uri: str
    dtype: Literal["float16"]
    shape: list[int] = Field(min_length=4, max_length=4)
    prompt_index: int = Field(ge=0)


class Experiment(StrictModel):
    """One prompt suite at one sequence length, scored on every prompt (one batch)."""
    id: str
    prompt_suite: str
    kind: Literal["measurement",
                  "control",]
    description: str
    seeds: list[int] = Field(min_length=1)  # one generated prompt per seed
    sequence_length: int = Field(gt=0)
    tokens_uri: str
    attention: AttentionAsset | None


class Node(StrictModel):
    """Represents an individual attention head in the model graph."""
    id: str
    type: Literal["attention_head"]
    layer: int = Field(ge=0)
    head: int = Field(ge=0)


class Observation(StrictModel):
    """Stores detector scores and supporting evidence for an attention head."""
    node_id: str
    experiment_id: str
    detector_id: str
    detector_version: str
    evidence_type: EvidenceType
    parameters: dict[str, int]  # exact detector settings used, e.g. {"burn_in": 2}
    score: float | None  # mean over examples
    score_std: float | None  # sample std (ddof=1) over examples; null when sample_count < 2
    per_example_scores: list[float]
    status: EvidenceStatus
    sample_count: int = Field(ge=0)  # number of examples (prompts), not mask cells
    method_reference: str


class RoleDef(StrictModel):
    """Defines a functional role within a neural circuit."""
    id: str
    label: str
    description: str


class Circuit(StrictModel):
    """Describes a neural circuit and its constituent functional roles."""
    id: str
    name: str
    origin: Literal["synthetic_fixture",
                    "published",]
    description: str
    citation: str | None
    roles: list[RoleDef]


class Membership(StrictModel):
    """Associates an attention head with a specific role in a circuit."""
    circuit_id: str
    node_id: str
    role: str
    source_reference: str
    evidence_status: EvidenceStatus


class Edge(StrictModel):
    """Represents a structural or evidence-supported relationship between nodes."""
    id: str
    source_node_id: str
    target_node_id: str
    edge_type: Literal["structural",
                       "schematic_published",
                       "experimentally_supported_in_paper",]
    evidence_status: EvidenceStatus
    citation: str | None
    description: str


class Provenance(StrictModel):
    """Records the origin, generator, and version of the analysis data."""
    generator: str
    generator_version: str
    notes: str
    environment: dict[str, str]  # library versions, device, dtype, tokenization policy


class Manifest(StrictModel):
    """Defines bundle metadata, experiment details, and the location of graph data."""
    schema_version: SchemaVersion
    bundle_id: str
    synthetic_fixture: bool
    display_warning: str | None
    model: ModelInfo
    experiments: list[Experiment]
    provenance: Provenance
    graph_uri: str


class Graph(StrictModel):
    """Contains the nodes, observations, circuits, memberships, and edges for visualization."""
    schema_version: SchemaVersion
    bundle_id: str
    nodes: list[Node]
    observations: list[Observation]
    circuits: list[Circuit]
    memberships: list[Membership]
    edges: list[Edge]


class TokenizedPrompt(StrictModel):
    """One prompt in an experiment's tokens file."""
    id: str
    seed: int
    token_ids: list[int]
    tokens: list[str]  # display string per id, leading spaces kept


class TokensFile(StrictModel):
    """Contents of Experiment.tokens_uri."""
    experiment_id: str
    prompts: list[TokenizedPrompt]
