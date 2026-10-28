"""Turn the verified IOI annotations (annotations.json) into bundle Circuit/Membership/Edge records.

No model, no inference, no detectors: this module only reads the reviewed JSON file. The JSON is
the single place where paper facts live (with a locator for every role, member and relation);
`docs/sources/ioi-wang-2022.md` records how they were checked.

Granularity: a relation's senders are either measured one head at a time ("per_head": one edge
per sender head into the receiver) or as a whole group ("group": one group edge). Receivers are
a role group or the logits. Nothing is expanded into head pairs.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cirqloom_analysis.published.base import PublishedCircuit
from cirqloom_analysis.schema.bundle import (
    Circuit,
    Edge,
    EdgeChannel,
    EdgeEffect,
    EdgePositions,
    Endpoint,
    HeadEndpoint,
    Membership,
    ModelInfo,
    RoleDef,
    RoleGroupEndpoint,
    SpecialEndpoint,
)
from cirqloom_analysis.schema.validation import head_node_id

ANNOTATIONS_PATH = Path(__file__).with_name("annotations.json")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _CircuitInfo(_Strict):
    id: str
    name: str
    description: str
    citation: str


class _ModelRequirement(_Strict):
    id: str
    hf_id: str
    layers: int
    heads_per_layer: int


class _Role(_Strict):
    id: str
    label: str
    description: str
    locator: str


class _Member(_Strict):
    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    role: str
    locator: str
    note: str | None


class _PerHeadSenders(_Strict):
    granularity: Literal["per_head"]
    heads: list[tuple[int, int]] = Field(min_length=1)


class _GroupSender(_Strict):
    granularity: Literal["group"]
    role_group: str


class _Receiver(_Strict):
    role_group: str | None = None
    special: Literal["logits"] | None = None


class _Relation(_Strict):
    id: str
    senders: _PerHeadSenders | _GroupSender = Field(discriminator="granularity")
    receiver: _Receiver
    channel: EdgeChannel
    positions: EdgePositions | None
    effect: EdgeEffect | None
    conditions: list[str]
    locator: str = Field(min_length=1)
    description: str


class IoiAnnotations(_Strict):
    annotation_version: str
    verified_on: str
    source_record: str
    circuit: _CircuitInfo
    model: _ModelRequirement
    sources: dict[str, dict[str, str]]
    roles: list[_Role]
    members: list[_Member]
    relations: list[_Relation]
    not_imported: list[str]


def load_annotations(path: Path = ANNOTATIONS_PATH) -> IoiAnnotations:
    return IoiAnnotations.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def _check_model(model: ModelInfo, required: _ModelRequirement) -> None:
    """Published roles belong to one model: never attach them to another one silently."""
    got = (model.id, model.hf_id, model.layers, model.heads_per_layer)
    want = (required.id, required.hf_id, required.layers, required.heads_per_layer)
    if got != want:
        raise ValueError(f"IOI annotations are for {want}, not {got}")


def _receiver(receiver: _Receiver, circuit_id: str) -> tuple[Endpoint, str]:
    if (receiver.role_group is None) == (receiver.special is None):
        raise ValueError("a receiver is exactly one of role_group or special")
    if receiver.special is not None:
        return SpecialEndpoint(kind="special", id=receiver.special), receiver.special
    assert receiver.role_group is not None
    return (
        RoleGroupEndpoint(kind="role_group", circuit_id=circuit_id, role=receiver.role_group),
        receiver.role_group,
    )


def _edges(relation: _Relation, model: ModelInfo, circuit_id: str) -> list[Edge]:
    target, target_slug = _receiver(relation.receiver, circuit_id)
    senders: list[tuple[Endpoint, str]]
    if isinstance(relation.senders, _PerHeadSenders):
        senders = [
            (HeadEndpoint(kind="head", node_id=head_node_id(model.id, layer, head)),
             _head_label(layer, head))
            for layer, head in relation.senders.heads
        ]
    else:
        group = relation.senders.role_group
        senders = [(RoleGroupEndpoint(kind="role_group", circuit_id=circuit_id, role=group),
                    group)]
    return [
        Edge(
            id=f"{circuit_id}:edge:{slug}->{target_slug}:{relation.channel}",
            circuit_id=circuit_id,
            source=source,
            target=target,
            channel=relation.channel,
            positions=relation.positions,
            evidence_kind="published_experimental",
            evidence_status="published",
            effect=relation.effect,
            magnitude=None,  # never digitised from a figure
            conditions=list(relation.conditions),
            locator=relation.locator,
            method_reference=None,
            description=relation.description,
        )
        for source, slug in senders
    ]


def import_ioi(model: ModelInfo, annotations: IoiAnnotations | None = None) -> PublishedCircuit:
    """The IOI circuit as bundle records for `model` (must be the GPT-2 Small it was found in)."""
    data = annotations or load_annotations()
    _check_model(model, data.model)
    circuit_id = data.circuit.id
    roles = {r.id: r for r in data.roles}
    for member in data.members:
        if member.role not in roles:
            raise ValueError(f"member {_head_label(member.layer, member.head)}: unknown role "
                             f"{member.role!r}")

    circuit = Circuit(
        id=circuit_id,
        name=data.circuit.name,
        origin="published",
        description=data.circuit.description,
        citation=data.circuit.citation,
        roles=[RoleDef(id=r.id, label=r.label, description=r.description) for r in data.roles],
    )
    paper = f"arXiv:{data.sources['paper']['arxiv']}"
    memberships = [
        Membership(
            circuit_id=circuit_id,
            node_id=head_node_id(model.id, m.layer, m.head),
            role=m.role,
            source_reference=f"Wang et al. (2022), {paper}: {m.locator}",
            evidence_status="published",
            note=m.note,
        )
        for m in data.members
    ]
    edges = [edge for relation in data.relations for edge in _edges(relation, model, circuit_id)]
    provenance = (
        f"IOI annotations v{data.annotation_version} ({paper}; code "
        f"{data.sources['code']['repository']}@{data.sources['code']['file_commit'][:7]}; "
        f"verified {data.verified_on}, see {data.source_record})"
    )
    return PublishedCircuit(circuit, memberships, edges, provenance)
