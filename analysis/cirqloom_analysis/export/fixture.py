"""Synthetic GPT-2-shaped demo fixture. Contains NO measured model data."""

from cirqloom_analysis.schema.bundle import (
    SCHEMA_VERSION,
    Circuit,
    Graph,
    Manifest,
    Membership,
    ModelInfo,
    Node,
    Provenance,
    RoleDef,
)
from cirqloom_analysis.schema.validation import head_node_id

MODEL_ID = "gpt2-small"
CIRCUIT_ID = f"circuit:{MODEL_ID}:fixture-demo"
FIXTURE_REF = "synthetic fixture (build_fixture_bundle.py)"


def build_fixture() -> tuple[Manifest, Graph]:
    """Generates a synthetic GPT-2-shaped bundle for UI and schema testing."""

    model = ModelInfo(
        id=MODEL_ID,
        hf_id="openai-community/gpt2",
        revision=None,
        layers=12,
        heads_per_layer=12,
        tokenizer_id="openai-community/gpt2",
    )
    nodes = [
        Node(id=head_node_id(MODEL_ID, layer, head), type="attention_head", layer=layer, head=head)
        for layer in range(model.layers)
        for head in range(model.heads_per_layer)
    ]
    circuit = Circuit(
        id=CIRCUIT_ID,
        name="Demo fixture circuit",
        origin="synthetic_fixture",
        description="Deliberately artificial diagonal pattern (LkHk). Exercises the UI only; "
        "not a finding about GPT-2.",
        citation=None,
        roles=[
            RoleDef(id="fixture_role_a", label="Fixture role A", description="Even layers."),
            RoleDef(id="fixture_role_b", label="Fixture role B", description="Odd layers."),
        ],
    )
    memberships = [
        Membership(
            circuit_id=CIRCUIT_ID,
            node_id=head_node_id(MODEL_ID, k, k),
            role="fixture_role_a" if k % 2 == 0 else "fixture_role_b",
            source_reference=FIXTURE_REF,
            evidence_status="experimental",
            note=None,
        )
        for k in range(min(model.layers, model.heads_per_layer))
    ]
    bundle_id = "fixture:gpt2-small-demo:v1"
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        bundle_id=bundle_id,
        synthetic_fixture=True,
        display_warning="Demo fixture — not measured model data.",
        model=model,
        experiments=[],
        provenance=Provenance(
            generator="analysis/scripts/build_fixture_bundle.py",
            generator_version="1",
            notes="No model weights loaded. No detectors run.",
            environment={},
        ),
        graph_uri="graph.json",
    )
    graph = Graph(
        schema_version=SCHEMA_VERSION,
        bundle_id=bundle_id,
        nodes=nodes,
        observations=[],
        masks=[],
        circuits=[circuit],
        memberships=memberships,
        edges=[],
    )
    return manifest, graph
