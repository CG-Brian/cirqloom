"""Cross-object semantic checks that field types cannot express."""

from collections import Counter

from cirqloom_analysis.schema.bundle import Graph, Manifest


class BundleValidationError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Bundle validation failed ({len(problems)} problems):\n{lines}")


def head_node_id(model_id: str, layer: int, head: int) -> str:
    return f"model:{model_id}:head:L{layer}H{head}"


def validate_bundle(manifest: Manifest, graph: Graph) -> None:
    problems: list[str] = []
    model = manifest.model

    # g. both files belong to the same bundle
    if manifest.bundle_id != graph.bundle_id:
        problems.append(f"bundle_id mismatch: {manifest.bundle_id!r} vs {graph.bundle_id!r}")
    if manifest.schema_version != graph.schema_version:
        problems.append("schema_version mismatch between manifest and graph")

    # a. complete, in-range head grid
    expected = model.layers * model.heads_per_layer
    if len(graph.nodes) != expected:
        problems.append(f"expected {expected} head nodes, found {len(graph.nodes)}")
    for node in graph.nodes:
        if node.layer >= model.layers or node.head >= model.heads_per_layer:
            problems.append(f"{node.id}: (layer={node.layer}, head={node.head}) out of range")
        # b. deterministic, zero-based ids
        want = head_node_id(model.id, node.layer, node.head)
        if node.id != want:
            problems.append(f"node id {node.id!r} should be {want!r}")
    for node_id, count in Counter(n.id for n in graph.nodes).items():
        if count > 1:
            problems.append(f"duplicate node id {node_id!r} ({count}x)")

    node_ids = {n.id for n in graph.nodes}
    circuits = {c.id: c for c in graph.circuits}
    experiment_ids = {e.id for e in manifest.experiments}

    # c + d. memberships reference real things and declared roles
    for mem in graph.memberships:
        if mem.node_id not in node_ids:
            problems.append(f"membership references unknown node {mem.node_id!r}")
        circuit = circuits.get(mem.circuit_id)
        if circuit is None:
            problems.append(f"membership references unknown circuit {mem.circuit_id!r}")
        elif mem.role not in {r.id for r in circuit.roles}:
            problems.append(f"role {mem.role!r} not declared in circuit {circuit.id!r}")

    # c + e. observations reference real things and never fake a score
    for obs in graph.observations:
        where = f"observation {obs.detector_id}@{obs.node_id}"
        if obs.node_id not in node_ids:
            problems.append(f"{where}: unknown node")
        if obs.experiment_id not in experiment_ids:
            problems.append(f"{where}: unknown experiment {obs.experiment_id!r}")
        if (obs.score is None) != (obs.status == "unavailable"):
            problems.append(f"{where}: score must be null if and only if status is 'unavailable'")
        if obs.score is not None and obs.sample_count == 0:
            problems.append(f"{where}: a score with sample_count=0 is not a measurement")

    # c. edges connect real nodes
    for edge in graph.edges:
        for end in (edge.source_node_id, edge.target_node_id):
            if end not in node_ids:
                problems.append(f"edge {edge.id!r} references unknown node {end!r}")

    # f. a fixture can never pose as real/published data (and vice versa)
    if manifest.synthetic_fixture:
        if not (manifest.display_warning or "").strip():
            problems.append("synthetic fixtures must carry a non-empty display_warning")
        for circuit in graph.circuits:
            if circuit.origin != "synthetic_fixture":
                problems.append(f"fixture bundle contains non-fixture circuit {circuit.id!r}")
    else:
        for circuit in graph.circuits:
            if circuit.origin == "synthetic_fixture":
                problems.append(f"real bundle contains fixture circuit {circuit.id!r}")

    if problems:
        raise BundleValidationError(problems)
