"""Cross-object semantic checks that field types cannot express."""

from collections import Counter

from cirqloom_analysis.schema.bundle import (
    Edge,
    Endpoint,
    Graph,
    HeadEndpoint,
    Manifest,
    RoleGroupEndpoint,
    SpecialEndpoint,
)

PUBLISHED_KINDS = {"published_schematic", "published_experimental"}
MEASURED_KINDS = {"weight_composition", "causal_intervention"}


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

    # h. experiments are unique and their attention asset matches their declared shape
    for exp_id, count in Counter(e.id for e in manifest.experiments).items():
        if count > 1:
            problems.append(f"duplicate experiment id {exp_id!r} ({count}x)")
    for exp in manifest.experiments:
        if len(set(exp.seeds)) != len(exp.seeds):
            problems.append(f"experiment {exp.id!r}: seeds are not distinct")
        if exp.attention is not None:
            seq = exp.sequence_length
            want = [model.layers, model.heads_per_layer, seq, seq]
            if exp.attention.shape != want:
                problems.append(
                    f"experiment {exp.id!r}: attention shape {exp.attention.shape}, expected {want}"
                )
            if exp.attention.prompt_index >= len(exp.seeds):
                problems.append(f"experiment {exp.id!r}: attention prompt_index out of range")

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
        if len(obs.per_example_scores) != (obs.sample_count if obs.score is not None else 0):
            problems.append(f"{where}: per_example_scores must hold one score per example")
        if (obs.score_std is None) != (obs.score is None or obs.sample_count < 2):
            problems.append(f"{where}: score_std must be null exactly when fewer than 2 examples")

    # i. one observation per (experiment, detector, node)
    keys = Counter((o.experiment_id, o.detector_id, o.node_id) for o in graph.observations)
    for key, count in keys.items():
        if count > 1:
            problems.append(f"duplicate observation {key} ({count}x)")

    # j. masks: one per (experiment, detector), in range, causal, and matching the observations
    #    they explain. Observations without a mask are allowed (not every detector is a mask).
    experiments = {e.id: e for e in manifest.experiments}
    explained: dict[tuple[str, str], set[tuple[str, tuple[tuple[str, int], ...]]]] = {}
    for obs in graph.observations:
        setting = (obs.detector_version, tuple(sorted(obs.parameters.items())))
        explained.setdefault((obs.experiment_id, obs.detector_id), set()).add(setting)
    for key, count in Counter((m.experiment_id, m.detector_id) for m in graph.masks).items():
        if count > 1:
            problems.append(f"duplicate mask {key} ({count}x)")
    for mask in graph.masks:
        where = f"mask {mask.detector_id}@{mask.experiment_id}"
        exp = experiments.get(mask.experiment_id)
        if exp is None:
            problems.append(f"{where}: unknown experiment")
        else:
            seq = exp.sequence_length
            bad = [c for c in mask.cells_qk if not 0 <= c[1] <= c[0] < seq]
            if bad:
                problems.append(f"{where}: cells outside 0 <= k <= q < {seq}: {bad[:3]}")
        if mask.cells_qk != sorted(set(mask.cells_qk)):
            problems.append(f"{where}: cells_qk must be sorted and unique")
        settings = explained.get((mask.experiment_id, mask.detector_id))
        if settings is None:
            problems.append(f"{where}: no observation uses this mask")
        elif settings != {(mask.detector_version, tuple(sorted(mask.parameters.items())))}:
            problems.append(f"{where}: detector_version/parameters differ from its observations")

    # k. edges: real endpoints inside their own circuit, a channel that fits the target, and
    #    provenance that fits the evidence kind
    for edge_id, count in Counter(e.id for e in graph.edges).items():
        if count > 1:
            problems.append(f"duplicate edge id {edge_id!r} ({count}x)")
    members = {(m.circuit_id, m.node_id) for m in graph.memberships}
    for edge in graph.edges:
        problems.extend(_edge_problems(edge, circuits, node_ids, members))

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


def _endpoint_problems(
    edge: Edge,
    end: Endpoint,
    side: str,
    circuits: dict,
    node_ids: set[str],
    members: set[tuple[str, str]],
) -> list[str]:
    where = f"edge {edge.id!r} {side}"
    if isinstance(end, HeadEndpoint):
        if end.node_id not in node_ids:
            return [f"{where}: unknown node {end.node_id!r}"]
        if (edge.circuit_id, end.node_id) not in members:
            return [f"{where}: {end.node_id!r} is not a member of {edge.circuit_id!r}"]
    elif isinstance(end, RoleGroupEndpoint):
        if end.circuit_id != edge.circuit_id:
            return [f"{where}: role group belongs to another circuit {end.circuit_id!r}"]
        circuit = circuits.get(end.circuit_id)
        if circuit is not None and end.role not in {r.id for r in circuit.roles}:
            return [f"{where}: role {end.role!r} not declared in {end.circuit_id!r}"]
    elif isinstance(end, SpecialEndpoint):
        # logits only receive, embed only sends.
        if (end.id == "logits") != (side == "target"):
            return [f"{where}: {end.id!r} cannot be the {side}"]
    return []


def _edge_problems(
    edge: Edge, circuits: dict, node_ids: set[str], members: set[tuple[str, str]]
) -> list[str]:
    where = f"edge {edge.id!r}"
    problems: list[str] = []
    circuit = circuits.get(edge.circuit_id)
    if circuit is None:
        problems.append(f"{where}: unknown circuit {edge.circuit_id!r}")
    for side, end in (("source", edge.source), ("target", edge.target)):
        problems.extend(_endpoint_problems(edge, end, side, circuits, node_ids, members))
    if edge.source == edge.target:
        problems.append(f"{where}: source and target are the same")

    to_logits = isinstance(edge.target, SpecialEndpoint) and edge.target.id == "logits"
    if (edge.channel == "direct_logit") != to_logits:
        problems.append(f"{where}: channel 'direct_logit' is used exactly for edges into logits")

    if edge.evidence_kind in PUBLISHED_KINDS:
        if circuit is not None and circuit.origin != "published":
            problems.append(f"{where}: published evidence in a non-published circuit")
        if edge.evidence_status != "published":
            problems.append(f"{where}: published evidence must have status 'published'")
        if not (edge.locator or "").strip():
            problems.append(f"{where}: published evidence needs a locator in the citation")
    else:  # measured by us
        if edge.evidence_status == "published":
            problems.append(f"{where}: our own measurement cannot have status 'published'")
        if not (edge.method_reference or "").strip():
            problems.append(f"{where}: a measured edge needs a method_reference")
        if edge.magnitude is None:
            problems.append(f"{where}: a measured edge needs a magnitude")
    if edge.evidence_kind == "weight_composition" and edge.effect is not None:
        problems.append(f"{where}: weight composition has no effect on a metric")
    if edge.evidence_kind == "causal_intervention" and not edge.conditions:
        problems.append(f"{where}: a causal edge must state the conditions it was measured under")
    return problems
