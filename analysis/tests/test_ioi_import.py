"""IOI import (M3 step 2b): every member and all 27 relations against the paper.

The expected tables below are copied independently from Wang et al. (2022), arXiv:2211.00593v1
(see docs/sources/ioi-wang-2022.md), not generated from annotations.json, so a change to either
side has to be made deliberately on both.
"""

import ast
from pathlib import Path

import pytest

from cirqloom_analysis.published.ioi.importer import (
    ANNOTATIONS_PATH,
    import_ioi,
    load_annotations,
)
from cirqloom_analysis.schema.bundle import (
    HeadEndpoint,
    ModelInfo,
    RoleGroupEndpoint,
    SpecialEndpoint,
)

GPT2 = ModelInfo(id="gpt2-small", hf_id="openai-community/gpt2", revision="any", layers=12,
                 heads_per_layer=12, tokenizer_id="openai-community/gpt2")
CIRCUIT = "circuit:gpt2-small:ioi:wang-2022"

# Figure 2 (p. 4); parenthesised heads are the "fuzzy" members of Appendix C.
MEMBERS = {
    "previous_token": {(2, 2), (4, 11)},
    "duplicate_token": {(0, 1), (3, 0), (0, 10)},
    "induction": {(5, 5), (6, 9), (5, 8), (5, 9)},
    "s_inhibition": {(7, 3), (7, 9), (8, 6), (8, 10)},
    "name_mover": {(9, 9), (9, 6), (10, 0)},
    "negative_name_mover": {(10, 7), (11, 10)},
    "backup_name_mover": {(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)},
}
FUZZY = {(0, 10), (5, 8), (5, 9)}
# Commented out in the official CIRCUIT dict: not members.
CODE_COMMENTED_OUT = {(7, 1), (2, 9), (4, 3), (4, 7), (5, 6), (3, 3), (3, 7), (3, 6)}

# (sender, receiver, channel, positions, effect direction, conditions, locator must contain)
# sender: (layer, head) for a per-head result, or a role id for a group-level result.
# receiver: a role id or "logits".
NM_LOGITS = ("logits", "direct_logit", ("END", None))
RELATIONS = (
    [((9, 9), *NM_LOGITS, "increase", (), "Figure 3a"),
     ((9, 6), *NM_LOGITS, "increase", (), "Figure 3a"),
     ((10, 0), *NM_LOGITS, "increase", (), "Figure 3a"),
     ((10, 7), *NM_LOGITS, "decrease", (), "Figure 3a"),
     ((11, 10), *NM_LOGITS, "decrease", (), "Figure 3a")]
    + [(h, "name_mover", "q", ("END", "END"), None, (), "Figure 4b")
       for h in [(7, 3), (7, 9), (8, 6), (8, 10)]]
    + [("s_inhibition", "name_mover", "q", ("END", "END"), None, (), "Figure 4c")]
    + [(h, "s_inhibition", "v", ("S2", "END"), None, (), "Figure 5b")
       for h in [(0, 1), (3, 0), (5, 5), (6, 9)]]
    + [(h, "s_inhibition", "k", ("S2", "END"), None, (), "Figure 12b")
       for h in [(0, 10), (5, 8), (5, 9)]]
    + [(h, "induction", "k", ("S1+1", "S2"), None, (), "Figure 13") for h in [(2, 2), (4, 11)]]
    + [(h, *NM_LOGITS, None, ("after_knockout:name_mover",), "Figure 15")
       for h in [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)]]
)


def head_of(node_id: str) -> tuple[int, int]:
    layer, head = node_id.rsplit(":L", 1)[1].split("H")
    return int(layer), int(head)


def endpoint_key(end):
    if isinstance(end, HeadEndpoint):
        return head_of(end.node_id)
    if isinstance(end, RoleGroupEndpoint):
        assert end.circuit_id == CIRCUIT
        return end.role
    assert isinstance(end, SpecialEndpoint)
    return end.id


@pytest.fixture(scope="module")
def ioi():
    return import_ioi(GPT2)


def test_seven_roles_and_26_members_match_figure_2(ioi):
    assert [r.id for r in ioi.circuit.roles] == list(MEMBERS)
    got: dict[str, set[tuple[int, int]]] = {}
    for m in ioi.memberships:
        got.setdefault(m.role, set()).add(head_of(m.node_id))
    assert got == MEMBERS
    assert len(ioi.memberships) == 26 == len({m.node_id for m in ioi.memberships})


def test_every_member_is_published_with_a_paper_locator(ioi):
    for m in ioi.memberships:
        assert m.circuit_id == CIRCUIT and m.evidence_status == "published"
        assert m.source_reference.startswith("Wang et al. (2022), arXiv:2211.00593v1: ")
        assert "Figure 2" in m.source_reference


def test_only_the_three_fuzzy_heads_carry_a_note(ioi):
    noted = {head_of(m.node_id) for m in ioi.memberships if m.note}
    assert noted == FUZZY
    assert all("Appendix C" in m.note for m in ioi.memberships if m.note)


def test_heads_commented_out_in_the_code_are_not_members(ioi):
    members = {head_of(m.node_id) for m in ioi.memberships}
    assert members.isdisjoint(CODE_COMMENTED_OUT)


def test_circuit_is_published_with_full_citation(ioi):
    c = ioi.circuit
    assert c.id == CIRCUIT and c.origin == "published"
    assert "arXiv:2211.00593v1" in (c.citation or "")
    assert "Interpretability in the Wild" in (c.citation or "")


def test_all_27_relations_match_the_paper(ioi):
    got = {
        (endpoint_key(e.source), endpoint_key(e.target), e.channel,
         (e.positions.source, e.positions.target) if e.positions else None,
         e.effect.direction if e.effect else None, tuple(e.conditions))
        for e in ioi.edges
    }
    want = {r[:6] for r in RELATIONS}
    assert len(ioi.edges) == len(RELATIONS) == 27
    assert got == want


@pytest.mark.parametrize("relation", RELATIONS, ids=lambda r: f"{r[0]}->{r[1]}:{r[2]}")
def test_each_relation_cites_its_figure(ioi, relation):
    sender, receiver, channel, *_, locator = relation
    edge = next(e for e in ioi.edges if endpoint_key(e.source) == sender
                and endpoint_key(e.target) == receiver and e.channel == channel)
    assert locator in (edge.locator or "")


def test_relations_keep_the_measured_granularity(ioi):
    # Senders are single heads, except the one group-level experiment (Figure 4c).
    group_senders = [e for e in ioi.edges if isinstance(e.source, RoleGroupEndpoint)]
    assert [e.locator for e in group_senders] == ["§3.2, Figure 4c (p. 7)"]
    # Receivers are always a whole role group or the logits: no head-pair edges at all.
    assert not any(isinstance(e.target, HeadEndpoint) for e in ioi.edges)


def test_per_head_senders_are_members_of_the_role_the_paper_names(ioi):
    roles = {head_of(m.node_id): m.role for m in ioi.memberships}
    sender_role = {
        "name_mover": {"name_mover", "negative_name_mover", "backup_name_mover"},
        "s_inhibition": {"duplicate_token", "induction"},
        "induction": {"previous_token"},
    }
    for e in ioi.edges:
        if not isinstance(e.source, HeadEndpoint):
            continue
        receiver = endpoint_key(e.target)
        if receiver == "logits":
            assert roles[head_of(e.source.node_id)] in sender_role["name_mover"]
        elif receiver == "name_mover":
            assert roles[head_of(e.source.node_id)] == "s_inhibition"
        else:
            assert roles[head_of(e.source.node_id)] in sender_role[receiver]


def test_published_edges_carry_no_invented_numbers(ioi):
    for e in ioi.edges:
        assert e.evidence_kind == "published_experimental"
        assert e.evidence_status == "published"
        assert e.magnitude is None and e.method_reference is None
        assert (e.channel == "direct_logit") == (endpoint_key(e.target) == "logits")


def test_edge_ids_are_stable_and_unique(ioi):
    ids = [e.id for e in ioi.edges]
    assert len(set(ids)) == 27
    assert f"{CIRCUIT}:edge:L7H3->name_mover:q" in ids
    assert f"{CIRCUIT}:edge:s_inhibition->name_mover:q" in ids
    assert import_ioi(GPT2).edges == ioi.edges  # deterministic


@pytest.mark.parametrize(
    "other",
    [
        GPT2.model_copy(update={"id": "pythia-160m"}),
        GPT2.model_copy(update={"hf_id": "EleutherAI/pythia-160m"}),
        GPT2.model_copy(update={"layers": 24}),
    ],
)
def test_refuses_to_attach_gpt2_roles_to_another_model(other):
    with pytest.raises(ValueError, match="IOI annotations are for"):
        import_ioi(other)


def test_annotations_record_their_sources():
    data = load_annotations()
    assert data.sources["paper"]["arxiv"] == "2211.00593v1"
    assert data.sources["code"]["file_commit"].startswith("e168301")
    assert data.source_record == "docs/sources/ioi-wang-2022.md"
    assert data.not_imported  # documented exclusions


def test_importer_does_not_touch_models_or_detectors():
    source = (ANNOTATIONS_PATH.parent / "importer.py").read_text()
    imported = {
        node.module or "" for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden = ("torch", "transformer_lens", "cirqloom_analysis.models",
                 "cirqloom_analysis.detectors", "cirqloom_analysis.capture")
    assert not [m for m in imported if m.startswith(forbidden)]
    assert Path(ANNOTATIONS_PATH).is_file()
