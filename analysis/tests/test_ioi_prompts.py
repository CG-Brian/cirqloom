"""IOI prompt suite (M3 step 4): templates, positions, tokenizer checks, and the pipeline.

A word-level fake tokenizer stands in for GPT-2 (no weights). The real tokenizer is checked by
the opt-in test in test_gpt2_adapter.py.
"""

import re

import numpy as np
import pytest
from fake_adapter import BOS, INDUCTION_HEAD, INFO, PREVIOUS_TOKEN_HEAD, ContentAwareAdapter

from cirqloom_analysis.capture.runner import run_experiment, unflatten_heads
from cirqloom_analysis.detectors.registry import default_registry
from cirqloom_analysis.export.bundle_writer import read_bundle, validate_bundle_files, write_bundle
from cirqloom_analysis.export.measured import build_measured_bundle
from cirqloom_analysis.prompts.ioi import (
    BABA_TEMPLATES,
    IOI_SUITE,
    abba,
    build_ioi_suite,
    fill,
    locate,
    single_token_vocabulary,
    template,
)
from cirqloom_analysis.schema.bundle import TokensFile
from cirqloom_analysis.schema.validation import head_node_id, validate_bundle

WORD = re.compile(r" ?[A-Za-z]+| ?[^\sA-Za-z]")


class WordTokenizer:
    """One token per word (with its leading space) or punctuation mark; ids below BOS.

    `split_longer_than` makes long words two tokens, to exercise the one-token filter.
    """

    def __init__(self, split_longer_than: int | None = None) -> None:
        self.vocab: dict[str, int] = {}
        self.split = split_longer_than

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 1
        return self.vocab[piece]

    def tokenize(self, text: str) -> np.ndarray:
        pieces: list[str] = []
        for piece in WORD.findall(text):
            if self.split and len(piece.strip()) > self.split:
                pieces += [piece[: len(piece) // 2], piece[len(piece) // 2:]]
            else:
                pieces.append(piece)
        return np.array([BOS, *(self._id(p) for p in pieces)], dtype=np.int64)

    def token_strings(self, ids: np.ndarray) -> list[str]:
        inverse = {v: k for k, v in self.vocab.items()}
        return ["<bos>" if int(i) == BOS else inverse[int(i)] for i in ids]


@pytest.fixture
def tok():
    return WordTokenizer()


def test_templates_are_the_papers_figure_14():
    assert len(BABA_TEMPLATES) == 15
    assert BABA_TEMPLATES[0] == "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]"
    assert BABA_TEMPLATES[-1] == (
        "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]")
    assert all(t.endswith(" to [A]") and t.count("[B]") == 2 and t.count("[A]") == 2
               for t in BABA_TEMPLATES)


def test_abba_swaps_only_the_first_names():
    assert abba(BABA_TEMPLATES[0]) == (
        "Then, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to [A]")
    assert template(0, "ABBA") == abba(BABA_TEMPLATES[0])
    with pytest.raises(ValueError, match="not a BABA template"):
        abba(template(0, "ABBA"))


def test_prompt_stops_before_the_answer():
    text = fill(BABA_TEMPLATES[5], io="Mary", s="John", place="store", obj="drink")
    assert text == "After John and Mary went to the store, John gave a drink to"


@pytest.mark.parametrize("pattern", ["BABA", "ABBA"])
@pytest.mark.parametrize("index", range(15))
def test_every_template_yields_checked_positions(tok, index, pattern):
    prompts = build_ioi_suite(index, pattern, [0, 1, 2], tok.tokenize, tok.token_strings)
    for p in prompts:
        strings = tok.token_strings(p.token_ids)
        pos = p.positions
        io_name, s_name = strings[pos["IO"]], strings[pos["S1"]]
        assert strings[pos["S2"]] == s_name != io_name
        assert pos["S1+1"] == pos["S1"] + 1
        assert pos["END"] == len(strings) - 1 and strings[pos["END"]] == " to"
        assert pos["S1"] < pos["S2"] < pos["END"]
        # BABA: S1 comes first; ABBA: IO comes first.
        assert (pos["S1"] < pos["IO"]) == (pattern == "BABA")
        assert p.text is not None and p.text.endswith(" to")
        assert p.id == f"{IOI_SUITE}:{pattern}-t{index:02d}:seed{p.seed}"
    # One experiment shares length and positions.
    assert len({p.token_ids.shape for p in prompts}) == 1
    assert all(p.positions == prompts[0].positions for p in prompts)


def test_prompts_are_deterministic_per_seed(tok):
    a = build_ioi_suite(0, "BABA", [0, 1, 2], tok.tokenize, tok.token_strings)
    b = build_ioi_suite(0, "BABA", [2, 0], tok.tokenize, tok.token_strings)
    assert a[0].text == b[1].text and a[2].text == b[0].text
    assert len({p.text for p in a}) == 3


def test_multi_token_fillers_are_dropped_and_reported():
    tok = WordTokenizer(split_longer_than=7)
    vocab = single_token_vocabulary(tok.tokenize)
    assert "Christopher" in vocab.dropped and "Christopher" not in vocab.names
    assert "restaurant" in vocab.dropped and "basketball" in vocab.dropped
    assert all(len(w) <= 7 for w in (*vocab.names, *vocab.places, *vocab.objects))
    build_ioi_suite(7, "ABBA", list(range(8)), tok.tokenize, tok.token_strings)


def test_locate_rejects_prompts_that_break_the_ioi_structure(tok):
    ids = tok.tokenize("Then, John and Mary went. John gave it to Mary")
    io, s = int(tok.tokenize(" Mary")[1]), int(tok.tokenize(" John")[1])
    with pytest.raises(ValueError, match="IO occurs 2 times") as info:
        locate(ids, io, s, " to", tok.token_strings)
    assert "last token is ' Mary'" in str(info.value)


def test_suite_rejects_bad_seeds_and_templates(tok):
    with pytest.raises(ValueError, match="empty"):
        build_ioi_suite(0, "BABA", [], tok.tokenize, tok.token_strings)
    with pytest.raises(ValueError, match="distinct"):
        build_ioi_suite(0, "BABA", [1, 1], tok.tokenize, tok.token_strings)
    with pytest.raises(ValueError, match="template index"):
        template(15, "BABA")
    with pytest.raises(ValueError, match="pattern"):
        template(0, "AABB")  # type: ignore[arg-type]


# --- Pipeline -------------------------------------------------------------------------------


@pytest.fixture
def ioi_run(tok):
    prompts = build_ioi_suite(0, "BABA", [0, 1, 2, 3], tok.tokenize, tok.token_strings)
    return run_experiment(ContentAwareAdapter(), prompts, default_registry()), tok


def test_only_ioi_detectors_run_on_ioi_prompts_with_positions_recorded(ioi_run):
    run, _ = ioi_run
    assert run.experiment_id == "ioi_templates_v1:BABA-t00"
    assert sorted(run.results) == [
        "ioi_end_to_io_attention", "ioi_end_to_s1_attention", "ioi_end_to_s2_attention",
        "ioi_s2_to_s1_attention", "ioi_s2_to_s1plus1_attention",
    ]
    pos = run.prompts[0].positions
    assert run.results["ioi_end_to_io_attention"].parameters == {"query": pos["END"],
                                                                 "key": pos["IO"]}


def test_scores_read_the_right_cells(ioi_run):
    run, _ = ioi_run
    mean = lambda d: unflatten_heads(run.results[d].summary.mean, INFO.n_layers, INFO.n_heads)  # noqa: E731
    # The fake induction head attends from S2 to the token after S1: exactly S1+1.
    assert mean("ioi_s2_to_s1plus1_attention")[INDUCTION_HEAD] == 1.0
    assert mean("ioi_s2_to_s1_attention")[INDUCTION_HEAD] == 0.0
    # The fake previous-token head attends from S2 to S2-1, which is none of the named cells.
    assert mean("ioi_s2_to_s1_attention")[PREVIOUS_TOKEN_HEAD] == 0.0


def test_ioi_experiment_exports_text_positions_and_single_cell_masks(ioi_run, tmp_path):
    run, tok = ioi_run
    manifest, graph, assets = build_measured_bundle(
        [run], INFO, tok.token_strings, bundle_id="toy:ioi:v1", generator="test",
        environment={}, notes="test",
    )
    validate_bundle(manifest, graph)
    exp = manifest.experiments[0]
    assert exp.kind == "measurement" and exp.attention is not None
    assert "IOI template BABA-t00" in exp.description
    pos = run.prompts[0].positions
    masks = {m.detector_id: m.cells_qk for m in graph.masks}
    assert masks["ioi_end_to_s2_attention"] == [(pos["END"], pos["S2"])]
    assert {o.status for o in graph.observations} == {"experimental"}
    obs = next(o for o in graph.observations if o.node_id == head_node_id("toy", *INDUCTION_HEAD)
               and o.detector_id == "ioi_s2_to_s1plus1_attention")
    assert obs.score == 1.0 and obs.sample_count == 4

    write_bundle(manifest, graph, tmp_path, assets)
    validate_bundle_files(tmp_path, read_bundle(tmp_path)[0])
    tokens = TokensFile.model_validate_json((tmp_path / exp.tokens_uri).read_text())
    assert tokens.prompts[0].positions == dict(pos)
    assert tokens.prompts[0].text == run.prompts[0].text


def test_file_validation_rejects_positions_that_differ_or_overflow(ioi_run, tmp_path):
    run, tok = ioi_run
    manifest, graph, assets = build_measured_bundle(
        [run], INFO, tok.token_strings, bundle_id="toy:ioi:v1", generator="test",
        environment={}, notes="test",
    )
    write_bundle(manifest, graph, tmp_path, assets)
    path = tmp_path / manifest.experiments[0].tokens_uri
    tokens = TokensFile.model_validate_json(path.read_text())
    tokens.prompts[1].positions = {**tokens.prompts[1].positions, "END": 99}
    path.write_text(tokens.model_dump_json())
    with pytest.raises(Exception) as info:
        validate_bundle_files(tmp_path, manifest)
    problems = " ".join(info.value.problems)
    assert "positions out of range" in problems and "do not share the same positions" in problems
