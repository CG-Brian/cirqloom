"""M1 Phase 2, P2-1: adapter contract (no model weights needed)."""

import numpy as np
import pytest
from attention_helpers import make_valid_pattern

from cirqloom_analysis.detectors.attention import AttentionPatternError
from cirqloom_analysis.models.base import AdapterInfo, BaseModelAdapter, validate_token_ids

BOS = 9
INFO = AdapterInfo(
    model_id="toy",
    hf_id="example/toy",
    revision="abc123",
    tokenizer_id="example/toy",
    n_layers=2,
    n_heads=3,
    n_ctx=8,
    vocab_size=10,
    bos_token_id=BOS,
)


def ids(*rows: list[int]) -> np.ndarray:
    return np.array(rows, dtype=np.int64)


# --- validate_token_ids: valid inputs -------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16, np.uint16])
def test_accepts_any_integer_dtype(dtype):
    validate_token_ids(np.array([[BOS, 1, 2]], dtype=dtype), INFO)


def test_accepts_lone_bos_and_full_context():
    validate_token_ids(ids([BOS]), INFO)  # seq = 1: nothing after position 0 to check
    validate_token_ids(ids([BOS] + [0] * 7), INFO)  # seq == n_ctx


def test_accepts_vocab_edges():
    validate_token_ids(ids([BOS, 0, 8], [BOS, 8, 0]), INFO)


# --- validate_token_ids: type and shape ------------------------------------------------------


@pytest.mark.parametrize("value", [[[BOS, 1, 2]], ((BOS, 1),), None, "9 1 2"])
def test_rejects_non_arrays(value):
    # A nested list passes np.ndim == 2 but has no .dtype.
    with pytest.raises(TypeError, match="must be a numpy array"):
        validate_token_ids(value, INFO)  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3), ()])
def test_rejects_wrong_ndim(shape):
    with pytest.raises(ValueError, match=r"must be \[batch, seq\]"):
        validate_token_ids(np.full(shape, BOS, dtype=np.int64), INFO)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, bool])
def test_rejects_non_integer_dtype(dtype):
    # numpy bool is not an integer subtype (unlike Python, where True is an int).
    with pytest.raises(TypeError, match="integer dtype"):
        validate_token_ids(np.array([[BOS, 1]]).astype(dtype), INFO)


def test_ragged_rows_cannot_even_become_an_array():
    # Why equal lengths need no check: numpy refuses to build a ragged 2-D int array.
    with pytest.raises(ValueError):
        np.array([[BOS, 1, 2], [BOS, 1]], dtype=np.int64)


# --- validate_token_ids: size ---------------------------------------------------------------


def test_rejects_empty_batch():
    with pytest.raises(ValueError, match="no prompts"):
        validate_token_ids(np.empty((0, 3), dtype=np.int64), INFO)


def test_rejects_empty_sequence():
    with pytest.raises(ValueError, match="no positions"):
        validate_token_ids(np.empty((2, 0), dtype=np.int64), INFO)


def test_rejects_longer_than_context():
    with pytest.raises(ValueError, match="seq = 9 exceeds .* n_ctx = 8"):
        validate_token_ids(ids([BOS] + [0] * 8), INFO)


# --- validate_token_ids: values -------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, 10, 999])
def test_rejects_ids_outside_vocab(bad):
    with pytest.raises(ValueError, match=r"must be in 0\.\.9"):
        validate_token_ids(ids([BOS, 1, bad]), INFO)


def test_rejects_rows_without_bos_and_names_them():
    with pytest.raises(ValueError, match=r"position 0 must be BOS .* rows \[1\]"):
        validate_token_ids(ids([BOS, 1, 2], [3, 1, 2]), INFO)


def test_rejects_bos_after_position_0_and_names_the_cell():
    with pytest.raises(ValueError, match=r"after position 0 at \(row, pos\) \[\(1, 2\)\]"):
        validate_token_ids(ids([BOS, 1, 2], [BOS, 1, BOS]), INFO)


def test_checks_run_in_documented_order():
    # Too long AND missing BOS: the size check reports first.
    with pytest.raises(ValueError, match="exceeds"):
        validate_token_ids(ids([0] * 9), INFO)


# --- BaseModelAdapter.run_attention ----------------------------------------------------------


class FakeAdapter(BaseModelAdapter):
    """Returns whatever _run_attention is told to, so the base class checks can be tested."""

    def __init__(self, output=None):
        self.output = output
        self.calls = 0

    @property
    def info(self) -> AdapterInfo:
        return INFO

    def tokenize(self, text: str) -> np.ndarray:
        return ids([BOS, *[int(c) for c in text]])[0]

    def token_strings(self, token_ids: np.ndarray) -> list[str]:
        return [f" t{int(i)}" for i in token_ids]

    def _run_attention(self, token_ids: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.output is not None:
            return self.output
        batch, seq = token_ids.shape
        flat = make_valid_pattern(batch, INFO.n_layers * INFO.n_heads, seq)
        return flat.reshape(batch, INFO.n_layers, INFO.n_heads, seq, seq)


def test_incomplete_adapter_cannot_be_created():
    class NoRun(BaseModelAdapter):
        info = INFO

        def tokenize(self, text):
            return ids([BOS])[0]

    with pytest.raises(TypeError, match="abstract"):
        NoRun()  # type: ignore[abstract]


def test_run_attention_returns_validated_patterns():
    patterns = FakeAdapter().run_attention(ids([BOS, 1, 2, 3], [BOS, 4, 5, 6]))
    assert patterns.shape == (2, 2, 3, 4, 4)  # [batch, layer, head, q, k]


def test_bad_input_never_reaches_the_model():
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="position 0 must be BOS"):
        adapter.run_attention(ids([1, 2, 3]))
    assert adapter.calls == 0


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 3, 3, 3, 3)),  # wrong layer count
        np.zeros((1, 2, 2, 3, 3)),  # wrong head count
        np.zeros((1, 6, 3, 3)),  # layers and heads not separated
        [[0.0]],  # not a numpy array (e.g. a torch tensor leaking out)
    ],
)
def test_wrong_output_shape_or_type_is_rejected(output):
    with pytest.raises(ValueError, match=r"expected \(1, 2, 3, 3, 3\)"):
        FakeAdapter(output).run_attention(ids([BOS, 1, 2]))


def test_invalid_attention_is_rejected():
    # Right shape, but rows do not sum to 1: a broken adapter must not reach a detector.
    with pytest.raises(AttentionPatternError):
        FakeAdapter(np.zeros((1, 2, 3, 3, 3))).run_attention(ids([BOS, 1, 2]))


def test_bundle_model_record():
    record = INFO.to_bundle_model()
    assert (record.id, record.layers, record.heads_per_layer) == ("toy", 2, 3)
    assert record.revision == "abc123"
