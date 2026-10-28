"""M1 Phase 2, P2-2: the real GPT-2 adapter. Downloads ~500 MB of weights on first run.

Skipped unless CIRQLOOM_REAL_MODEL=1 (AGENTS.md §12: normal test runs never download weights):

    CIRQLOOM_REAL_MODEL=1 pytest tests/test_gpt2_adapter.py
"""

import os

import numpy as np
import pytest

if os.environ.get("CIRQLOOM_REAL_MODEL") != "1":
    pytest.skip("set CIRQLOOM_REAL_MODEL=1 to run real-model tests", allow_module_level=True)
pytest.importorskip("transformer_lens")

from cirqloom_analysis.models.gpt2 import REVISION, GPT2Adapter  # noqa: E402

BOS = 50256


@pytest.fixture(scope="module")
def adapter() -> GPT2Adapter:
    return GPT2Adapter(device="cpu", seed=0)


def repeated_prompt(repeat_len: int, seed: int) -> np.ndarray:
    # (BOS, X, X) with X drawn from non-special ids. Phase 3 will own prompt generation.
    x = np.random.default_rng(seed).integers(0, BOS, size=repeat_len)
    return np.concatenate([[BOS], x, x]).astype(np.int64)


def test_info_matches_gpt2_small(adapter):
    info = adapter.info
    assert (info.n_layers, info.n_heads) == (12, 12)  # M1 "done when": 12×12 verified
    assert (info.n_ctx, info.vocab_size, info.bos_token_id) == (1024, 50257, BOS)
    assert info.revision == REVISION
    assert info.to_bundle_model().heads_per_layer == 12


def test_tokenize_puts_bos_first_and_only_first(adapter):
    ids = adapter.tokenize("When Mary and John went to the store")
    assert ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer)
    assert ids[0] == BOS
    assert BOS not in ids[1:]


def test_run_attention_shape_and_validity(adapter):
    batch = np.stack([repeated_prompt(5, seed) for seed in range(2)])
    patterns = adapter.run_attention(batch)  # base class validates rows, causality, shape
    assert patterns.shape == (2, 12, 12, 11, 11)
    assert patterns.dtype == np.float32


def test_tokenized_text_runs_end_to_end(adapter):
    ids = adapter.tokenize("The cat sat on the mat.")
    patterns = adapter.run_attention(ids[None, :])
    assert patterns.shape[:3] == (1, 12, 12)


def test_same_input_gives_identical_patterns(adapter):
    batch = repeated_prompt(5, seed=0)[None, :]
    np.testing.assert_array_equal(adapter.run_attention(batch), adapter.run_attention(batch))


def test_row_in_batch_does_not_depend_on_other_rows(adapter):
    # No padding and no cross-row leakage: a prompt gives the same pattern alone or batched.
    a, b = repeated_prompt(5, seed=0), repeated_prompt(5, seed=1)
    alone = adapter.run_attention(a[None, :])[0]
    batched = adapter.run_attention(np.stack([a, b]))[0]
    np.testing.assert_allclose(alone, batched, atol=1e-5)
