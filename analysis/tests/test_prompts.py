"""M1 Phase 3: repeated-token prompts and the paired non-repeated control."""

import numpy as np
import pytest
from fake_adapter import BOS, INFO

from cirqloom_analysis.models.base import validate_token_ids
from cirqloom_analysis.prompts.repeated import (
    CONTROL_SUITE,
    REPEATED_SUITE,
    build_suite,
    control_prompt,
    repeated_prompt,
)


def test_repeated_layout_is_bos_x_x():
    p = repeated_prompt(5, seed=0, info=INFO)
    ids = p.token_ids
    assert ids.shape == (11,) and ids.dtype == np.int64
    assert ids[0] == BOS
    np.testing.assert_array_equal(ids[1:6], ids[6:])
    assert p.id == "repeated_tokens_v1:L5:seed0"


def test_control_shares_bos_and_x_but_not_the_second_half():
    r, c = repeated_prompt(20, 3, INFO), control_prompt(20, 3, INFO)
    np.testing.assert_array_equal(r.token_ids[:21], c.token_ids[:21])
    assert not np.array_equal(r.token_ids[21:], c.token_ids[21:])
    assert c.suite == CONTROL_SUITE and c.token_ids.shape == r.token_ids.shape


def test_prompts_are_deterministic():
    a = repeated_prompt(10, 4, INFO).token_ids
    np.testing.assert_array_equal(a, repeated_prompt(10, 4, INFO).token_ids)


def test_adding_lengths_or_seeds_does_not_change_existing_prompts():
    # Each prompt has its own generator, keyed by (repeat_len, seed).
    alone = build_suite(REPEATED_SUITE, 10, [2], INFO)[0].token_ids
    among = build_suite(REPEATED_SUITE, 10, [0, 1, 2, 3], INFO)[2].token_ids
    np.testing.assert_array_equal(alone, among)


def test_different_seeds_and_lengths_differ():
    assert not np.array_equal(
        repeated_prompt(10, 0, INFO).token_ids, repeated_prompt(10, 1, INFO).token_ids
    )
    assert not np.array_equal(
        repeated_prompt(10, 0, INFO).token_ids[1:11], repeated_prompt(11, 0, INFO).token_ids[1:11]
    )


@pytest.mark.parametrize("seed", range(20))
def test_random_part_never_contains_bos_and_passes_adapter_checks(seed):
    for prompt in (repeated_prompt(30, seed, INFO), control_prompt(30, seed, INFO)):
        assert BOS not in prompt.token_ids[1:]
        validate_token_ids(prompt.token_ids[None, :], INFO)


@pytest.mark.parametrize(
    ("repeat_len", "seed", "error", "match"),
    [
        (2, 0, ValueError, "repeat_len must be >= 3"),
        (64, 0, ValueError, "n_ctx"),
        (5, -1, ValueError, "seed must be >= 0"),
        (5.0, 0, TypeError, "repeat_len must be an integer"),
        (5, True, TypeError, "seed must be an integer"),
    ],
)
def test_rejects_bad_settings(repeat_len, seed, error, match):
    with pytest.raises(error, match=match):
        repeated_prompt(repeat_len, seed, INFO)


def test_build_suite_checks_its_inputs():
    with pytest.raises(ValueError, match="unknown prompt suite"):
        build_suite("natural_text", 5, [0], INFO)
    with pytest.raises(ValueError, match="must not be empty"):
        build_suite(REPEATED_SUITE, 5, [], INFO)
    with pytest.raises(ValueError, match="distinct"):
        build_suite(REPEATED_SUITE, 5, [1, 1], INFO)
