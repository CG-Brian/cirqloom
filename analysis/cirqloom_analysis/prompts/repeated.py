"""Repeated random-token prompts and their non-repeated control (M1 Phase 3, AGENTS.md §7.4).

    repeated_tokens_v1      (BOS, X, X)   X: repeat_len random tokens
    nonrepeated_tokens_v1   (BOS, X, Y)   Y: repeat_len fresh random tokens, same length

Random tokens, not text: in a repeated sentence a head could predict the second copy from
language statistics alone; with random tokens the only way to do it is to copy from context.

Paired design: the control prompt with the same (repeat_len, seed) has the *same* BOS and X
as the repeated prompt; only the second half differs. Every mask therefore selects the same
cells in both, and the only change is whether the content repeats.

Each prompt has its own generator, seeded by (suite salt, repeat_len, seed). Adding lengths or
seeds later never changes existing prompts, which a single shared generator would do.

Ids are drawn uniformly from 0..bos_token_id-1 (for GPT-2, every id except <|endoftext|>), so
no second BOS appears mid-sequence. Y can share a token with X by chance (about L/50256 per
position); that is left in rather than distorting the distribution, and is negligible.

X itself can also contain a token twice (for L = 50, about 2.5% of prompts with GPT-2's
vocabulary). Then the second copy has two earlier matches and an ideal induction head may
legitimately attend to either; this slightly lowers induction scores and is left in as well.
"""

from dataclasses import dataclass

import numpy as np

from cirqloom_analysis.detectors.induction import MIN_REPEAT_LEN
from cirqloom_analysis.detectors.masks import check_integer
from cirqloom_analysis.models.base import AdapterInfo

REPEATED_SUITE = "repeated_tokens_v1"
CONTROL_SUITE = "nonrepeated_tokens_v1"

# Fixed salt so these streams differ from any other use of the same small seeds.
_SALT = 20_221_000


@dataclass(frozen=True)
class Prompt:
    """One generated input. token_ids is int64 [1 + 2 * repeat_len], BOS first."""

    id: str
    suite: str
    repeat_len: int
    seed: int
    token_ids: np.ndarray


def _check(repeat_len: int, seed: int, info: AdapterInfo) -> None:
    check_integer("repeat_len", repeat_len)
    check_integer("seed", seed)
    if repeat_len < MIN_REPEAT_LEN:
        raise ValueError(f"repeat_len must be >= {MIN_REPEAT_LEN}, got {repeat_len}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if 1 + 2 * repeat_len > info.n_ctx:
        raise ValueError(
            f"repeat_len={repeat_len} gives seq {1 + 2 * repeat_len} > n_ctx {info.n_ctx}"
        )
    if info.bos_token_id < 1:
        raise ValueError("random ids are drawn from 0..bos_token_id-1, which is empty")
    if info.bos_token_id != info.vocab_size - 1:
        # Holds for GPT-2. Another tokenizer needs an explicit list of excluded special ids.
        raise ValueError("this generator assumes BOS is the last vocabulary id")


def _halves(repeat_len: int, seed: int, info: AdapterInfo) -> tuple[np.ndarray, np.ndarray]:
    """X and Y for one (repeat_len, seed). X is shared by the repeated and control prompt."""
    _check(repeat_len, seed, info)
    rng = np.random.default_rng([_SALT, repeat_len, seed])
    x = rng.integers(0, info.bos_token_id, size=repeat_len, dtype=np.int64)
    y = rng.integers(0, info.bos_token_id, size=repeat_len, dtype=np.int64)
    return x, y


def _prompt(suite: str, repeat_len: int, seed: int, ids: np.ndarray) -> Prompt:
    return Prompt(
        id=f"{suite}:L{repeat_len}:seed{seed}",
        suite=suite,
        repeat_len=repeat_len,
        seed=seed,
        token_ids=ids,
    )


def repeated_prompt(repeat_len: int, seed: int, info: AdapterInfo) -> Prompt:
    """(BOS, X, X)."""
    x, _ = _halves(repeat_len, seed, info)
    ids = np.concatenate([[info.bos_token_id], x, x]).astype(np.int64)
    return _prompt(REPEATED_SUITE, repeat_len, seed, ids)


def control_prompt(repeat_len: int, seed: int, info: AdapterInfo) -> Prompt:
    """(BOS, X, Y): same BOS and X as repeated_prompt(repeat_len, seed), no repetition."""
    x, y = _halves(repeat_len, seed, info)
    ids = np.concatenate([[info.bos_token_id], x, y]).astype(np.int64)
    return _prompt(CONTROL_SUITE, repeat_len, seed, ids)


SUITE_BUILDERS = {REPEATED_SUITE: repeated_prompt, CONTROL_SUITE: control_prompt}


def build_suite(
    suite: str, repeat_len: int, seeds: list[int], info: AdapterInfo
) -> list[Prompt]:
    """All prompts of one suite at one length: one experiment, one batch (no padding)."""
    if suite not in SUITE_BUILDERS:
        raise ValueError(f"unknown prompt suite {suite!r}; known: {sorted(SUITE_BUILDERS)}")
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {seeds}")
    return [SUITE_BUILDERS[suite](repeat_len, seed, info) for seed in seeds]
