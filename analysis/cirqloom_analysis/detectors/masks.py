"""Rules shared by the detector mask builders (previous-token, first-token, ...).

Burn-in: the first `burn_in` destination rows are left out of a mask. Early rows have few
sources to choose from, so they carry little evidence about what a head does:
  - row 0 can only attend to itself, so it is 1.0 for every head;
  - row 1 can only choose position 0 or itself, so a previous-token head and an attention-sink
    head both put their weight on (1, 0) and cannot be told apart there.
Using the same burn-in for every detector also gives them the same destination rows, so their
supports match and scores can be compared side by side.
"""

import numpy as np

DEFAULT_BURN_IN = 2
"""Skip rows 0 and 1: the safe default (AGENTS.md §0 "do not overclaim")."""

MIN_BURN_IN = 1
"""Row 0 is never evidence. burn_in=1 keeps row 1, which is allowed but ambiguous."""


def check_integer(name: str, value: object) -> None:
    """Raise TypeError unless `value` is an integer (Python or numpy) and not a bool.

    For settings that come from configs or prompt suites (burn_in, repeat length), whose type
    is not guaranteed the way a dimension size like pattern.shape[-1] is.
    bool is a subclass of int (True == 1), and numpy integers (e.g. np.int64) are not
    subclasses of int, so both need explicit handling.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


def check_burn_in(seq: int, burn_in: int) -> None:
    """Raise if `burn_in` is not a usable burn-in for a sequence of length `seq`.

    Order matters: the type is checked first, so the range checks never compare a str or
    float.
    """
    # Check 1: an integer, but not a bool (see check_integer).
    check_integer("burn_in", burn_in)

    # Check 2: never include row 0, which is 1.0 for every head.
    if burn_in < MIN_BURN_IN:
        raise ValueError(f"burn_in must be >= {MIN_BURN_IN}, got {burn_in}")

    # Check 3: at least one destination row must remain after skipping `burn_in` rows.
    if seq < burn_in + 1:
        raise ValueError(f"burn_in={burn_in} needs seq >= {burn_in + 1}, got {seq}")
