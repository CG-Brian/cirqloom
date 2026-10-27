"""Checkpoint 4: shared burn-in rule."""

import numpy as np
import pytest

from cirqloom_analysis.detectors.masks import (
    DEFAULT_BURN_IN,
    MIN_BURN_IN,
    check_burn_in,
    check_integer,
)


@pytest.mark.parametrize("value", [0, 7, -3, np.int64(2), np.uint8(1)])
def test_check_integer_accepts_python_and_numpy_ints(value):
    check_integer("x", value)


@pytest.mark.parametrize("value", [True, 2.0, "2", None, np.float64(2)])
def test_check_integer_rejects_other_types_and_names_the_setting(value):
    with pytest.raises(TypeError, match="my_setting must be an integer"):
        check_integer("my_setting", value)


def test_defaults():
    assert DEFAULT_BURN_IN == 2
    assert MIN_BURN_IN == 1


@pytest.mark.parametrize("burn_in", [1, 2, 5, np.int64(2), np.int32(3)])
def test_accepts_python_and_numpy_integers(burn_in):
    check_burn_in(10, burn_in)


@pytest.mark.parametrize("burn_in", [True, False, 2.0, "2", None])
def test_rejects_non_integers_with_type_error(burn_in):
    # bool is an int subclass and 2.0 compares equal to 2, so both must be rejected explicitly.
    with pytest.raises(TypeError, match="must be an integer"):
        check_burn_in(10, burn_in)


@pytest.mark.parametrize("burn_in", [0, -1])
def test_rejects_burn_in_below_minimum(burn_in):
    # 0 would include row 0 (always 1.0); -1 would slice from the end of the mask.
    with pytest.raises(ValueError, match=r"burn_in must be >= 1"):
        check_burn_in(10, burn_in)


def test_minimum_seq_is_accepted():
    check_burn_in(3, 2)  # exactly one row (row 2) remains
    check_burn_in(2, 1)


@pytest.mark.parametrize(("seq", "burn_in"), [(2, 2), (1, 1), (4, 5)])
def test_rejects_seq_with_no_row_left(seq, burn_in):
    expected = rf"burn_in={burn_in} needs seq >= {burn_in + 1}, got {seq}"
    with pytest.raises(ValueError, match=expected):
        check_burn_in(seq, burn_in)


def test_type_is_checked_before_range():
    # A str must give a clear TypeError, not a comparison error from the range checks.
    with pytest.raises(TypeError):
        check_burn_in(1, "9")
