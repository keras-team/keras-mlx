"""Tests for inputs that used to end the process instead of raising.

An empty input to `sparsemax` is SIGFPE on the cpu backend. See
`numpy_test.py` for the rest.
"""

import numpy as np
import pytest

from keras.src import ops


@pytest.mark.parametrize("shape", [(0,), (0, 2), (2, 0)])
def test_sparsemax_handles_an_empty_input(shape):
    out = ops.convert_to_numpy(ops.sparsemax(np.zeros(shape, "float32")))

    assert out.shape == shape


@pytest.mark.parametrize(
    "dtype,expected", [("float32", "float32"), ("int32", "float32")]
)
def test_sparsemax_empty_keeps_the_dtype_of_the_full_path(dtype, expected):
    # An integer input promotes to float, so the empty result must too.
    empty = ops.convert_to_numpy(ops.sparsemax(np.zeros((0,), dtype)))
    full = ops.convert_to_numpy(ops.sparsemax(np.zeros((3,), dtype)))

    assert empty.dtype == full.dtype == np.dtype(expected)


def test_sparsemax_still_projects():
    out = ops.convert_to_numpy(ops.sparsemax(np.array([1.0, 2.0, 3.0])))

    assert out.tolist() == [0.0, 0.0, 1.0]
