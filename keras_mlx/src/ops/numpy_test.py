"""Tests for inputs that used to end the process instead of raising.

A crash takes the worker down before pytest can record it, so none of these
can be expressed in `excluded_tests.txt`.
"""

import numpy as np
import pytest

from keras.src import ops


def test_unravel_index_rejects_a_zero_dimension():
    with pytest.raises(ValueError, match="zero dimension"):
        ops.unravel_index(np.array([3], "int32"), (0, 2))


def test_unravel_index_still_works():
    row, column = ops.unravel_index(np.array([3], "int32"), (2, 2))

    assert ops.convert_to_numpy(row).tolist() == [1]
    assert ops.convert_to_numpy(column).tolist() == [1]


@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_take_along_axis_rejects_non_integer_indices(dtype):
    indices = np.array([0, 1]).astype(dtype)

    with pytest.raises(ValueError, match="must be an integer array"):
        ops.take_along_axis(np.arange(4.0), indices, axis=0)


def test_take_along_axis_still_works():
    out = ops.take_along_axis(np.arange(4.0), np.array([3, 1]), axis=0)

    assert ops.convert_to_numpy(out).tolist() == [3.0, 1.0]


@pytest.mark.parametrize("op_name", ["tril", "triu"])
def test_tril_and_triu_name_the_op_for_input_under_2d(op_name):
    op = getattr(ops, op_name)

    with pytest.raises(ValueError, match=f"`{op_name}` must be at least 2D"):
        op(np.arange(4.0))


@pytest.mark.parametrize("op_name", ["tril", "triu"])
def test_tril_and_triu_still_work(op_name):
    out = getattr(ops, op_name)(np.arange(6.0).reshape(2, 3))

    assert ops.convert_to_numpy(out).shape == (2, 3)


@pytest.mark.parametrize(
    "op_name,extra",
    [
        ("quantile", (0.5,)),
        ("percentile", (50,)),
        ("nanquantile", (0.5,)),
        ("nanpercentile", (50,)),
        ("nanmedian", ()),
    ],
)
def test_quantile_family_rejects_an_empty_input(op_name, extra):
    with pytest.raises(ValueError, match="quantile of an empty array"):
        getattr(ops, op_name)(np.zeros((0,), "float32"), *extra)


@pytest.mark.parametrize("shape,axis", [((0, 3), 1), ((3, 0), 0)])
def test_quantile_rejects_an_empty_input_it_does_not_reduce(shape, axis):
    with pytest.raises(ValueError, match="quantile of an empty array"):
        ops.quantile(np.zeros(shape, "float32"), 0.5, axis=axis)


@pytest.mark.parametrize(
    "op_name,extra",
    [
        ("quantile", (0.5,)),
        ("percentile", (50,)),
        ("nanquantile", (0.5,)),
        ("nanpercentile", (50,)),
        ("nanmedian", ()),
        ("median", ()),
    ],
)
def test_quantile_family_still_works(op_name, extra):
    out = getattr(ops, op_name)(np.array([1.0, 2.0, 3.0, 4.0]), *extra)

    assert float(ops.convert_to_numpy(out)) == 2.5


def test_tri_still_works():
    assert ops.convert_to_numpy(ops.tri(3)).shape == (3, 3)
