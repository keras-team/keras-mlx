"""Tests for Python list inputs to ops and for `Variable.__array__`."""

import numpy as np
from absl.testing import parameterized

import keras
from keras import ops
from keras.src import testing


class ListInputsTest(testing.TestCase):
    # keras checks list inputs only for the ops ported to its OpTestCase.
    @parameterized.named_parameters(
        ("fmax", ops.fmax, np.fmax),
        ("fmin", ops.fmin, np.fmin),
        ("floor_divide", ops.floor_divide, np.floor_divide),
    )
    def test_binary_op(self, op, reference_fn):
        x = [[1, 2, 3], [3, 2, 1]]
        y = [[4, 5, 6], [3, 2, 1]]
        self.assertAllClose(op(x, y), reference_fn(x, y))
        self.assertDType(op(x, y), "int32")

    def test_binary_op_promotes_list_dtype(self):
        self.assertDType(ops.add([1.5, 2.5], [1, 2]), "float32")
        self.assertDType(ops.fmax([1, 2], 1.5), "float32")


class VariableArrayTest(testing.TestCase):
    def test_no_copy_with_dtype_change_raises(self):
        v = keras.Variable(np.array([1, 2, 3], dtype="float32"))
        with self.assertRaises(ValueError):
            v.__array__(dtype="int32", copy=False)
