"""Tests for `linspace` cases the keras op tests do not cover."""

import mlx.core as mx
import numpy as np
from absl.testing import parameterized

from keras import ops
from keras.src import testing


class LinspaceTest(testing.TestCase):
    def test_traced_scalar_bounds(self):
        f = mx.compile(lambda a, b: ops.linspace(a, b, 5))
        self.assertAllClose(
            f(mx.array(-1.0), mx.array(3.0)), np.linspace(-1.0, 3.0, 5)
        )

    @parameterized.named_parameters(
        ("negative_int", -3, 4, 4, "int32"),
        ("int_array", np.array([0, 0]), np.array([10, -7]), 4, "int32"),
        ("bool", 0, 1, 3, "bool"),
    )
    def test_non_float_dtype(self, start, stop, num, dtype):
        self.assertAllEqual(
            ops.linspace(start, stop, num, dtype=dtype),
            np.linspace(start, stop, num, dtype=dtype),
        )

    def test_retstep_single_sample(self):
        _, step = ops.linspace(0.0, 1.0, 1, retstep=True)
        self.assertTrue(np.isnan(step))

    def test_negative_num_raises(self):
        with self.assertRaisesRegex(ValueError, "num=-1"):
            ops.linspace(0.0, 1.0, -1)
