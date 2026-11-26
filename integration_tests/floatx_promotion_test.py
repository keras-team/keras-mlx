"""Tests that float ops promote int and bool inputs to floatx."""

import numpy as np
from absl.testing import parameterized

import keras
from keras import ops
from keras.src import testing

FLOAT_OPS = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cos",
    "cosh",
    "exp",
    "exp2",
    "expm1",
    "hard_shrink",
    "hard_sigmoid",
    "hard_silu",
    "leaky_relu",
    "log",
    "log10",
    "log1p",
    "log2",
    "log_softmax",
    "mean",
    "norm",
    "reciprocal",
    "selu",
    "sigmoid",
    "sin",
    "sinh",
    "softmax",
    "soft_shrink",
    "softsign",
    "sparse_plus",
    "sparse_sigmoid",
    "sqrt",
    "squareplus",
    "tan",
    "tanh",
]
BINARY_FLOAT_OPS = [
    "arctan2",
    "divide",
    "divide_no_nan",
    "logaddexp",
    "true_divide",
]


class FloatxPromotionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        floatx = keras.config.floatx()
        self.addCleanup(keras.config.set_floatx, floatx)

    @parameterized.product(
        op=FLOAT_OPS,
        floatx=("bfloat16", "float16", "float32"),
        dtype=("bool", "int8", "int32", "uint8"),
    )
    def test_unary(self, op, floatx, dtype):
        keras.config.set_floatx(floatx)
        fn = getattr(ops, op)
        symbolic = fn(keras.KerasTensor((2, 3), dtype=dtype))
        self.assertDType(fn(np.ones((2, 3), dtype)), symbolic.dtype)
        self.assertDType(fn(np.ones((2, 3), dtype)), floatx)

    @parameterized.product(
        op=BINARY_FLOAT_OPS,
        floatx=("bfloat16", "float16", "float32"),
        dtype=("bool", "int8", "int32", "uint8"),
    )
    def test_binary(self, op, floatx, dtype):
        keras.config.set_floatx(floatx)
        fn = getattr(ops, op)
        symbolic = keras.KerasTensor((2, 3), dtype=dtype)
        x = np.ones((2, 3), dtype)
        self.assertDType(fn(x, x), fn(symbolic, symbolic).dtype)
        self.assertDType(fn(x, x), floatx)
        self.assertDType(fn(x, 2), floatx)
        self.assertDType(fn(x, np.ones((2, 3), "float32")), "float32")

    @parameterized.parameters("bfloat16", "float16", "float32")
    def test_cdist(self, floatx):
        keras.config.set_floatx(floatx)
        x = np.ones((2, 3), "int32")
        self.assertDType(ops.cdist(x, x), floatx)

    @parameterized.product(
        op=FLOAT_OPS, dtype=("bfloat16", "float16", "float32")
    )
    def test_unary_keeps_float_dtype(self, op, dtype):
        self.assertDType(getattr(ops, op)(ops.ones((2, 3), dtype)), dtype)
