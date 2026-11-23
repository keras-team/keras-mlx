"""Tests for the 32-bit cap on the paths only the MLX backend has."""

import mlx.core as mx
import numpy as np
from absl.testing import parameterized

import keras
from keras import ops
from keras.src import testing
from keras_mlx.src.ops import core

# (test name, mlx dtype, capped dtype)
MLX_CAPPED = [
    ("int64", mx.int64, "int32"),
    ("uint64", mx.uint64, "uint32"),
    ("float64", mx.float64, "float32"),
]


def _mlx_array(values, dtype):
    # Metal cannot hold float64, so the array lives on the CPU stream.
    with mx.stream(mx.cpu):
        return mx.array(values, dtype=dtype)


class MlxArrayCapTest(testing.TestCase):
    @parameterized.named_parameters(MLX_CAPPED)
    def test_convert_to_tensor(self, dtype, capped):
        x = _mlx_array([1, 2], dtype)
        for out in (ops.convert_to_tensor(x), ops.convert_to_tensor([x, x])):
            self.assertDType(out, capped)
        self.assertAllClose(ops.convert_to_tensor(x), [1, 2])

    @parameterized.named_parameters(MLX_CAPPED)
    def test_ops(self, dtype, capped):
        x = _mlx_array([1, 2], dtype)
        self.assertDType(ops.square(x), capped)
        self.assertDType(ops.squeeze(x), capped)
        self.assertDType(ops.stop_gradient(x), capped)
        self.assertDType(core.reverse_sequence(x), capped)
        self.assertAllClose(core.reverse_sequence(x), [2, 1])

    @parameterized.named_parameters(MLX_CAPPED)
    def test_scalar_operands(self, dtype, capped):
        x = _mlx_array([1, 2], dtype)
        self.assertDType(ops.multiply(x, 2), capped)
        self.assertAllClose(ops.multiply(x, 2), [2, 4])
        self.assertDType(ops.clip(x, 0, 1), capped)
        self.assertAllClose(ops.clip(x, 0, 1), [1, 1])

    @parameterized.named_parameters(MLX_CAPPED)
    def test_variable_assign(self, dtype, capped):
        v = keras.Variable(np.zeros((2,)), dtype=capped)
        v.assign(_mlx_array([1, 2], dtype))
        self.assertDType(v.value, capped)
        self.assertAllClose(v, [1, 2])

    def test_mlx_dtype_argument(self):
        x = ops.ones((2,), dtype="float32")
        self.assertDType(ops.cast(x, mx.int64), "int32")

    def test_uint64_values(self):
        x = _mlx_array([2**32 + 3, 7], mx.uint64)
        self.assertDType(ops.convert_to_tensor(x), "uint32")

    def test_complex128(self):
        x = np.ones((2,), dtype="complex128")
        self.assertDType(ops.convert_to_tensor(x), "complex64")
        self.assertDType(ops.zeros((2,), dtype="complex128"), "complex64")

    def test_complex128_eye_and_arange(self):
        if mx.default_device() == mx.gpu:
            self.skipTest("Metal has no complex64 kernel for eye and arange.")
        self.assertDType(ops.eye(2, dtype="complex128"), "complex64")
        self.assertDType(ops.arange(3, dtype="complex128"), "complex64")


class NumpyArrayCastTest(testing.TestCase):
    def test_float64_to_int_keeps_the_value(self):
        x = np.array([16777217.0], dtype="float64")
        out = ops.convert_to_tensor(x, dtype="int32")
        self.assertEqual(out.item(), 16777217)

    def test_int64_to_float_keeps_the_value(self):
        x = np.array([2**40], dtype="int64")
        out = ops.convert_to_tensor(x, dtype="float32")
        self.assertEqual(out.item(), 2.0**40)

    def test_numpy_scalars_in_a_list(self):
        x = [np.int64(1), np.int64(2)]
        self.assertDType(ops.convert_to_tensor(x), "int32")
        self.assertAllClose(ops.convert_to_tensor(x), [1, 2])


class PythonIntTest(testing.TestCase):
    @parameterized.named_parameters(
        ("scalar", lambda: ops.convert_to_tensor(300, dtype="int8")),
        ("mixed_list", lambda: ops.convert_to_tensor([np.int64(1), 2**40])),
        ("past_int32", lambda: ops.multiply(ops.ones((2,), "int8"), 2**40)),
        ("arange", lambda: ops.arange(2**40, 2**40 + 3)),
    )
    def test_raises(self, fn):
        with self.assertRaises(OverflowError):
            fn()

    def test_wraps_into_the_dtype(self):
        x = ops.convert_to_tensor([1, 2], dtype="int8")
        self.assertAllEqual(ops.multiply(x, 300), [44, 88])
        self.assertAllEqual(ops.full((2,), 2**40, dtype="int32"), [0, 0])

    def test_list_past_int64_to_float(self):
        x = ops.convert_to_tensor([2**70, 1], dtype="float32")
        self.assertAllClose(x, [2.0**70, 1.0])

    def test_compares_by_value(self):
        x = ops.convert_to_tensor([1, 2], dtype="int8")
        self.assertAllEqual(ops.equal(x, 300), [False, False])
        self.assertAllEqual(ops.less(x, 300), [True, True])


class FloatToIntCastTest(testing.TestCase):
    def test_saturates(self):
        x = ops.convert_to_tensor([3e9, -3e9, float("nan"), 300.7])
        self.assertAllEqual(ops.cast(x, "int32"), [2**31 - 1, -(2**31), 0, 300])
        self.assertAllEqual(ops.cast(x, "int64"), ops.cast(x, "int32"))
        self.assertAllEqual(ops.cast(x, "uint8"), [255, 0, 0, 255])
        x = ops.convert_to_tensor([3e9 + 1j])
        self.assertAllEqual(ops.cast(x, "int32"), [2**31 - 1])
