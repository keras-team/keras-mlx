"""Values of mlx kernels that the keras tests never evaluate."""

import mlx.core as mx
import numpy as np
from absl.testing import parameterized

from keras import ops
from keras.src import testing


class MetalKernelsTest(testing.TestCase):
    def test_any_all_one_byte_ints(self):
        x = np.array([[0, 2], [0, 0]], dtype="int8")
        self.assertAllEqual(ops.any(x, axis=1), [True, False])
        self.assertTrue(ops.all(np.array([1, 2], dtype="uint8")))

    def test_bool_sort(self):
        x = np.array([True, False, True, False])
        self.assertAllEqual(ops.sort(x), [False, False, True, True])
        self.assertAllEqual(np.take(x, ops.argsort(x)), np.sort(x))
        indices = ops.argpartition(x, 1)
        self.assertAllEqual(np.take(x, indices)[:2], [False, False])

    def test_bool_unique_with_size(self):
        x = np.array([True, False, True, True])
        values, counts = ops.unique(
            x, return_counts=True, size=3, fill_value=False
        )
        self.assertAllEqual(values, [False, True, False])
        self.assertAllEqual(counts, [1, 3, 0])

    def test_argmax_keepdims_scalar(self):
        self.assertEqual(ops.argmax(np.float32(1.0), keepdims=True).shape, ())

    def test_expm1(self):
        x = np.linspace(-20, 20, 4001, dtype="float32")
        self.assertAllClose(ops.expm1(x), np.expm1(x), rtol=1e-6, atol=0)

    def test_conv_kernel_gradient_of_empty_batch(self):
        inputs = mx.zeros((0, 8, 3))
        grad = mx.grad(lambda k: ops.conv(inputs, k).sum())(mx.ones((2, 3, 4)))
        self.assertAllEqual(grad, np.zeros((2, 3, 4)))

    @parameterized.product(
        normalization=(
            lambda x, w: ops.layer_normalization(x, gamma=w, beta=w),
            lambda x, w: ops.rms_normalization(x, scale=w),
        ),
        shape=((0, 7), (4, 0)),
    )
    def test_normalization_weight_gradient_of_empty_input(
        self, normalization, shape
    ):
        x = mx.zeros(shape)
        weights = mx.ones((shape[-1],))
        gradient = mx.grad(lambda w: mx.sum(normalization(x, w)))(weights)
        self.assertAllEqual(gradient, np.zeros(shape[-1]))
