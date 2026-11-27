"""Tests for shapes that mlx gets wrong on Metal."""

import mlx.core as mx
import numpy as np
from absl.testing import parameterized

from keras import ops
from keras.src import testing


class MetalWorkaroundsTest(testing.TestCase):
    def test_depthwise_conv1d_same_padding_even_kernel(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(size=(2, 9, 3)).astype("float32")
        kernel = rng.uniform(size=(2, 3, 1)).astype("float32")
        # "same" only pads the end for a kernel of two.
        padded = np.pad(x, [(0, 0), (0, 1), (0, 0)])
        expected = sum(padded[:, k : k + 9] * kernel[k, :, 0] for k in range(2))
        self.assertAllClose(
            ops.depthwise_conv(x, kernel, padding="same"), expected, atol=1e-5
        )

    @parameterized.named_parameters(("same", "same"), ("valid", "valid"))
    def test_depthwise_conv1d_transpose_input_gradient(self, padding):
        rng = np.random.RandomState(0)
        x = mx.array(rng.uniform(size=(2, 9, 3)).astype("float32"))
        kernel = rng.uniform(size=(2, 3, 1)).astype("float32")
        # One group with a diagonal kernel computes the same thing.
        full_kernel = np.zeros((2, 3, 3), dtype="float32")
        for c in range(3):
            full_kernel[:, c, c] = kernel[:, c, 0]
        weights = mx.array(rng.uniform(size=(1, 20, 1)).astype("float32"))

        def gradient(kernel):
            def loss(x):
                y = ops.conv_transpose(x, kernel, padding=padding)
                return mx.sum(y * weights[:, : y.shape[1]])

            return mx.grad(loss)(x)

        self.assertAllClose(gradient(kernel), gradient(full_kernel), atol=1e-5)
