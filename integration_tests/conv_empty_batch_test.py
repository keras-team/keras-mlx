"""Tests for convolution gradients on an empty batch on the MLX backend."""

import mlx.core as mx
import numpy as np
from absl.testing import parameterized

import keras
from keras import layers
from keras import ops
from keras.src import testing


class ConvEmptyBatchTest(testing.TestCase):
    @parameterized.named_parameters(
        ("conv1d", ops.conv, (0, 8, 3), (2, 3, 4)),
        ("conv2d", ops.conv, (0, 8, 8, 3), (2, 2, 3, 4)),
        ("conv3d", ops.conv, (0, 4, 4, 4, 3), (2, 2, 2, 3, 4)),
        ("grouped_conv2d", ops.conv, (0, 8, 8, 4), (2, 2, 2, 4)),
        ("grouped_conv3d", ops.conv, (0, 4, 4, 4, 4), (2, 2, 2, 2, 4)),
        ("depthwise_conv", ops.depthwise_conv, (0, 8, 8, 3), (2, 2, 3, 2)),
        ("conv_transpose1d", ops.conv_transpose, (0, 8, 3), (2, 4, 3)),
        ("conv_transpose2d", ops.conv_transpose, (0, 8, 8, 3), (2, 2, 4, 3)),
        (
            "conv_transpose3d",
            ops.conv_transpose,
            (0, 4, 4, 4, 3),
            (2, 2, 2, 4, 3),
        ),
    )
    def test_kernel_gradient_is_zero(self, op, input_shape, kernel_shape):
        inputs = mx.zeros(input_shape)
        for dtype in (mx.float32, mx.float16):
            kernel = mx.ones(kernel_shape, dtype=dtype)
            grad = mx.grad(lambda k: op(inputs, k).sum())(kernel)
            self.assertEqual(grad.dtype, dtype)
            self.assertAllEqual(grad, mx.zeros(kernel_shape))

    @parameterized.named_parameters(
        ("conv", layers.Conv2D(4, 2)),
        ("depthwise_conv", layers.DepthwiseConv2D(2)),
        ("separable_conv", layers.SeparableConv2D(4, 2)),
        ("conv_transpose", layers.Conv2DTranspose(4, 2)),
    )
    def test_train_step_with_no_images(self, conv):
        # A batch whose samples hold no images.
        images = keras.Input((None, 8, 8, 3))
        features = keras.Input((4,))
        x = ops.reshape(images, (-1, 8, 8, 3))
        x = ops.sum(conv(x)) + layers.Dense(1)(features)
        model = keras.Model([images, features], x)
        for jit_compile in (False, True):
            model.compile(optimizer="sgd", loss="mse", jit_compile=jit_compile)
            before = [np.array(w) for w in conv.weights]
            model.train_on_batch(
                [np.zeros((2, 0, 8, 8, 3)), np.ones((2, 4))], np.ones((2, 1))
            )
            for weight, value in zip(conv.weights, before):
                self.assertAllEqual(weight, value)
