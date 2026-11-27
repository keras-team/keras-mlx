"""Tests for ops that take windows of a strided view."""

import mlx.core as mx
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from keras import ops
from keras.src import testing


class StridedViewTest(testing.TestCase):
    def test_max_pool(self):
        x = np.arange(2 * 16 * 16 * 4, dtype="float32").reshape((2, 16, 16, 4))
        view = x[..., ::2]
        windows = sliding_window_view(view, (3, 3), axis=(1, 2))
        self.assertAllEqual(
            ops.max_pool(mx.array(x)[..., ::2], 3, 2),
            windows[:, ::2, ::2].max(axis=(-2, -1)),
        )

    def test_extract_sequences(self):
        x = np.arange(64, dtype="float32").reshape((2, 32))
        self.assertAllEqual(
            ops.extract_sequences(mx.array(x)[:, ::2], 4, 3),
            sliding_window_view(x[:, ::2], 4, axis=-1)[:, ::3],
        )
