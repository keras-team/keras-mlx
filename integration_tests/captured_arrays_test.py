"""Tests for layer held arrays, like `adapt()` state, under `mx.compile`."""

import os
from unittest import mock

import numpy as np
import pytest

import keras
from keras.src import testing


class KeepsCallTensor(keras.layers.Layer):
    def call(self, x):
        self.last = x * 2
        return self.last


class CapturedArraysTest(testing.TestCase):
    def _make_model(self):
        x = np.random.RandomState(0).uniform(size=(64, 8)).astype("float32")
        normalization = keras.layers.Normalization()
        normalization.adapt(x)
        model = keras.Sequential([keras.Input((8,)), normalization])
        model.compile(jit_compile=True)
        return model, normalization, x

    def test_retrace_sees_a_later_adapt(self):
        model, normalization, x = self._make_model()
        model.predict_on_batch(x[:8])

        x = x * 5 + 3
        normalization.adapt(x)
        # A new batch size traces the step again.
        self.assertAllClose(
            model.predict_on_batch(x[:16]),
            (x[:16] - x.mean(axis=0)) / np.sqrt(x.var(axis=0)),
            atol=1e-4,
        )

    def test_layer_keeping_a_call_tensor(self):
        x = np.ones((24, 8), dtype="float32")
        y = np.zeros((24, 1), dtype="float32")
        model = keras.Sequential(
            [keras.Input((8,)), KeepsCallTensor(), keras.layers.Dense(1)]
        )
        model.compile(optimizer="sgd", loss="mse", jit_compile=True)
        model.train_on_batch(x[:8], y[:8])
        model.train_on_batch(x[:8], y[:8])
        # A new batch size traces again, and so does a new function.
        model.train_on_batch(x[:16], y[:16])
        self.assertAllClose(model.predict_on_batch(x), model(x))

    @pytest.mark.skipif(
        "MLX_DISABLE_COMPILE" in os.environ,
        reason="The step runs untraced with compilation turned off.",
    )
    def test_arrays_are_evaluated_once_per_trace(self):
        model, _, x = self._make_model()
        with mock.patch.object(
            model,
            "_eval_captured_arrays",
            wraps=model._eval_captured_arrays,
        ) as eval_captured_arrays:
            for _ in range(3):
                model.predict_on_batch(x[:8])
            self.assertEqual(eval_captured_arrays.call_count, 1)
            model.predict_on_batch(x[:16])
            self.assertEqual(eval_captured_arrays.call_count, 2)
