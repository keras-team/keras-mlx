"""Tests for `ops.while_loop` in a compiled step on the MLX backend."""

import os
import warnings

import mlx.core as mx
import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras import ops
from keras.src import testing

pytestmark = pytest.mark.skipif(
    "MLX_DISABLE_COMPILE" in os.environ,
    reason="Nothing is traced with compilation turned off.",
)


class DoubleUntil(keras.layers.Layer):
    def __init__(self, maximum_iterations=None, **kwargs):
        super().__init__(**kwargs)
        self.maximum_iterations = maximum_iterations

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(), initializer="ones")
        self.calls = self.add_weight(
            shape=(), initializer="zeros", trainable=False
        )

    def call(self, x):
        def body(x):
            self.calls.assign_add(1.0)
            return (x * 2.0,)

        return ops.while_loop(
            lambda x: ops.sum(x) < 100.0,
            body,
            (x * self.scale,),
            maximum_iterations=self.maximum_iterations,
        )[0]


def _fit(maximum_iterations, jit_compile):
    keras.utils.set_random_seed(0)
    inputs = keras.Input((4,))
    layer = DoubleUntil(maximum_iterations)
    model = keras.Model(inputs, layer(inputs))
    model.compile(optimizer="sgd", loss="mse", jit_compile=jit_compile)
    x = np.ones((1, 4), "float32")
    outputs = [model.predict(x, verbose=0), model.predict(8 * x, verbose=0)]
    model.fit(x, 33 * x, epochs=2, verbose=0)
    model.evaluate(x, x, verbose=0)
    return model, outputs, [v.numpy() for v in model.variables]


def _root_loop(x):
    # The condition keeps `sqrt` inside its domain.
    return ops.while_loop(
        lambda x: x > 1.0,
        lambda x: ops.sqrt(x - 1.0),
        x,
        maximum_iterations=6,
    )


class RootLoop(keras.layers.Layer):
    """The loop body reads a weight."""

    def build(self, input_shape):
        self.offset = self.add_weight(shape=(), initializer="ones")

    def call(self, x):
        return ops.while_loop(
            lambda x: ops.all(x > self.offset),
            lambda x: ops.sqrt(x - self.offset),
            x,
            maximum_iterations=6,
        )


def _fit_root_loop(jit_compile):
    inputs = keras.Input((1,))
    model = keras.Model(inputs, RootLoop()(inputs))
    model.compile(optimizer="sgd", loss="mse", jit_compile=jit_compile)
    x = np.full((1, 1), 7.0, "float32")
    model.fit(x, 0 * x, epochs=2, verbose=0)
    return model.layers[-1].offset.numpy()


class WhileLoopTest(testing.TestCase):
    def test_gradient_of_weight_read_in_body(self):
        offset = _fit_root_loop(True)
        self.assertNotAllClose(offset, 1.0)
        self.assertAllClose(offset, _fit_root_loop(False))

    @parameterized.named_parameters(
        ("stops_early", 10), ("reaches_the_bound", 3)
    )
    def test_bounded_loop_is_compiled(self, maximum_iterations):
        _, eager_outputs, eager_variables = _fit(maximum_iterations, False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model, outputs, variables = _fit(maximum_iterations, True)
        self.assertTrue(model.jit_compile)
        self.assertAllClose(outputs, eager_outputs)
        # Covers the gradient and the assignments in the body.
        self.assertAllClose(variables, eager_variables)
        self.assertNotAllClose(model.layers[-1].scale, 1.0)

    def test_unbounded_loop_falls_back(self):
        _, eager_outputs, eager_variables = _fit(None, False)
        with self.assertWarnsRegex(UserWarning, "jit_compile=False") as caught:
            model, outputs, variables = _fit(None, True)
        # predict, fit and evaluate share the one warning.
        messages = [str(w.message) for w in caught.warnings]
        self.assertLen([m for m in messages if "jit_compile" in m], 1)
        self.assertFalse(model.jit_compile)
        self.assertAllClose(outputs, eager_outputs)
        self.assertAllClose(variables, eager_variables)

    def test_other_errors_are_raised(self):
        class HostRead(keras.layers.Layer):
            def call(self, x):
                return x * ops.sum(x).item()

        inputs = keras.Input((4,))
        model = keras.Model(inputs, HostRead()(inputs))
        model.compile()
        with self.assertRaisesRegex(ValueError, "eval"):
            model.predict(np.ones((1, 4), "float32"), verbose=0)

    def test_compiled_op(self):
        def loop(x, limit):
            # The loop is traced from its third iteration.
            i, x = ops.while_loop(
                lambda i, x: ops.logical_or(i < 2, ops.sum(x) < limit),
                lambda i, x: [i + 1, x * 2.0],
                (0, x),
                maximum_iterations=8,
            )
            return x

        x = mx.ones((4,))
        for limit in (1.0, 20.0, 100.0, 1e6):
            self.assertAllClose(
                mx.compile(loop)(x, mx.array(limit)), loop(x, mx.array(limit))
            )

    def test_compiled_op_gradient(self):
        for value in (7.0, 5.0, 3.3):
            x = mx.array(value)
            self.assertAllClose(
                mx.compile(mx.grad(_root_loop))(x), mx.grad(_root_loop)(x)
            )

    def test_compiled_op_condition_shape(self):
        def loop(x):
            return ops.while_loop(
                lambda x: ops.reshape(x < 5.0, (1,)),
                lambda x: x + 1.0,
                x,
                maximum_iterations=10,
            )

        self.assertEqual(mx.compile(loop)(mx.array(0.0)).shape, ())

        def vector_cond(x):
            return ops.while_loop(
                lambda x: x < 5.0, lambda x: x + 1.0, x, maximum_iterations=10
            )

        with self.assertRaisesRegex(ValueError, "length-1 arrays"):
            mx.compile(vector_cond)(mx.array([0.0, 3.0]))

    def test_compiled_op_errors(self):
        def unbounded(x):
            return ops.while_loop(lambda x: x < 5.0, lambda x: x + 1.0, x)

        with self.assertRaisesRegex(ValueError, "maximum_iterations"):
            mx.compile(unbounded)(mx.array(0.0))

        def changes_dtype(x):
            return ops.while_loop(
                lambda x: x < 5,
                lambda x: x + 0.5,
                x,
                maximum_iterations=10,
            )

        with self.assertRaisesRegex(ValueError, "same kind as `loop_vars`"):
            mx.compile(changes_dtype)(mx.array(0))
