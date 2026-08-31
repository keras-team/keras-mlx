"""Tests for behaviour the keras suite cannot cover for this backend.

`keras/src/ops/core_test.py::test_custom_gradient` is gated on the backend
being tensorflow, jax or torch, and the two other tests that exercise
`custom_gradient` are gated the same way, so nothing upstream reaches this
code path.
"""

import mlx.core as mx
import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import ops


def _log1pexp():
    @ops.custom_gradient
    def log1pexp(x):
        e = ops.exp(x)

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))

        return ops.log(1 + e), grad

    return log1pexp


def test_gradient_comes_from_the_supplied_function():
    # The naive derivative of log(1 + exp(x)) overflows at 100 and gives nan.
    # The supplied gradient is the point of the decorator, so it must be the
    # one that runs.
    log1pexp = _log1pexp()
    x = mx.array(100.0)

    supplied = mx.grad(log1pexp)(x)
    naive = mx.grad(lambda v: ops.log(1 + ops.exp(v)))(x)

    assert supplied.item() == 1.0
    assert np.isnan(naive.item())


def test_gradient_with_several_arguments():
    @ops.custom_gradient
    def scaled_product(a, b):
        def grad(upstream):
            return ops.multiply(upstream, 10.0), ops.multiply(upstream, 100.0)

        return ops.multiply(a, b), grad

    da, db = mx.grad(scaled_product, argnums=(0, 1))(
        mx.array(2.0), mx.array(3.0)
    )

    assert da.item() == 10.0
    assert db.item() == 100.0


def test_variable_arguments_are_unwrapped():
    log1pexp = _log1pexp()

    class Log1PExpLayer(layers.Layer):
        def __init__(self):
            super().__init__()
            self.v = backend.Variable(5.0, trainable=False)

        def call(self, inputs):
            return log1pexp(self.v) + inputs

    layer = Log1PExpLayer()
    # The layer adds its inputs, so the derivative with respect to them is 1.
    grad = mx.grad(layer)(mx.array(100.0))

    assert grad.item() == 1.0


def test_forward_pass_returns_the_first_output():
    @ops.custom_gradient
    def doubled(x):
        def grad(upstream):
            return ops.multiply(upstream, 3.0)

        return ops.multiply(x, 2.0), grad

    out = doubled(mx.array([1.0, 2.0, 3.0]))

    assert np.allclose(np.array(out), [2.0, 4.0, 6.0])
