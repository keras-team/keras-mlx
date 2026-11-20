import numpy as np
from absl.testing import parameterized

import keras
from keras import layers
from keras import ops
from keras.src import testing
from keras.src.ops.function import Function


def _matmul_model():
    inputs = keras.Input((4,))
    dense = layers.Dense(3)
    x = dense(inputs)
    # The transposed kernel is eager, so the graph stores it as a lazy mlx
    # array. It is a copy and does not follow the kernel.
    constant = ops.transpose(dense.kernel)
    outputs = ops.matmul(ops.matmul(x, constant), dense.kernel)
    return keras.Model(inputs, outputs)


def _list_argument_model():
    inputs = keras.Input((4,))
    dense = layers.Dense(3)
    x = dense(inputs)
    constant = ops.sum(dense.kernel, axis=0, keepdims=True)
    return keras.Model(inputs, layers.Add()([x, constant]))


def _nested_model():
    inputs = keras.Input((4,))
    return keras.Model(inputs, _matmul_model()(inputs))


class _FunctionLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(3)
        inputs = keras.Input((4,))
        x = self.dense(inputs)
        outputs = ops.add(x, ops.sum(self.dense.kernel, axis=0))
        self.function = Function(inputs, outputs)

    def call(self, x):
        return self.function(x)


def _function_attribute_model():
    return keras.Sequential([keras.Input((4,)), _FunctionLayer()])


class FunctionalConstantTest(testing.TestCase):
    @parameterized.named_parameters(
        ("matmul", _matmul_model),
        ("list_argument", _list_argument_model),
        ("nested", _nested_model),
        ("function_attribute", _function_attribute_model),
    )
    def test_compiled_steps(self, make_model):
        x = np.ones((8, 4), dtype="float32")
        y = np.zeros((8, 3), dtype="float32")

        # An eager call would evaluate the constant, so each model goes
        # straight to its compiled step.
        model = make_model()
        model.compile(optimizer="sgd", loss="mse")
        predictions = model.predict(x, verbose=0)
        self.assertAllClose(predictions, ops.convert_to_numpy(model(x)))

        model = make_model()
        model.compile(optimizer="sgd", loss="mse")
        loss = model.evaluate(x, y, verbose=0)
        expected = ops.convert_to_numpy(model(x))
        self.assertAllClose(loss, np.mean(np.square(expected)))

        model = make_model()
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, verbose=0)
