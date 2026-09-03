"""End to end checks that the backend trains, saves and reloads a model.

The keras suite exercises ops and layers in isolation. Nothing there fails if
the backend fits without learning or reloads a model to different weights, so
cover the whole cycle here. Synthetic data only, nothing is downloaded.
"""

import os
import tempfile

import numpy as np
import pytest

import keras

SAMPLES = 256
FEATURES = 8


def _dataset(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(SAMPLES, FEATURES)).astype("float32")
    # A linear target, so a working backend reaches a low loss quickly and a
    # broken one cannot get there by luck.
    weights = rng.normal(size=(FEATURES, 1)).astype("float32")
    y = (x @ weights).astype("float32")
    return x, y


def _model():
    inputs = keras.Input((FEATURES,))
    hidden = keras.layers.Dense(32, activation="relu")(inputs)
    hidden = keras.layers.Dropout(0.1)(hidden)
    hidden = keras.layers.LayerNormalization()(hidden)
    outputs = keras.layers.Dense(1)(hidden)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def test_backend_is_mlx():
    assert keras.backend.backend() == "mlx"


def test_fit_reduces_the_loss():
    x, y = _dataset()
    history = _model().fit(x, y, epochs=20, batch_size=32, verbose=0)

    losses = history.history["loss"]
    # A backend whose gradients or updates are no ops still returns a history,
    # so assert the loss actually moved rather than that fit returned.
    assert losses[-1] < losses[0] / 2


def test_evaluate_and_predict_agree():
    x, y = _dataset()
    model = _model()
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)

    loss = model.evaluate(x, y, verbose=0)[0]
    predictions = model.predict(x, verbose=0)

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()
    assert loss == pytest.approx(np.mean((predictions - y) ** 2), rel=1e-3)


@pytest.mark.parametrize("filename", ["model.keras", "model.weights.h5"])
def test_predictions_survive_a_save_and_load(filename):
    x, y = _dataset()
    model = _model()
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    before = model.predict(x, verbose=0)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, filename)
        if filename.endswith(".keras"):
            model.save(path)
            restored = keras.saving.load_model(path)
        else:
            model.save_weights(path)
            restored = _model()
            restored.load_weights(path)
        after = restored.predict(x, verbose=0)

    np.testing.assert_allclose(before, after, atol=1e-6)
