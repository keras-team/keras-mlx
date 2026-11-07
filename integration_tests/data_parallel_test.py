"""Tests for `keras.distribution` on the MLX backend.

The multi process test runs this file as two workers on the ring backend,
then checks that they end with the weights a single process gets on the
global batches.
"""

import os
import socket
import subprocess
import sys
import tempfile

import numpy as np

import keras
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib

NUM_PROCESSES = 2
PER_PROCESS_BATCH = 8


def _data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 4)).astype("float32")
    y = (x @ rng.standard_normal((4, 1))).astype("float32")
    return x, y


def _model(seed):
    if seed is not None:
        keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.Input((4,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _flat_weights(model):
    return np.concatenate([w.ravel() for w in model.get_weights()])


def _worker(out_dir):
    keras.distribution.initialize()
    keras.distribution.set_distribution(
        keras.distribution.DataParallel(auto_shard_dataset=False)
    )
    rank = backend_dlib.process_id()
    # Each process takes its slice of every global batch.
    x, y = _data()
    global_batch = PER_PROCESS_BATCH * NUM_PROCESSES
    idx = (
        np.arange(0, len(x), global_batch)[:, None]
        + rank * PER_PROCESS_BATCH
        + np.arange(PER_PROCESS_BATCH)
    ).ravel()
    # initialize() synced the seed, so these weights match without a fit.
    synced = _flat_weights(_model(seed=None))
    # A different seed per process, the trainer broadcasts process 0.
    model = _model(seed=rank)
    model.evaluate(x[idx], y[idx], batch_size=PER_PROCESS_BATCH, verbose=0)
    evaluated = _flat_weights(model)
    history = model.fit(
        x[idx],
        y[idx],
        batch_size=PER_PROCESS_BATCH,
        epochs=2,
        shuffle=False,
        verbose=0,
    )
    np.savez(
        os.path.join(out_dir, f"{rank}.npz"),
        synced=synced,
        evaluated=evaluated,
        weights=_flat_weights(model),
        loss=history.history["loss"],
    )


def _free_ports(n):
    sockets = [socket.socket() for _ in range(n)]
    for s in sockets:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports


class DataParallelTest(testing.TestCase):
    def tearDown(self):
        keras.distribution.set_distribution(None)
        super().tearDown()

    def test_initialize_single_process(self):
        keras.distribution.initialize()
        backend_dlib.initialize(num_processes=1, process_id=0)
        self.assertEqual(backend_dlib.num_processes(), 1)
        self.assertEqual(backend_dlib.process_id(), 0)
        self.assertLen(keras.distribution.list_devices(), 1)

    def test_distribute_tensor(self):
        x = np.ones((4, 2), "float32")
        self.assertIs(backend_dlib.distribute_tensor(x, None), x)
        replicated = keras.distribution.TensorLayout([None, None])
        self.assertIs(backend_dlib.distribute_tensor(x, replicated), x)
        sharded = keras.distribution.TensorLayout(["batch", None])
        with self.assertRaises(NotImplementedError):
            backend_dlib.distribute_tensor(x, sharded)

    def test_model_parallel_raises(self):
        mesh = keras.distribution.DeviceMesh(
            (1,), ["model"], keras.distribution.list_devices()
        )
        layout_map = keras.distribution.LayoutMap(mesh)
        keras.distribution.set_distribution(
            keras.distribution.ModelParallel(layout_map=layout_map)
        )
        x, y = _data()
        with self.assertRaises(NotImplementedError):
            _model(seed=0).fit(x, y, verbose=0)

    def test_matches_single_process(self):
        addresses = ",".join(
            f"127.0.0.1:{port}" for port in _free_ports(NUM_PROCESSES)
        )
        with tempfile.TemporaryDirectory() as out_dir:
            workers = []
            logs = []
            try:
                for rank in range(NUM_PROCESSES):
                    env = dict(
                        os.environ,
                        KERAS_DISTRIBUTION_JOB_ADDRESSES=addresses,
                        KERAS_DISTRIBUTION_NUM_PROCESSES=str(NUM_PROCESSES),
                        KERAS_DISTRIBUTION_PROCESS_ID=str(rank),
                        # A script doesn't see keras in the cwd.
                        PYTHONPATH=os.pathsep.join(sys.path),
                    )
                    env.pop("MLX_HOSTFILE", None)
                    # A file, not a pipe, so a full pipe can't block the ring.
                    logs.append(
                        open(os.path.join(out_dir, f"{rank}.log"), "w+")
                    )
                    workers.append(
                        subprocess.Popen(
                            [sys.executable, __file__, out_dir],
                            env=env,
                            stdout=logs[-1],
                            stderr=subprocess.STDOUT,
                        )
                    )
                for worker, log in zip(workers, logs):
                    worker.wait(timeout=300)
                    log.seek(0)
                    self.assertEqual(worker.returncode, 0, log.read())
            finally:
                # A worker left alone waits on the ring until it times out.
                for worker in workers:
                    worker.kill()
                for log in logs:
                    log.close()
            results = [
                np.load(os.path.join(out_dir, f"{rank}.npz"))
                for rank in range(NUM_PROCESSES)
            ]

        x, y = _data()
        model = _model(seed=0)
        history = model.fit(
            x,
            y,
            batch_size=PER_PROCESS_BATCH * NUM_PROCESSES,
            epochs=2,
            shuffle=False,
            verbose=0,
        )
        self.assertAllEqual(results[0]["synced"], results[1]["synced"])
        # evaluate() starts the replicas from process 0's weights.
        self.assertAllEqual(results[0]["evaluated"], results[1]["evaluated"])
        for result in results:
            self.assertAllClose(result["weights"], _flat_weights(model))
            self.assertAllClose(result["loss"], history.history["loss"])
        self.assertAllEqual(results[0]["weights"], results[1]["weights"])


if __name__ == "__main__":
    _worker(sys.argv[1])
