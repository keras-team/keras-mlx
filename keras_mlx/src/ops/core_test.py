import multiprocessing

import pytest

from keras.src import ops
from keras.src import testing


def _double(values):
    # A computed array, so evaluating it needs the stream's thread.
    return ops.convert_to_numpy(ops.multiply(ops.convert_to_tensor(values), 2))


def _double_in_child(queue):
    queue.put(_double([1.0, 2.0]).tolist())


@pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="keras only forks its PyDataset pool where fork is the default",
)
class SharedStreamsForkTest(testing.TestCase):
    def test_forked_child_can_evaluate(self):
        # The parent evaluates on the shared streams first, so the child
        # inherits them without the threads that serve them.
        _double([0.0])
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        child = context.Process(target=_double_in_child, args=(queue,))
        child.start()
        child.join(timeout=30)
        hung = child.is_alive()
        if hung:
            child.kill()
        self.assertFalse(hung, "the child blocked on an inherited stream")
        self.assertEqual(queue.get(timeout=5), [2.0, 4.0])
