from keras.src.backend.common.name_scope import name_scope
from keras_mlx.src import ops
from keras_mlx.src import random
from keras_mlx.src import rnn
from keras_mlx.src.ops.core import compute_output_spec
from keras_mlx.src.ops.core import device_scope
from keras_mlx.src.variable import Variable

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = True
# Async callbacks would evaluate off the training thread.
IS_THREAD_SAFE = False

distribution_lib = None
