from keras.src.backend.common import KerasVariable
from keras_mlx.src.ops.core import convert_to_numpy
from keras_mlx.src.ops.core import convert_to_tensor


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = convert_to_tensor(value, dtype=self._dtype)

    def _direct_assign(self, value):
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    def __mlx_array__(self):
        return self.value

    def __array__(self, dtype=None):
        value = convert_to_numpy(self.value)
        if dtype:
            return value.astype(dtype)
        return value
