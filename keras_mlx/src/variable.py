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

    def __complex__(self):
        # mlx tries complex() before __mlx_array__, making x * variable
        # complex64. Remove once mlx checks __mlx_array__ first.
        raise TypeError(
            "A Keras Variable cannot be converted to a Python complex. "
            "Use `ops.convert_to_numpy(variable)` to read its value."
        )

    def __array__(self, dtype=None):
        value = convert_to_numpy(self.value)
        if dtype:
            return value.astype(dtype)
        return value
