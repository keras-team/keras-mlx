import os

# keras star imports the plugin namespace, so keras has to initialise before
# anything reaches keras_mlx. Setting the backend here rather than leaning on
# the environment keeps the suite runnable with a bare `pytest`.
os.environ.setdefault("KERAS_BACKEND", "mlx")

import keras  # noqa: E402, F401
