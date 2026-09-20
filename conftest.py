import os

# keras star imports the plugin namespace, so keras has to initialise before
# anything reaches keras_mlx. These tests only mean anything on mlx, so set
# the backend rather than defer to the environment.
os.environ["KERAS_BACKEND"] = "mlx"

import keras  # noqa: E402, F401
