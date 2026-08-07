# MLX backend implementation for Keras

## Local development

For development, you need the `keras` repository and the `keras-mlx`
repository checked out locally. That's because the unit tests code is in the
keras repository.

We first check out the main `keras` repository and the `pluggable_backend`
branch.

```
gh repo clone keras-team/keras
cd keras
git checkout pluggable_backend
pip install -r requirements.txt
cd ..
```

Assuming you have a fork of `keras-mlx`, you will run the following. This also
also installs `keras-mlx` locally so that `keras` can find and import the
`keras-mlx` module.

```
gh repo clone <your_github_handle>/keras-mlx
cd keras-mlx
pip install -r requirements.txt
pip install -e .
cd ..
```

Running tests happens from the root of the `keras` repository.

```
cd keras
KERAS_BACKEND=mlx pytest keras --ignore=keras/src/applications
```
