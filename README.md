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
pip install -r requirements-common.txt
cd ..
```

Assuming you have a fork of `keras-mlx`, you will run the following. This also
installs `keras-mlx` locally so that `keras` can find and import the
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
KERAS_BACKEND=mlx pytest keras --ignore=keras/src/applications \
    --ignore=keras/src/wrappers
```

## Formatting the code

The code is formatted and linted with [Ruff](https://docs.astral.sh/ruff/), run
through `pre-commit`. The first time you are setting up the repo, please run
`pre-commit install`. Note that this needs to be done only once at the
beginning.

Now, whenever you run `git commit -m "<message>"`, the code is automatically
formatted and linted. If there's any error, the commit will not go through.
Please fix the error (most of the times, the error is fixed automatically by the
formatter/linter) and re-run the following:

```
git add .
git commit -m "<message>"
```

In case you want to run the above manually on all files, you can do the
following:

```
pre-commit run --all-files
```
