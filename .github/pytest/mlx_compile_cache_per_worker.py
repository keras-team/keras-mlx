"""Gives each pytest-xdist worker its own mlx CPU compile cache.

mlx 0.32.3 and earlier share the cache across processes, so a worker can load
a kernel another worker is still writing and fail with "file too short".
Remove this once CI runs an mlx release with ml-explore/mlx#4638.
"""

import os
import tempfile


def pytest_configure(config):
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        return
    # mlx builds the cache path from TMPDIR on every compile.
    tmpdir = os.path.join(tempfile.gettempdir(), f"mlx-{worker}")
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
