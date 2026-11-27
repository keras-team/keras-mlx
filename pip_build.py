"""Script to create (and optionally install) distributions for Keras MLX.

Usage:

1. Create `.whl` and `.tar.gz` files in `dist/`:

```
python3 pip_build.py
```

2. Also install the new package immediately after:

```
python3 pip_build.py --install
```
"""

import argparse
import ast
import datetime
import pathlib
import re
import shutil
import subprocess
import sys

package = "keras_mlx"
package_name = "keras-mlx"
build_directory = "tmp_build_dir"
dist_directory = "dist"
to_copy = ["pyproject.toml", "README.md", "LICENSE"]


def ignore_files(_, filenames):
    return [
        f
        for f in filenames
        if f == "__pycache__"
        or f.endswith((".pyc", "_test.py", "_test_utils.py", "_test_base.py"))
    ]


def read_version(version_path):
    """Read `__version__` statically from `version.py` without importing."""
    tree = ast.parse(version_path.read_text(), filename=str(version_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, str):
                        return value
    raise ValueError(f"Could not find __version__ in {version_path}")


def _rewrite_nightly_pyproject(build_path, pkg_name):
    """Switch package name and `keras` dependency to nightly variants."""
    nightly_name = f"{pkg_name}-nightly"
    pyproj_path = build_path / "pyproject.toml"
    pyproj_str = pyproj_path.read_text()

    pyproj_str, name_subs = re.subn(
        rf'^name\s*=\s*"{re.escape(pkg_name)}"',
        f'name = "{nightly_name}"',
        pyproj_str,
        flags=re.MULTILINE,
    )
    if name_subs != 1:
        raise ValueError(f"Could not rewrite package name for {pkg_name}")

    pyproj_str, dep_subs = re.subn(
        r'^([ \t]+)(["\'])keras\2,([ \t]*(?:#.*)?)$',
        r"\1\2keras-nightly\2,\3",
        pyproj_str,
        flags=re.MULTILINE,
    )
    if dep_subs != 1:
        raise ValueError("Could not rewrite `keras` dependency for nightly")

    pyproj_path.write_text(pyproj_str)
    return nightly_name


def export_version_string(build_path, version, is_nightly=False):
    """Export version string and nightly package metadata in `build_path`."""
    dist_pkg_name = package_name
    if is_nightly:
        match = re.match(r"^([0-9]+\.[0-9]+\.[0-9]+)", version)
        if not match:
            raise ValueError(f"Unexpected version format: {version}")
        date = datetime.datetime.now(datetime.timezone.utc)
        version = f"{match.group(1)}.dev{date:%Y%m%d%H%M}"
        dist_pkg_name = _rewrite_nightly_pyproject(build_path, package_name)

    version_path = build_path / package / "src" / "version.py"
    version_contents, subs = re.subn(
        r'^__version__\s*=\s*["\'][^"\']*["\']',
        f'__version__ = "{version}"',
        version_path.read_text(),
        flags=re.MULTILINE,
    )
    if subs != 1:
        raise ValueError(f"Could not update `__version__` in {version_path}")
    version_path.write_text(version_contents)
    return dist_pkg_name, version


def copy_source_to_build_directory(root_path, build_path):
    """Copy package sources and build metadata files to `build_path`."""
    shutil.copytree(
        root_path / package, build_path / package, ignore=ignore_files
    )
    for fname in to_copy:
        shutil.copy(root_path / fname, build_path / fname)


def build_and_save_output(build_path, dist_path, dist_pkg_name, version):
    """Run `python -m build` and copy artifacts to `dist_path`."""
    subprocess.run([sys.executable, "-m", "build"], cwd=build_path, check=True)

    dist_path.mkdir(exist_ok=True)
    for fpath in (build_path / dist_directory).glob("*.*"):
        shutil.copy(fpath, dist_path)

    norm_name = dist_pkg_name.replace("-", "_")
    whl_path = (dist_path / f"{norm_name}-{version}-py3-none-any.whl").resolve()
    sdist_path = (dist_path / f"{norm_name}-{version}.tar.gz").resolve()
    if not whl_path.exists() or not sdist_path.exists():
        raise FileNotFoundError(
            f"Expected distributions not found: {whl_path}, {sdist_path}"
        )
    print(f"Build successful. Wheel file available at {whl_path}")
    return whl_path


def build(root_path, is_nightly=False):
    """Stage sources into a temporary build dir and emit wheel + sdist."""
    build_path = root_path / build_directory
    dist_path = root_path / dist_directory
    if build_path.exists():
        raise ValueError(f"Directory already exists: {build_path}")

    build_path.mkdir()
    try:
        copy_source_to_build_directory(root_path, build_path)
        version_file = build_path / package / "src" / "version.py"
        current_version = read_version(version_file)
        dist_pkg_name, version = export_version_string(
            build_path, current_version, is_nightly
        )
        return build_and_save_output(
            build_path, dist_path, dist_pkg_name, version
        )
    finally:
        shutil.rmtree(build_path, ignore_errors=True)


def install_whl(whl_fpath):
    """Force-reinstall the built wheel into the active Python environment."""
    print(f"Installing wheel file: {whl_fpath}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(whl_fpath),
            "--force-reinstall",
            "--no-dependencies",
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install",
        action="store_true",
        help="Whether to install the generated wheel file.",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Whether to generate nightly wheel file.",
    )
    args = parser.parse_args()
    root_path = pathlib.Path(__file__).parent.resolve()
    whl_path = build(root_path, args.nightly)
    if whl_path and args.install:
        install_whl(whl_path)
