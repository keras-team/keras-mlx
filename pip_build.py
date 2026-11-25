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


_KERAS_DEP_RE = re.compile(
    r"^keras\s*(\[[^\]]*\])?(?:\s*[<>=!~][^;]*)?(\s*;.*)?$"
)


def _find_dependencies_span(pyproj_str):
    """Return `(start, end)` of the `dependencies = [...]` body."""
    match = re.search(r"^dependencies\s*=\s*\[", pyproj_str, re.MULTILINE)
    if not match:
        raise ValueError("Could not locate `dependencies` in pyproject.toml")
    start = match.end()
    depth = 1
    in_quote = None
    in_comment = False
    i = start
    while i < len(pyproj_str):
        ch = pyproj_str[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
        elif in_quote:
            if ch == "\\" and in_quote == '"':
                i += 1
            elif ch == in_quote:
                in_quote = None
        elif ch == "#":
            in_comment = True
        elif ch in ('"', "'"):
            in_quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return start, i
        i += 1
    raise ValueError("Unterminated `dependencies` in pyproject.toml")


def _rewrite_keras_dependency(body):
    """Replace the `"keras"` dependency inside a TOML list body."""
    parts = []
    last = 0
    in_comment = False
    i = 0
    replacements = 0
    while i < len(body):
        ch = body[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
            i += 1
            continue
        if ch == "#":
            in_comment = True
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            str_start = i + 1
            i += 1
            while i < len(body):
                if body[i] == "\\" and quote == '"':
                    i += 2
                    continue
                if body[i] == quote:
                    break
                i += 1
            if i >= len(body):
                raise ValueError("Unterminated string in `dependencies`")
            dep_str = body[str_start:i]
            dep_match = _KERAS_DEP_RE.match(dep_str)
            if dep_match:
                extras, marker = dep_match.groups()
                new_dep = f"keras-nightly{extras or ''}{marker or ''}"
                parts.append(body[last:str_start])
                parts.append(new_dep)
                last = i
                replacements += 1
            i += 1
            continue
        i += 1
    if replacements != 1:
        raise ValueError("Could not rewrite `keras` dependency for nightly")
    parts.append(body[last:])
    return "".join(parts)


def _rewrite_nightly_pyproject(build_path, pkg_name):
    """Switch package name and `keras` dependency to nightly variants."""
    nightly_name = f"{pkg_name}-nightly"
    pyproj_path = build_path / "pyproject.toml"
    pyproj_str = pyproj_path.read_text()

    pyproj_str, name_subs = re.subn(
        rf'^name\s*=\s*"{re.escape(pkg_name)}"',
        f'name = "{nightly_name}"',
        pyproj_str,
        count=1,
        flags=re.MULTILINE,
    )
    if name_subs != 1:
        raise ValueError(f"Could not rewrite package name for {pkg_name}")

    start, end = _find_dependencies_span(pyproj_str)
    updated_body = _rewrite_keras_dependency(pyproj_str[start:end])
    pyproj_str = pyproj_str[:start] + updated_body + pyproj_str[end:]

    pyproj_path.write_text(pyproj_str)
    return nightly_name


def export_version_string(build_path, version, is_nightly=False, rc_index=None):
    """Export version string and nightly package metadata in `build_path`."""
    if is_nightly and rc_index is not None:
        raise ValueError("Cannot specify both `--nightly` and `--rc`")
    if rc_index is not None and rc_index < 0:
        raise ValueError(f"`--rc` must be non-negative, got {rc_index}")

    dist_pkg_name = package_name
    if is_nightly or rc_index is not None:
        match = re.match(r"^([0-9]+\.[0-9]+\.[0-9]+)", version)
        if not match:
            raise ValueError(f"Unexpected version format: {version}")
        base_version = match.group(1)
        if is_nightly:
            date = datetime.datetime.now(datetime.timezone.utc)
            version = f"{base_version}.dev{date:%Y%m%d%H%M}"
            dist_pkg_name = _rewrite_nightly_pyproject(build_path, package_name)
        else:
            version = f"{base_version}rc{rc_index}"

    version_path = build_path / package / "src" / "version.py"
    version_contents, subs = re.subn(
        r'^__version__\s*=\s*["\'][^"\']*["\']',
        f'__version__ = "{version}"',
        version_path.read_text(),
        count=1,
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


def build(root_path, is_nightly=False, rc_index=None):
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
            build_path, current_version, is_nightly, rc_index
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
    parser.add_argument(
        "--rc",
        type=int,
        help="Specify `[0-9]` when generating RC wheels.",
    )
    args = parser.parse_args()
    root_path = pathlib.Path(__file__).parent.resolve()
    whl_path = build(root_path, args.nightly, args.rc)
    if whl_path and args.install:
        install_whl(whl_path)
