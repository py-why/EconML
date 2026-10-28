# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
"""Resolve and record the dependencies used inside an isolated build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

from build import ProjectBuilder
from build.env import DefaultIsolatedEnv

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def build_system(root: Path) -> dict:
    """Read the project's declared build backend and requirements."""
    with (root / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["build-system"]


def fingerprint(root: Path, recipe: Path, distribution: str) -> str:
    """Identify build inputs independently of runtime dependency versions."""
    inputs = {
        "build-system": build_system(root),
        "recipe": recipe.read_text(encoding="utf-8"),
        "distribution": distribution,
        "python": list(sys.version_info[:2]),
        "platform": sysconfig.get_platform(),
        "free-threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "backend-sources": {
            name: (root / name).read_text(encoding="utf-8")
            for name in ("setup.py", "setup.cfg") if (root / name).is_file()
        },
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]


def matches(path: Path, digest: str) -> bool:
    """Check that a recorded build resolution belongs to these inputs."""
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as stream:
        return stream.readline().strip() == f"# build-recipe-sha: {digest}"


def clean_cython_sources(root: Path) -> None:
    """Force each build to generate C with its own constrained Cython."""
    for source in (root / "econml").rglob("*.pyx"):
        source.with_suffix(".c").unlink(missing_ok=True)


def resolve(root: Path, recipe: Path, distribution: str, output: Path) -> None:
    """Lock static, backend-requested, and transitive build requirements."""
    requirements = set(build_system(root)["requires"])
    env = os.environ.copy()
    # A previous wheel's constraints must not influence the next resolution.
    for name in ("UV_BUILD_CONSTRAINT", "PIP_BUILD_CONSTRAINT"):
        env.pop(name, None)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv must be installed before resolving build constraints")

    with tempfile.TemporaryDirectory(prefix="econml-build-") as directory:
        source = Path(directory) / "requirements.in"
        with DefaultIsolatedEnv(installer="uv") as isolated:
            builder = ProjectBuilder.from_isolated_env(isolated, str(root))
            for _ in range(8):
                source.write_text("\n".join(sorted(requirements)) + "\n", encoding="utf-8")
                # Never inherit output-file preferences: a nightly must advance.
                output.unlink(missing_ok=True)
                subprocess.run([
                    uv, "--config-file", str(recipe), "pip", "compile", str(source),
                    "--python", sys.executable, "--output-file", str(output),
                    "--no-header", "--no-annotate", "--quiet",
                ], check=True, env=env)
                subprocess.run([
                    uv, "pip", "sync", "--python", isolated.python_executable,
                    str(output), "--quiet",
                ], check=True, env=env)
                extra = builder.get_requires_for_build(distribution)
                if distribution == "sdist":
                    # The sdist job also installs EconML editable for its
                    # runtime environment, using these same constraints.
                    extra |= builder.get_requires_for_build("editable")
                if extra <= requirements:
                    return
                requirements.update(extra)
    raise RuntimeError("build backend requirements did not converge")


def prepare(root: Path, recipe: Path, distribution: str, cache: Path,
            cell: str, output: Path, use_lkg: bool) -> str:
    """Use a matching build freeze or resolve a candidate to test."""
    digest = fingerprint(root, recipe, distribution)
    recorded = cache / f"lkg-{cell}-build.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    if use_lkg and matches(recorded, digest):
        print(f"{cell}: using build constraints {recorded}")
        shutil.copyfile(recorded, output)
    else:
        prefix = "::notice::" if use_lkg else ""
        print(f"{prefix}{cell}: resolving build dependencies (freeze missing, stale, or disabled)")
        try:
            resolve(root, recipe, distribution, output)
            resolved = output.read_text(encoding="utf-8")
            output.write_text(f"# build-recipe-sha: {digest}\n{resolved}", encoding="utf-8")
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        finally:
            # Backend hooks can themselves cythonize, before the final
            # resolution includes all backend-requested dependencies.
            clean_cython_sources(root)
    clean_cython_sources(root)
    return digest


def wheel_cell(cell: str) -> str:
    """Distinguish wheel interpreters, ABIs, and target architectures."""
    abi = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
    return f"{cell}-cp{sys.version_info.major}{sys.version_info.minor}{abi}-{sysconfig.get_platform()}"


def main(argv: list[str] | None = None) -> None:
    """Prepare constraints before installing or building EconML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cell", required=True)
    parser.add_argument("--distribution", choices=("editable", "wheel", "sdist"), default="editable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-lkg", choices=("true", "false"), default="true")
    parser.add_argument("--wheel-context", action="store_true")
    parser.add_argument("--record-directory", type=Path)
    parser.add_argument("--export-env", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cell = wheel_cell(args.cell) if args.wheel_context else args.cell
    output = args.output.resolve()
    digest = prepare(root, root / "profiles" / "build.toml", args.distribution,
                     root / "lkg-cache", cell, output, args.use_lkg == "true")
    if args.record_directory:
        args.record_directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, args.record_directory / f"{cell}-build-requirements.txt")
    if args.export_env:
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as stream:
            stream.write(f"UV_BUILD_CONSTRAINT={output}\nPIP_BUILD_CONSTRAINT={output}\n")
            stream.write(f"BUILD_RECIPE_SHA={digest}\n")


if __name__ == "__main__":
    main()
