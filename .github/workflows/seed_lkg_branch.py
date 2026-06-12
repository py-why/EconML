# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
"""Seed (or re-seed) the long-lived ``lkg`` branch from a CI run's per-cell freezes.

Walks ``input_dir`` looking for subdirectories named ``requirements-*``, each
containing one ``*-requirements.txt`` freeze file uploaded by a tests / notebooks
/ build_sdist / create_docs job. Copies each file into ``output_dir`` with the
name ``lkg-<stem>.txt`` where ``<stem>`` is the artifact file's basename minus
the ``-requirements.txt`` suffix.

This is the manual / recovery counterpart to the ``push-lkg`` job in
``ci.yml``: nightly runs (and ``workflow_dispatch`` runs from ``main`` with
``use_lkg=false``) populate the ``lkg`` branch automatically, but this script
lets you seed the branch from any chosen run, including:

* Bootstrapping the branch before the first nightly fires (not normally
  needed — the install step falls back to a floating install when its
  per-cell file is missing).
* Recovering from a corrupted ``lkg`` branch by reconstructing it from a
  known-good run's artifacts.
* Importing freezes from a workflow_dispatch run that didn't auto-push
  (e.g. because it was dispatched from a feature branch).

Manual recipe::

    # 1. Pick a known-good run (typically a recent green nightly on main).
    RUN_ID=...

    # 2. Download the per-cell freeze artifacts.
    mkdir /tmp/lkg-seed-input
    gh run download $RUN_ID --pattern "requirements-*" --dir /tmp/lkg-seed-input

    # 3. In a worktree for the lkg branch, run this script:
    git worktree add ../econml-lkg-worktree lkg
    cd ../econml-lkg-worktree
    python /path/to/seed_lkg_branch.py /tmp/lkg-seed-input .

    # 4. Commit and push.
    git add -A
    git commit -s -m "Seed lkg branch from run $RUN_ID"
    git push origin lkg

    # 5. Clean up.
    cd -
    git worktree remove ../econml-lkg-worktree
    rm -rf /tmp/lkg-seed-input
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def rename_artifact_file(artifact_filename: str) -> str:
    """Map a freeze artifact filename to its lkg-branch filename.

    >>> rename_artifact_file("tests-ubuntu-latest-3.12-main-requirements.txt")
    'lkg-tests-ubuntu-latest-3.12-main.txt'
    >>> rename_artifact_file("docs-ubuntu-latest-3.12-requirements.txt")
    'lkg-docs-ubuntu-latest-3.12.txt'
    """
    suffix = "-requirements.txt"
    if not artifact_filename.endswith(suffix):
        raise ValueError(
            f"Expected filename to end with {suffix!r}, got: {artifact_filename!r}"
        )
    stem = artifact_filename[: -len(suffix)]
    return f"lkg-{stem}.txt"


def collect_freezes(input_dir: Path) -> list[tuple[Path, str]]:
    """Find all freeze files under ``input_dir/requirements-*/`` subdirs.

    Returns a list of ``(source_path, destination_filename)`` tuples, sorted
    by destination filename for deterministic output.
    """
    found: list[tuple[Path, str]] = []
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("requirements-"):
            continue
        for path in sorted(subdir.iterdir()):
            if path.is_file() and path.name.endswith("-requirements.txt"):
                found.append((path, rename_artifact_file(path.name)))
    found.sort(key=lambda pair: pair[1])
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy per-cell pip-freeze artifacts from a CI run into a directory "
            "structured for the lkg branch. See module docstring for the full "
            "manual recipe."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help=(
            "Directory containing requirements-* subdirectories, as produced "
            "by `gh run download $RUN_ID --pattern requirements-*`."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Destination directory. Usually a worktree on the lkg branch; "
            "created if it does not exist."
        ),
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(
            f"error: input_dir does not exist or is not a directory: {args.input_dir}",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    freezes = collect_freezes(args.input_dir)
    if not freezes:
        print(
            f"error: no requirements-*/*-requirements.txt files found under "
            f"{args.input_dir}",
            file=sys.stderr,
        )
        return 1

    for src, dest_name in freezes:
        dest = args.output_dir / dest_name
        shutil.copyfile(src, dest)
        print(f"  {src.relative_to(args.input_dir)} -> {dest_name}")
    print(f"Copied {len(freezes)} per-cell LKG files into {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
