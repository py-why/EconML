# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
"""Tests for ``.github/workflows/seed_lkg_branch.py``.

Run with::

    pytest .github/workflows/tests/

These tests are run from the ``lint`` job in ``ci.yml`` so changes to the
seed script can't regress the artifact-name -> per-cell-file mapping that
``push-lkg`` and the install steps rely on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the sibling script without requiring a package wiring.
_SCRIPT = Path(__file__).resolve().parent.parent / "seed_lkg_branch.py"
_spec = importlib.util.spec_from_file_location("seed_lkg_branch", _SCRIPT)
assert _spec is not None and _spec.loader is not None
seed_lkg_branch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_lkg_branch)


@pytest.mark.parametrize(
    "artifact_name,expected",
    [
        # tests-job cells: ${os}-${py}-${kind}
        ("tests-ubuntu-latest-3.12-main-requirements.txt",
         "lkg-tests-ubuntu-latest-3.12-main.txt"),
        ("tests-ubuntu-latest-3.9-serial-requirements.txt",
         "lkg-tests-ubuntu-latest-3.9-serial.txt"),
        ("tests-windows-latest-3.14-other-requirements.txt",
         "lkg-tests-windows-latest-3.14-other.txt"),
        ("tests-macos-latest-3.13-ray-requirements.txt",
         "lkg-tests-macos-latest-3.13-ray.txt"),
        # notebooks-job cells: ${kind}-${version}
        ("notebooks-except-customer-scenarios-3.12-requirements.txt",
         "lkg-notebooks-except-customer-scenarios-3.12.txt"),
        ("notebooks-customer-scenarios-3.12-requirements.txt",
         "lkg-notebooks-customer-scenarios-3.12.txt"),
        # Single-cell jobs
        ("build-ubuntu-latest-3.12-requirements.txt",
         "lkg-build-ubuntu-latest-3.12.txt"),
        ("docs-ubuntu-latest-3.12-requirements.txt",
         "lkg-docs-ubuntu-latest-3.12.txt"),
    ],
)
def test_rename_artifact_file_known_cells(artifact_name: str, expected: str) -> None:
    assert seed_lkg_branch.rename_artifact_file(artifact_name) == expected


def test_rename_artifact_file_rejects_wrong_suffix() -> None:
    with pytest.raises(ValueError, match="-requirements.txt"):
        seed_lkg_branch.rename_artifact_file("freeze.txt")


def test_collect_freezes_walks_requirements_dirs(tmp_path: Path) -> None:
    # Two real artifact subdirs:
    d1 = tmp_path / "requirements-ubuntu-latest-3.12-main"
    d1.mkdir()
    (d1 / "tests-ubuntu-latest-3.12-main-requirements.txt").write_text("numpy==1.0\n")
    d2 = tmp_path / "requirements-windows-latest-3.9-serial"
    d2.mkdir()
    (d2 / "tests-windows-latest-3.9-serial-requirements.txt").write_text("numpy==2.0\n")
    # Unrelated dirs that must be ignored.
    (tmp_path / "coverage-ubuntu-latest-3.12-main").mkdir()
    (tmp_path / "coverage-ubuntu-latest-3.12-main" / ".coverage.foo").write_text("xx")
    (tmp_path / "tests-ubuntu-latest-3.12-main").mkdir()  # not prefixed with requirements-
    (tmp_path / "stray-file.txt").write_text("ignore me")

    found = seed_lkg_branch.collect_freezes(tmp_path)
    names = [dest for _, dest in found]
    assert names == [
        "lkg-tests-ubuntu-latest-3.12-main.txt",
        "lkg-tests-windows-latest-3.9-serial.txt",
    ]


def test_main_copies_files(tmp_path: Path) -> None:
    src_root = tmp_path / "in"
    out_root = tmp_path / "out"
    cell = src_root / "requirements-ubuntu-latest-3.12-main"
    cell.mkdir(parents=True)
    (cell / "tests-ubuntu-latest-3.12-main-requirements.txt").write_text("numpy==1.0\n")

    rc = seed_lkg_branch.main([str(src_root), str(out_root)])

    assert rc == 0
    assert (out_root / "lkg-tests-ubuntu-latest-3.12-main.txt").read_text() == "numpy==1.0\n"


def test_main_fails_when_input_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src_root = tmp_path / "in"
    src_root.mkdir()

    rc = seed_lkg_branch.main([str(src_root), str(tmp_path / "out")])

    assert rc == 1
    err = capsys.readouterr().err
    assert "no requirements-" in err


def test_main_fails_when_input_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = seed_lkg_branch.main([str(tmp_path / "does-not-exist"), str(tmp_path / "out")])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err
