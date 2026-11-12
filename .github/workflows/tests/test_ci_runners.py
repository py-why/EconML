# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Check temporary runner compatibility without constraining package Python support."""

import re
from pathlib import Path

import pytest
import yaml

CI = yaml.safe_load((Path(__file__).resolve().parents[1] / "ci.yml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("job_name", ["tests", "store-reqs-per-env"])
@pytest.mark.parametrize("os", ["ubuntu-latest", "windows-latest", "macos-latest"])
@pytest.mark.parametrize("python_version", ["3.9", "3.10", "3.12", "3.14"])
def test_python39_linux_runner_only(job_name, os, python_version):
    """Pin only Python 3.9 Linux jobs to Ubuntu 24.04, preserving all other matrix runners.

    Parse the workflow's restricted equality/AND/OR selector and evaluate its captured
    operands for both matrix jobs, including versions used by notebook LKG entries.
    This remains valid when Python 3.9 is removed from the matrices after the release.
    """
    selector = CI["jobs"][job_name]["runs-on"]
    match = re.fullmatch(
        r"\$\{\{ matrix\.os == '([^']+)' && matrix\.python-version == '([^']+)' "
        r"&& '([^']+)' \|\| matrix\.os \}\}",
        selector,
    )
    assert match is not None, f"Unexpected runner selector: {selector}"
    selected_os, selected_version, pinned_runner = match.groups()
    runner = pinned_runner if os == selected_os and python_version == selected_version else os
    expected = "ubuntu-24.04" if os == "ubuntu-latest" and python_version == "3.9" else os
    assert runner == expected


def test_lint_uses_python312_and_runs_workflow_checks():
    """Run lint on Python 3.12 and execute workflow tests without package-test plugin options."""
    steps = CI["jobs"]["lint"]["steps"]
    versions = [
        step["with"]["python-version"]
        for step in steps
        if step.get("uses", "").startswith("actions/setup-python@")
    ]
    assert versions == ["3.12"]
    command = 'pip install pytest pyyaml tomli packaging && pytest .github/workflows/tests/ -q -o addopts=""'
    assert any(
        step.get("run") == command
        for step in steps
    )
