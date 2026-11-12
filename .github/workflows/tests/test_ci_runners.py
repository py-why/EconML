# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Check the lint interpreter and workflow-test configuration."""

from pathlib import Path

import yaml

CI = yaml.safe_load((Path(__file__).resolve().parents[1] / "ci.yml").read_text(encoding="utf-8"))


def test_lint_uses_python312_and_runs_workflow_checks():
    """Run lint on Python 3.12 and execute workflow tests without package-test plugin options."""
    steps = CI["jobs"]["lint"]["steps"]
    versions = [
        step["with"]["python-version"]
        for step in steps
        if step.get("uses", "").startswith("actions/setup-python@")
    ]
    assert versions == ["3.12"]
    command = (
        'python -m pip install pytest pyyaml tomli packaging'
        ' && python -m pytest .github/workflows/tests/ -q -o addopts=""'
    )
    assert any(
        step.get("run") == command
        for step in steps
    )
