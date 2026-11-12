# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Keep package metadata and CI interpreters in sync."""

import re
from pathlib import Path

import pytest
import tomli
import yaml
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
PROJECT = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
SUPPORTED = SpecifierSet(PROJECT["requires-python"])


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS.glob("*.yml")), ids=lambda path: path.name)
def test_ci_interpreters_match_package_support(workflow):
    """Require literal setup-python versions and matrix versions to satisfy Requires-Python.

    Check matrix axes and include entries in every workflow, leaving expression-based
    setup-python versions to their corresponding matrix checks.
    """
    for job in yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"].values():
        matrix = job.get("strategy", {}).get("matrix", {})
        for field in ("python-version", "version"):
            for version in matrix.get(field, []):
                assert isinstance(version, str), f"{workflow.name}: quote matrix interpreter {version!r}"
                assert version in SUPPORTED, f"{workflow.name}: unsupported matrix interpreter {version}"
            for cell in matrix.get("include", []):
                if field in cell:
                    version = cell[field]
                    assert isinstance(version, str), f"{workflow.name}: quote include interpreter {version!r}"
                    assert version in SUPPORTED, f"{workflow.name}: unsupported include interpreter {version}"
        for step in job.get("steps", []):
            if str(step.get("uses", "")).startswith("actions/setup-python@"):
                version = str(step.get("with", {}).get("python-version", ""))
                if version and not version.startswith("${{"):
                    assert version in SUPPORTED, f"{workflow.name}: unsupported setup interpreter {version}"


@pytest.mark.parametrize("generic_classifiers", [
    [],
    ["Programming Language :: Python :: 3", "Programming Language :: Python :: 3 :: Only"],
])
def test_test_matrix_covers_advertised_python_versions(generic_classifiers):
    """Match minor-version classifiers to CI, ignoring generic classifiers and retaining the 3.10 floor."""
    ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    versions = set(ci["jobs"]["tests"]["strategy"]["matrix"]["python-version"])
    advertised = {
        classifier.removeprefix("Programming Language :: Python :: ")
        for classifier in PROJECT["classifiers"] + generic_classifiers
        if re.fullmatch(r"Programming Language :: Python :: \d+\.\d+", classifier)
    }
    assert versions == advertised
    assert "3.10" in versions
    assert "3.9" not in SUPPORTED


@pytest.mark.parametrize("field", ["python-version", "version"])
@pytest.mark.parametrize("include", [False, True])
def test_numeric_matrix_versions_require_quotes(tmp_path, field, include):
    """Reject unquoted YAML 3.10 in matrix axes and include entries with an actionable diagnostic."""
    matrix = f"include:\n          - {field}: 3.10" if include else f"{field}: [3.10]"
    workflow = tmp_path / "numeric.yml"
    workflow.write_text(
        f"jobs:\n  tests:\n    strategy:\n      matrix:\n        {matrix}\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match=r"numeric.yml: quote (matrix|include) interpreter 3\.1"):
        test_ci_interpreters_match_package_support(workflow)
