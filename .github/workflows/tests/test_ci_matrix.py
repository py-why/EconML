# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
"""Checks on the CI matrix that are cheap to get wrong and expensive to debug.

Combinations created by a matrix ``include:`` entry inherit nothing from the
kind-based entries, so a variant-profile cell that forgets ``extras`` or ``opts``
silently runs the wrong thing. These tests expand the matrix the way GitHub
Actions does and assert the invariants the job bodies rely on.
"""
import itertools
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"


def expand(matrix):
    """Expand a GitHub Actions matrix (dimensions, then exclude, then include)."""
    dims = {k: v for k, v in matrix.items() if k not in ("include", "exclude")}
    combos = [dict(zip(dims, vals)) for vals in itertools.product(*dims.values())]

    for excluded in matrix.get("exclude", []):
        combos = [c for c in combos
                  if not all(c.get(k) == v for k, v in excluded.items())]

    extra = []
    for inc in matrix.get("include", []):
        overlap = {k: v for k, v in inc.items() if k in dims}
        matched = False
        for combo in combos:
            if overlap and all(combo.get(k) == v for k, v in overlap.items()):
                combo.update({k: v for k, v in inc.items() if k not in dims})
                matched = True
            elif not overlap:
                combo.update(inc)
                matched = True
        if not matched:
            # An include naming a value absent from the dimensions creates a new
            # combination rather than decorating an existing one.
            extra.append(dict(inc))
    return combos + extra


@pytest.fixture(scope="module")
def ci():
    return yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("workflow", [
    "ci.yml", "publish-documentation.yml", "publish-package.yml",
])
def test_workflow_parses(workflow):
    assert yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))


@pytest.mark.parametrize("job,required", [
    ("tests", ("extras", "opts", "profile")),
    ("notebooks", ("extras", "pattern", "profile")),
])
def test_every_cell_is_fully_specified(ci, job, required):
    for cell in expand(ci["jobs"][job]["strategy"]["matrix"]):
        for field in required:
            assert field in cell, f"{job} cell {cell} is missing {field!r}"


def test_every_referenced_profile_has_a_recipe(ci):
    for job in ("tests", "notebooks"):
        for cell in expand(ci["jobs"][job]["strategy"]["matrix"]):
            recipe = ROOT / "profiles" / f"{cell['profile']}.toml"
            assert recipe.is_file(), f"{job}: missing {recipe.relative_to(ROOT)}"


def test_recipes_are_valid_uv_config():
    tomllib = pytest.importorskip("tomllib")
    recipes = sorted((ROOT / "profiles").glob("*.toml"))
    assert recipes, "no profile recipes found"
    allowed = {
        "prerelease", "resolution", "constraint-dependencies",
        "override-dependencies", "exclude-newer", "index-url", "no-binary",
    }
    for recipe in recipes:
        parsed = tomllib.loads(recipe.read_text(encoding="utf-8"))
        # Guard against a typo silently becoming a no-op: an unrecognised key
        # would leave the profile quietly meaningless.
        unexpected = set(parsed) - allowed
        assert not unexpected, f"{recipe.name}: unexpected keys {sorted(unexpected)}"
        assert "prerelease" in parsed, (
            f"{recipe.name}: set prerelease explicitly so the profile does not "
            "inherit uv's default")


def test_only_the_fallback_path_is_annotated():
    # ::notice:: creates a GitHub annotation, surfaced in the run summary. Using a
    # recorded freeze is the expected outcome for ~100 cells per run, so
    # annotating it would bury the case actually worth seeing -- a cell resolving
    # fresh when it should not have -- among a hundred identical entries.
    # Annotate the exception; log the rule.
    for workflow in ("ci.yml", "publish-documentation.yml", "publish-package.yml"):
        text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
        for line in text.splitlines():
            if "::notice::" in line:
                assert "installing from freeze" not in line, (
                    f"{workflow}: the freeze path should be a plain log line, not "
                    f"an annotation: {line.strip()}")


def test_verify_job_depends_on_eval(ci):
    # `Verify CI checks` is the only required status check on main, and a skipped
    # job counts as a pass. If it does not depend on eval, an eval failure makes
    # every downstream job skip and the required check report green with nothing
    # having run.
    assert "eval" in ci["jobs"]["verify"]["needs"]


def test_matrix_jobs_have_explicit_names(ci):
    # A combination created by `include` exposes *every* key in the auto-generated
    # job name, so an all-floor cell would otherwise render as
    #   Run tests (ubuntu-latest, 3.12, other, all-floor, false, -m "cate_api ...", [plt])
    # which is unreadable and, worse, unstable: adding a matrix property would
    # rename the check and silently break any branch protection rule naming it.
    # An explicit name pins the displayed identity to the dimensions we choose.
    for job in ("tests", "notebooks"):
        name = ci["jobs"][job].get("name", "")
        assert "matrix.profile" in name, (
            f"{job}: set an explicit name including matrix.profile so job names "
            "stay readable and stable")
