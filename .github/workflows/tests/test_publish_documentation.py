# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Check Pages publishing and the build-only reusable workflow contract."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.load(
    (WORKFLOW_DIR / "publish-documentation.yml").read_text(encoding="utf-8"),
    Loader=yaml.BaseLoader,
)
CI = yaml.load((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_publisher_preflight_uses_the_ci_workflow_test_environment():
    """Install the full helper-test dependencies, including upstream's TOML parser."""
    ci_step = next(step for step in CI["jobs"]["lint"]["steps"] if step.get("name") == "Test workflow helpers")
    publisher_step = next(
        step for step in WORKFLOW["jobs"]["check_site"]["steps"]
        if step.get("name") == "Test archive and workflow contracts"
    )
    assert publisher_step["run"] == ci_step["run"]


@pytest.mark.parametrize("trigger", ["workflow_dispatch", "workflow_call"])
def test_inputs_keep_build_only_without_azure_test_environment(trigger):
    """Both entry points support build-only runs without an obsolete test destination."""
    inputs = WORKFLOW["on"][trigger]["inputs"]
    assert inputs["publish"]["type"] == "boolean"
    assert "environment" not in inputs
    caller = CI["jobs"]["docs"]
    assert caller["with"]["publish"] == "false"
    assert set(caller["with"]) <= set(inputs)
    assert caller["uses"] == "./.github/workflows/publish-documentation.yml"


def test_pages_deployment_uses_successful_build_artifact():
    """Deploy an assembled site from an immutable candidate, not an arbitrary rebuild."""
    build = WORKFLOW["jobs"]["create_docs"]
    publish = WORKFLOW["jobs"]["publish-docs"]
    assert build["needs"] == "check_site"
    assert publish["needs"] == "create_docs"
    assert publish["if"] == "${{ github.event_name == 'push' || inputs.publish }}"
    assert not any(
        step.get("uses", "").startswith(("actions/configure-pages@", "actions/deploy-pages@"))
        for step in build["steps"]
    )
    build_steps = build["steps"]
    docs_artifact = next(step for step in build_steps if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert docs_artifact["with"]["if-no-files-found"] == "error"
    assert any("sphinx-build" in step.get("run", "") and "-W" in step["run"] for step in build_steps)
    assert any(
        "-b doctest" in step.get("run", "") and "inputs.run_doctests" in step["if"]
        for step in build_steps
    )
    pages_artifact = next(
        step for step in build_steps if step.get("uses", "").startswith("actions/upload-pages-artifact@")
    )
    assert pages_artifact["with"]["path"] == docs_artifact["with"]["path"]
    assert "if" not in pages_artifact
    steps = publish["steps"]
    configure = next(step for step in steps if step.get("uses", "").startswith("actions/configure-pages@"))
    deploy = next(step for step in steps if step.get("uses", "").startswith("actions/deploy-pages@"))
    download = next(step for step in steps if step.get("uses", "").startswith("actions/download-artifact@"))
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-pages-artifact@"))
    assert download["with"]["artifact-ids"] == "${{ needs.create_docs.outputs.candidate-id }}"
    assert "artifact-id" in build["outputs"]["candidate-id"]
    assert "candidate-id" in build["outputs"]
    assert configure["uses"].startswith("actions/configure-pages@")
    assert deploy["uses"].startswith("actions/deploy-pages@")
    assert upload["with"]["path"] == "site"
    assert upload["with"].get("name", "github-pages") == deploy.get("with", {}).get(
        "artifact_name", "github-pages"
    )
    assert pages_artifact["with"]["name"] != upload["with"].get("name", "github-pages")
    assert publish["environment"]["url"] == "${{ steps." + deploy["id"] + ".outputs.page_url }}"


def test_publishing_preserves_approval_and_serializes_deployments():
    """Retain production approval gates and avoid cancelling an in-progress site deployment."""
    publish = WORKFLOW["jobs"]["publish-docs"]
    assert publish["environment"]["name"] == (
        "${{ (github.event_name == 'push' || inputs.publication == 'prerelease')"
        " && 'docs-prerelease' || 'prod' }}"
    )
    assert publish["concurrency"] == {"group": "github-pages", "cancel-in-progress": "false"}


def test_pages_permissions_are_scoped_and_supported_by_ci_caller():
    """Build read-only; permit only the deployment job to publish with the built-in token."""
    assert WORKFLOW["permissions"] == {"contents": "read"}
    publish = WORKFLOW["jobs"]["publish-docs"]
    assert publish["permissions"] == {"contents": "write", "pages": "write", "id-token": "write"}
    assert CI["jobs"]["docs"]["permissions"] == publish["permissions"]
    assert WORKFLOW["jobs"]["create_docs"]["permissions"] == {"contents": "read"}
    assert all("secrets." not in str(step) for step in publish["steps"])


def test_automatic_prerelease_builds_exact_push_without_arbitrary_ref_or_build_secrets():
    """Pushes build the event SHA; candidate code is never checked out in the publisher."""
    assert WORKFLOW["on"]["push"] == {"branches": ["main"]}
    build = WORKFLOW["jobs"]["create_docs"]
    checkout = next(step for step in build["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"].startswith("${{ github.event_name == 'push' && github.sha ||")
    assert checkout["with"]["persist-credentials"] == "false"
    guard = build["steps"][0]
    assert 'if [[ -z "$SOURCE_REF" ]]' in guard["run"]
    assert 'if [[ -n "$SOURCE_REF" ]]' in guard["run"]
    publisher = WORKFLOW["jobs"]["publish-docs"]
    assert publisher["steps"][0]["with"]["ref"] == "${{ github.workflow_sha }}"
    assert publisher["steps"][0]["with"]["persist-credentials"] == "false"


def test_archive_assembly_persistence_and_deploy_share_lock_and_stale_guard():
    """The full critical section shares one lock; stale main candidates cannot persist/deploy."""
    publish = WORKFLOW["jobs"]["publish-docs"]
    assert publish["concurrency"]["group"] == "github-pages"
    steps = publish["steps"]
    fresh_index = next(i for i, step in enumerate(steps) if step.get("id") == "fresh")
    guard = steps[fresh_index]["run"]
    assert "refs/heads/main" in guard
    assert '"$SOURCE_SHA" != "$latest"' in guard
    assert "current=false" in guard and "GITHUB_STEP_SUMMARY" in guard
    for step in steps[fresh_index + 1:]:
        assert step["if"] == "steps.fresh.outputs.current == 'true'"
    assembly_index = next(i for i, step in enumerate(steps) if "--archive docs-store" in step.get("run", ""))
    persist_index = next(i for i, step in enumerate(steps) if "refs/heads/docs-archive" in step.get("run", "")
                         and "push origin" in step["run"])
    deploy_index = next(i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/deploy-pages@"))
    assert fresh_index < assembly_index < persist_index < deploy_index
    assert '--expected-sha "$SOURCE_SHA"' in steps[assembly_index]["run"]
    assert "rsync -a --delete --exclude=.git site/ docs-store/" in steps[persist_index]["run"]


def test_reruns_do_not_collide_with_previous_attempt_artifacts():
    """Failed-job reruns retain the candidate ID but upload a fresh deployment artifact."""
    for job in WORKFLOW["jobs"].values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith(("actions/upload-artifact@", "actions/upload-pages-artifact@")):
                assert "${{ github.run_attempt }}" in step["with"]["name"]
    publisher = WORKFLOW["jobs"]["publish-docs"]
    download = next(
        step for step in publisher["steps"] if step.get("uses", "").startswith("actions/download-artifact@")
    )
    assert download["with"]["artifact-ids"] == "${{ needs.create_docs.outputs.candidate-id }}"


@pytest.mark.parametrize(
    "mode,source,latest,current",
    [("prerelease", "a" * 40, "a" * 40, "true"),
     ("prerelease", "a" * 40, "b" * 40, "false"),
     ("release", "a" * 40, "b" * 40, "true"),
     ("prerelease", "a" * 40, "", None)],
)
def test_stale_guard_executes_for_current_superseded_and_release_builds(tmp_path, mode, source, latest, current):
    """Execute the actual workflow guard; only superseded prereleases should be skipped."""
    guard = next(step["run"] for step in WORKFLOW["jobs"]["publish-docs"]["steps"] if step.get("id") == "fresh")
    output, summary = tmp_path / "output", tmp_path / "summary"
    output.touch()
    summary.touch()
    result = subprocess.run(
        [
            shutil.which("bash"), "-c",
            'set -euo pipefail; git() { printf "%s\\trefs/heads/main\\n" "$LATEST"; }\n' + guard,
        ],
        env={**os.environ, "MODE": mode, "SOURCE_SHA": source, "LATEST": latest,
             "GITHUB_OUTPUT": output.as_posix(), "GITHUB_STEP_SUMMARY": summary.as_posix()},
        capture_output=True, text=True,
    )
    if current is None:
        assert result.returncode != 0
        assert "Could not resolve" in result.stdout
        assert not output.read_text()
        return
    result.check_returncode()
    assert f"current={current}" in output.read_text()
    assert ("Skipping superseded" in result.stdout) == (current == "false")


@pytest.mark.parametrize(
    "publish,mode,ref,valid",
    [("true", "release", "", False),
     ("true", "release", "v0.17.1", True),
     ("true", "prerelease", "unreviewed-branch", False),
     ("true", "prerelease", "", True),
     ("true", "unknown", "", False),
     ("false", "release", "", True)],
)
def test_publication_request_requires_explicit_release_or_unoverridden_main(publish, mode, ref, valid):
    """Run the actual input guard, including the build-only CI path."""
    guard = WORKFLOW["jobs"]["create_docs"]["steps"][0]["run"]
    result = subprocess.run(
        [shutil.which("bash"), "-c", "set -euo pipefail;\n" + guard],
        env={**os.environ, "PUBLISH": publish, "MODE": mode, "SOURCE_REF": ref},
        capture_output=True, text=True,
    )
    assert (result.returncode == 0) == valid
