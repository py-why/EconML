# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Exercise the archive lifecycle without installing or importing EconML."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs_site.py"
SPEC = importlib.util.spec_from_file_location("docs_site", SCRIPT)
SITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SITE)
SOURCE = "a" * 40
ARCHIVE_COMMIT = "b" * 40


def html_tree(path, version, text="Example"):
    path.mkdir(parents=True)
    (path / "index.html").write_text(f"<html><body>{text}</body></html>", encoding="utf-8")
    (path / "objects.inv").write_bytes(b"Sphinx inventory")
    static = path / "_static"
    static.mkdir()
    (static / "documentation_options.js").write_text(f"const options = {{VERSION: '{version}'}};", encoding="utf-8")
    nested = path / "_autosummary"
    nested.mkdir()
    (nested / "Class.html").write_text("<html><body>API</body></html>", encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    archive = tmp_path / "archive"
    html_tree(archive, "0.16.0", "Existing stable")
    candidate = tmp_path / "candidate"
    html_tree(candidate / "html", "0.17.1", "New build")
    SITE.write_json(candidate / "build.json", {
        "source_sha": SOURCE,
        "version": "0.17.1",
        "python": "3.12",
        "requirements": ["example==1.0"],
        "builder_sha": "c" * 40,
    })
    return archive, candidate


def assemble(inputs, output, mode="prerelease", **kwargs):
    archive, candidate = inputs
    return SITE.assemble(
        archive, candidate, output, mode, SOURCE, bootstrap_commit=ARCHIVE_COMMIT, **kwargs
    )


def test_prerelease_bootstraps_real_version_and_leaves_inputs_unchanged(inputs, tmp_path):
    archive, candidate = inputs
    before = SITE.checksum(archive), SITE.checksum(candidate)
    output = tmp_path / "site"
    manifest = assemble(inputs, output)
    assert manifest["stable"] == "v0.16.0"
    assert [entry["id"] for entry in manifest["versions"]] == ["prerelease", "v0.16.0"]
    assert manifest["versions"][1]["archive_commit"] == ARCHIVE_COMMIT
    assert "Existing stable" in (output / "index.html").read_text()
    assert "New build" in (output / "prerelease" / "index.html").read_text()
    assert before == (SITE.checksum(archive), SITE.checksum(candidate))
    for relative, link in [
        ("index.html", "./_versions/versions.js"),
        ("_autosummary/Class.html", "../_versions/versions.js"),
        ("v0.16.0/index.html", "../_versions/versions.js"),
        ("prerelease/_autosummary/Class.html", "../../_versions/versions.js"),
    ]:
        page = (output / relative).read_text()
        assert link in page
        assert page.count('id="econml-docs-versions"') == 1
    assert "not yet released" in (output / "prerelease" / "index.html").read_text()
    assert SOURCE[:12] in (output / "prerelease" / "index.html").read_text()


def test_release_promotes_root_without_destroying_other_versions(inputs, tmp_path):
    old_site, promoted = tmp_path / "before", tmp_path / "after"
    assemble(inputs, old_site)
    old_release = SITE.checksum(old_site / "v0.16.0")
    prerelease = SITE.checksum(old_site / "prerelease")
    manifest = assemble((old_site, inputs[1]), promoted, "release")
    assert manifest["stable"] == "v0.17.1"
    assert manifest["versions"][1]["build"]["source_sha"] == SOURCE
    assert "New build" in (promoted / "index.html").read_text()
    assert "./_versions/versions.js" in (promoted / "index.html").read_text()
    assert "../../_versions/versions.js" in (promoted / "v0.17.1" / "_autosummary" / "Class.html").read_text()
    assert SITE.checksum(promoted / "v0.16.0") == old_release
    assert SITE.checksum(promoted / "prerelease") == prerelease
    assert "0.17.1 (stable)" in (promoted / "versions.html").read_text()


def test_new_main_replaces_prerelease_but_not_stable_or_archived_pages(inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    original = assemble(inputs, first, "release")
    (inputs[1] / "html" / "_autosummary" / "Class.html").unlink()
    manifest = assemble((first, inputs[1]), second)
    assert not (second / "prerelease" / "_autosummary" / "Class.html").exists()
    assert (second / "v0.17.1" / "_autosummary" / "Class.html").exists()
    assert manifest["root_sha256"] == original["root_sha256"]
    for identifier in ("v0.16.0", "v0.17.1"):
        assert SITE.checksum(first / identifier) == SITE.checksum(second / identifier)


def test_new_release_environment_does_not_rebuild_or_rewrite_previous_html(inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assemble(inputs, first, "release")
    old_html = (first / "v0.17.1" / "index.html").read_bytes()
    info_path = inputs[1] / "build.json"
    info = SITE.read_json(info_path)
    info.update(version="0.18.0", python="3.14", requirements=["example==2.0"])
    SITE.write_json(info_path, info)
    manifest = assemble((first, inputs[1]), second, "release")
    assert manifest["stable"] == "v0.18.0"
    assert (second / "v0.17.1" / "index.html").read_bytes() == old_html
    assert "0.18.0" not in old_html.decode()
    assert "0.18.0" in (second / "versions.json").read_text()
    assert manifest["versions"][0]["build"]["python"] == "3.14"
    assert manifest["versions"][1]["build"]["python"] == "3.12"


def test_release_retries_are_idempotent_but_replacements_are_rejected(inputs, tmp_path):
    first, retry = tmp_path / "first", tmp_path / "retry"
    assemble(inputs, first, "release")
    assemble((first, inputs[1]), retry, "release")
    assert SITE.checksum(first) == SITE.checksum(retry)
    (inputs[1] / "html" / "index.html").write_text("<html><body>Changed</body></html>")
    with pytest.raises(ValueError, match="cannot be overwritten"):
        assemble((first, inputs[1]), tmp_path / "replacement", "release")


@pytest.mark.parametrize("relative", ["index.html", "v0.16.0/index.html", "prerelease/index.html"])
def test_detects_archive_corruption(inputs, tmp_path, relative):
    first = tmp_path / "first"
    assemble(inputs, first)
    (first / relative).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        assemble((first, inputs[1]), tmp_path / "second")


@pytest.mark.parametrize("change", ["path", "root_files", "duplicate", "untracked"])
def test_rejects_malformed_archive_state(inputs, tmp_path, change):
    first = tmp_path / "first"
    manifest = assemble(inputs, first)
    if change == "path":
        manifest["versions"][0]["path"] = "../outside/"
    elif change == "root_files":
        manifest["root_files"] = ["../outside"]
    elif change == "duplicate":
        manifest["versions"].append(manifest["versions"][0])
    else:
        (first / "unexpected.txt").touch()
    SITE.write_json(first / "versions.json", manifest)
    with pytest.raises(ValueError):
        assemble((first, inputs[1]), tmp_path / "second")


@pytest.mark.parametrize("version", ["0.17.1rc1", "0.17.1.dev0", "0.17.1+local", "0.17.1.post1", "0.17"])
def test_only_final_three_part_releases_can_be_promoted(inputs, tmp_path, version):
    info_path = inputs[1] / "build.json"
    info = SITE.read_json(info_path)
    info["version"] = version
    SITE.write_json(info_path, info)
    with pytest.raises(ValueError, match="release version"):
        assemble(inputs, tmp_path / "site", "release")


def test_bootstrap_refuses_to_mislabel_an_unknown_snapshot(inputs, tmp_path):
    (inputs[0] / "_static" / "documentation_options.js").write_text("VERSION: '0.15.0'")
    with pytest.raises(ValueError, match="not the expected"):
        assemble(inputs, tmp_path / "site")


@pytest.mark.parametrize("invalid", ["sha", "inventory", "body", "reserved", "git", "nested-git"])
def test_rejects_invalid_candidate_before_publication(inputs, tmp_path, invalid):
    if invalid == "sha":
        SITE.write_json(inputs[1] / "build.json", {"source_sha": "d" * 40})
    elif invalid == "inventory":
        (inputs[1] / "html" / "objects.inv").unlink()
    elif invalid == "body":
        (inputs[1] / "html" / "index.html").write_text("Not a full page")
    elif invalid == "git":
        (inputs[1] / "html" / ".git").touch()
    elif invalid == "nested-git":
        (inputs[1] / "html" / "_static" / ".git").touch()
    else:
        (inputs[1] / "html" / "versions.json").touch()
    with pytest.raises(ValueError):
        assemble(inputs, tmp_path / "site")


@pytest.mark.parametrize("input_index", [0, 1])
def test_output_cannot_be_nested_under_either_input(inputs, input_index):
    with pytest.raises(ValueError, match="inside an input"):
        assemble(inputs, inputs[input_index] / "output")


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation requires special privileges")
def test_rejects_symlink_before_following_it(inputs, tmp_path):
    (inputs[1] / "html" / "linked").symlink_to(inputs[0], target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        assemble(inputs, tmp_path / "site")


def test_site_size_budget_is_enforced_exactly(inputs, tmp_path, monkeypatch):
    first = tmp_path / "first"
    assemble(inputs, first)
    size = sum(path.stat().st_size for path in SITE.files_in(first))
    monkeypatch.setattr(SITE, "MAX_SITE_BYTES", size)
    assemble(inputs, tmp_path / "at-limit")
    monkeypatch.setattr(SITE, "MAX_SITE_BYTES", size - 1)
    with pytest.raises(ValueError, match="Pages budget"):
        assemble(inputs, tmp_path / "too-big")


def test_navigation_is_idempotent_and_version_list_is_not_baked_into_archived_pages(inputs, tmp_path):
    first = tmp_path / "site"
    assemble(inputs, first)
    tree = first / "v0.16.0"
    before = SITE.checksum(tree)
    SITE.add_navigation(tree, "v0.16.0")
    assert SITE.checksum(tree) == before
    page = (tree / "index.html").read_text()
    assert "0.17.1" not in page
    assert "All documentation versions" in page


def test_shared_navigation_is_replaced_as_a_unit(inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assemble(inputs, first)
    (first / "_versions" / "obsolete.js").write_text("obsolete")
    (first / "_versions" / "versions.js").write_text("broken")
    assemble((first, inputs[1]), second)
    assert not (second / "_versions" / "obsolete.js").exists()
    assert (second / "_versions" / "versions.js").read_bytes() == (SITE.ASSETS / "versions.js").read_bytes()


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation requires special privileges")
def test_shared_archive_symlinks_are_rejected_before_copying(inputs, tmp_path):
    first = tmp_path / "first"
    assemble(inputs, first)
    (first / "_versions" / "linked").symlink_to(inputs[1], target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        assemble((first, inputs[1]), tmp_path / "second")


def test_cli_emits_usable_site_and_provenance(inputs, tmp_path):
    output = tmp_path / "site"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--archive", str(inputs[0]), "--candidate", str(inputs[1]),
            "--output", str(output), "--mode", "prerelease", "--expected-sha", SOURCE,
            "--bootstrap-commit", ARCHIVE_COMMIT,
        ],
        capture_output=True, text=True, check=True,
    )
    assert "Assembled 2 versions" in result.stdout
    manifest = json.loads((output / "versions.json").read_text())
    assert manifest["versions"][0]["build"]["requirements"] == ["example==1.0"]
