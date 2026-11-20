# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Assemble a complete Pages site from preserved HTML and one new build.

The archive branch stores the deployed tree, not Python environments. Only new
trees acquire the navigation hook; adding versions never rewrites older trees.
"""

import argparse
import hashlib
import html
import json
import re
import shutil
import tempfile
from pathlib import Path

from packaging.version import Version

MAX_SITE_BYTES = 1_000_000_000
HOOK = re.compile(r"\n<!-- econml-versions:start -->.*?<!-- econml-versions:end -->\n", re.DOTALL)
RELEASE = re.compile(r"v\d+\.\d+\.\d+")
SHA = re.compile(r"[0-9a-f]{40}")
ASSETS = Path(__file__).with_name("docs_assets")


def read_json(path):
    """Read an object, failing explicitly on corrupt or missing metadata."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def files_in(tree, names=None):
    """Enumerate regular files without following symlinks or including Git internals."""
    roots = [tree / name for name in names] if names is not None else list(tree.iterdir())
    files = []

    def visit(path):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in documentation: {path}")
        if path.name == ".git":
            raise ValueError(f"Nested Git metadata is not documentation: {path}")
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in path.iterdir():
                visit(child)
        else:
            raise ValueError(f"Missing or non-regular documentation entry: {path}")

    for root in roots:
        if root.name != ".git":
            visit(root)
    return sorted(files, key=lambda path: path.relative_to(tree).as_posix())


def checksum(tree, names=None):
    """Hash file paths and bytes so removed, added and renamed files are detected."""
    digest = hashlib.sha256()
    for path in files_in(tree, names):
        digest.update(path.relative_to(tree).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def check_html(tree, *, allow_git=False):
    """Require a real landing page and reject unsafe tree entries before copying."""
    if not allow_git and ((tree / ".git").exists() or (tree / ".git").is_symlink()):
        raise ValueError(f"Git metadata is not documentation: {tree}")
    files_in(tree)
    if not (tree / "index.html").is_file():
        raise ValueError(f"Documentation has no index.html: {tree}")
    if not (tree / "objects.inv").is_file():
        raise ValueError(f"Documentation has no Sphinx inventory: {tree}")


def add_navigation(tree, version_id, *, at_root=False, source_sha=None, names=None):
    """Add a theme-independent hook with URLs relative to each page's mount point."""
    for path in files_in(tree, names):
        if path.suffix != ".html":
            continue
        text = HOOK.sub("", path.read_text(encoding="utf-8"))
        body = re.search(r"</body\s*>", text, re.IGNORECASE)
        if body is None:
            raise ValueError(f"Cannot add version navigation to HTML without </body>: {path}")
        depth = len(path.relative_to(tree).parts) - 1 + (0 if at_root else 1)
        base = "../" * depth or "./"
        label = "Prerelease (not yet released)" if version_id == "prerelease" else version_id
        revision = f" &#8212; source {source_sha[:12]}" if source_sha else ""
        hook = (
            '\n<!-- econml-versions:start -->\n'
            f'<link rel="stylesheet" href="{base}_versions/versions.css">\n'
            f'<aside id="econml-docs-versions" aria-label="Documentation versions"'
            f' data-version="{html.escape(version_id, quote=True)}">\n'
            f'<strong>{html.escape(label)}{revision}</strong>\n'
            f'<a href="{base}versions.html">All documentation versions</a>\n'
            '<label for="econml-version-select">Switch version</label>\n'
            '<select id="econml-version-select" hidden></select>\n'
            '<span role="status"></span>\n'
            '</aside>\n'
            f'<script src="{base}_versions/versions.js" defer></script>\n'
            '<!-- econml-versions:end -->\n'
        )
        path.write_text(text[:body.start()] + hook + text[body.start():], encoding="utf-8")


def validate_manifest(archive, manifest):
    """Verify immutable trees and the stable root before any state is replaced."""
    files_in(archive)
    if manifest.get("schema") != 1 or not isinstance(manifest.get("versions"), list):
        raise ValueError("Unsupported or malformed documentation manifest")
    identifiers = []
    for entry in manifest["versions"]:
        identifier = entry["id"]
        if identifier != "prerelease" and RELEASE.fullmatch(identifier) is None:
            raise ValueError(f"Invalid documentation version: {identifier}")
        if entry["path"] != identifier + "/":
            raise ValueError(f"Invalid documentation path: {entry['path']}")
        if identifier in identifiers:
            raise ValueError(f"Duplicate documentation version: {identifier}")
        identifiers.append(identifier)
        check_html(archive / identifier)
        if checksum(archive / identifier) != entry["sha256"]:
            raise ValueError(f"Archived documentation checksum mismatch: {identifier}")
    if manifest.get("stable") not in identifiers or manifest["stable"] == "prerelease":
        raise ValueError("Stable documentation must name an archived release")
    names = manifest.get("root_files")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("Missing or duplicate stable root entries")
    reserved = {*identifiers, ".git", "_versions", "versions.json", "versions.html"}
    for name in names:
        if not isinstance(name, str) or name in reserved or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Invalid stable root entry: {name}")
    if checksum(archive, names) != manifest["root_sha256"]:
        raise ValueError("Stable root checksum mismatch")
    expected = {*names, *identifiers, "_versions", "versions.json", "versions.html", ".git"}
    if any(path.name not in expected for path in archive.iterdir()):
        raise ValueError("Untracked entries in the documentation archive")


def release_id(version):
    parsed = Version(version)
    if parsed.is_prerelease or parsed.is_devrelease or parsed.local or parsed.post is not None:
        raise ValueError(f"Stable documentation requires a final release version: {version}")
    identifier = "v" + str(parsed)
    if RELEASE.fullmatch(identifier) is None:
        raise ValueError(f"Expected a three-part release version: {version}")
    return identifier


def root_names(tree):
    names = [path.name for path in tree.iterdir() if path.name != ".git"]
    if any(
        name in {"_versions", "versions.json", "versions.html", "prerelease"}
        or RELEASE.fullmatch(name)
        for name in names
    ):
        raise ValueError("HTML build uses a reserved versioned-site path")
    return sorted(names)


def copy_entries(source, target, names):
    for name in names:
        src, dst = source / name, target / name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def bootstrap(archive, output, version, commit):
    """Preserve the known existing HTML, without claiming it was rebuilt from source."""
    identifier = release_id(version)
    if SHA.fullmatch(commit) is None:
        raise ValueError("Bootstrap requires the original HTML archive commit")
    check_html(archive, allow_git=True)
    options = (archive / "_static" / "documentation_options.js").read_text(encoding="utf-8")
    if re.search(r"\bVERSION:\s*['\"]" + re.escape(version) + r"['\"]", options) is None:
        raise ValueError(f"Existing site is not the expected {version} snapshot")
    names = root_names(archive)
    copy_entries(archive, output, names)
    add_navigation(output, identifier, at_root=True)
    (output / identifier).mkdir()
    copy_entries(archive, output / identifier, names)
    add_navigation(output / identifier, identifier)
    return {
        "schema": 1,
        "stable": identifier,
        "root_files": names,
        "root_sha256": checksum(output, names),
        "versions": [{
            "id": identifier,
            "label": version,
            "path": identifier + "/",
            "archive_commit": commit,
            "sha256": checksum(output / identifier),
        }],
    }


def render_navigation(output, manifest):
    shared = output / "_versions"
    if shared.exists():
        shutil.rmtree(shared)
    shared.mkdir()
    for name in ("versions.js", "versions.css"):
        shutil.copy2(ASSETS / name, shared / name)
    links = []
    for entry in manifest["versions"]:
        label = entry["label"] + (" (stable)" if entry["id"] == manifest["stable"] else "")
        links.append(f'<li><a href="{entry["path"]}index.html">{html.escape(label)}</a></li>')
    (output / "versions.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>EconML documentation versions</title></head><body>'
        '<h1>EconML documentation versions</h1><p><a href="./">Stable documentation</a></p><ul>'
        + "".join(links) + "</ul></body></html>\n",
        encoding="utf-8",
    )
    write_json(output / "versions.json", manifest)


def assemble(archive, candidate, output, mode, expected_sha, bootstrap_version="0.16.0", bootstrap_commit=""):
    """Create a new site without mutating either the archive or candidate input."""
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    if any(output.resolve().is_relative_to(source.resolve()) for source in (archive, candidate)):
        raise ValueError("Output must not be inside an input directory")
    if mode not in {"release", "prerelease"}:
        raise ValueError(f"Unknown publication mode: {mode}")
    info = read_json(candidate / "build.json")
    if SHA.fullmatch(expected_sha) is None or info["source_sha"] != expected_sha:
        raise ValueError("Candidate source does not match the approved build")
    check_html(candidate / "html")
    output.mkdir(parents=True)
    if (archive / "versions.json").exists():
        manifest = read_json(archive / "versions.json")
        validate_manifest(archive, manifest)
        copy_entries(archive, output, [path.name for path in archive.iterdir() if path.name != ".git"])
    else:
        manifest = bootstrap(archive, output, bootstrap_version, bootstrap_commit)
    identifier = "prerelease" if mode == "prerelease" else release_id(info["version"])
    names = root_names(candidate / "html")
    with tempfile.TemporaryDirectory(prefix="econml-docs-", dir=output.parent) as temp:
        prepared = Path(temp) / "html"
        shutil.copytree(candidate / "html", prepared)
        add_navigation(prepared, identifier, source_sha=expected_sha if mode == "prerelease" else None)
        entry = {
            "id": identifier,
            "label": "Prerelease (main)" if mode == "prerelease" else info["version"],
            "path": identifier + "/",
            "sha256": checksum(prepared),
            "build": info,
        }
        previous = next((item for item in manifest["versions"] if item["id"] == identifier), None)
        if previous and mode == "release" and previous != entry:
            raise ValueError(f"Released documentation cannot be overwritten: {identifier}")
        destination = output / identifier
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(prepared), destination)
    manifest["versions"] = [item for item in manifest["versions"] if item["id"] != identifier] + [entry]
    manifest["versions"].sort(
        key=lambda item: (
            item["id"] == "prerelease",
            Version(item["id"][1:]) if item["id"] != "prerelease" else Version("0"),
        ),
        reverse=True,
    )
    if mode == "release":
        for name in manifest["root_files"]:
            path = output / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        copy_entries(candidate / "html", output, names)
        add_navigation(output, identifier, at_root=True, names=names)
        manifest.update(stable=identifier, root_files=names, root_sha256=checksum(output, names))
    render_navigation(output, manifest)
    validate_manifest(output, manifest)
    size = sum(path.stat().st_size for path in files_in(output))
    if size > MAX_SITE_BYTES:
        raise ValueError(f"Site is {size} bytes, exceeding the {MAX_SITE_BYTES}-byte Pages budget")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("archive", "candidate", "output"):
        parser.add_argument("--" + argument, required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["prerelease", "release"])
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--bootstrap-version", default="0.16.0")
    parser.add_argument("--bootstrap-commit", default="")
    args = parser.parse_args()
    manifest = assemble(**vars(args))
    print(f"Assembled {len(manifest['versions'])} versions; stable is {manifest['stable']}")


if __name__ == "__main__":
    main()
