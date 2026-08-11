# Maintaining EconML

Tasks that require maintainer permissions. Day-to-day contributor workflow --
environment setup, running the tests, building the docs -- is in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Release process

We use GitHub Actions to build and publish the package and documentation.  To create a new release, an admin should perform the following steps:

1. Update the version number in `econml/_version.py` and add a mention of the new version in the news section of `README.md` and commit the changes.
2. Manually run the publish_package.yml workflow to build and publish the package to PyPI.
3. Manually run the publish_docs.yml workflow to build and publish the documentation.
4. Under https://github.com/py-why/EconML/releases, create a new release with a corresponding tag, and update the release notes.

This section covers infrastructure that only repository maintainers (not
contributors or end users) need to know about.

## Last-known-good (LKG) branch

CI pins dependency versions for reproducibility using a long-lived orphan
branch named `lkg`. Each matrix cell in the `tests`, `notebooks`,
`build_sdist`, and `create_docs` jobs owns one file on that branch
containing the `pip freeze` from the most recent green nightly run for
that cell.

### How it works

- **Nightly (and `workflow_dispatch` with `use_lkg=false` from `main`)**:
  each cell installs floating versions, runs, then uploads its
  `pip freeze` as a `requirements-*` artifact. The `push-lkg` job
  collects all those artifacts, copies each one onto the `lkg` branch
  under its canonical filename, and commits + pushes a single update.
  A concurrency group serializes pushes; a retry loop handles races
  with parallel runs.
- **PR runs (and `workflow_dispatch` with `use_lkg=true`)**: each cell
  checks out the `lkg` branch into a `lkg-cache/` directory (sparse,
  `continue-on-error: true` for bootstrap) and installs `-r` its own
  per-cell file. If the file is missing, the cell falls back to a
  floating install so a brand-new cell can still bootstrap.

### File naming convention

Files live flat at the root of the `lkg` branch:

| Job                | Filename                                  |
| ------------------ | ----------------------------------------- |
| `tests` cell       | `lkg-tests-<os>-<py>-<kind>.txt`          |
| `notebooks` cell   | `lkg-notebooks-<kind>-<py>.txt`           |
| `build_sdist`      | `lkg-build-ubuntu-latest-3.12.txt`        |
| `create_docs`      | `lkg-docs-ubuntu-latest-3.12.txt`         |

Rename rule for the seed/recovery script: drop the
`-requirements.txt` suffix from each downloaded artifact file and
prepend `lkg-`.

### Push-job trigger gating

`push-lkg` only runs when **all** of these are true:

- The workflow was triggered by `schedule`, **or** by
  `workflow_dispatch` from `main` (with `inputs.ref` empty / `main`)
  with `use_lkg=false`.
- The run was not cancelled.

This excludes `workflow_dispatch` runs from feature branches with
`use_lkg=false` — those are typically used to test how a breaking
dependency upgrade behaves before it has been reviewed, and must not
be allowed to overwrite `main`'s pins.

### One-time setup

1. Create the empty orphan branch:
   ```sh
   git checkout --orphan lkg
   git rm -rf .
   git commit --allow-empty --no-verify -m "Initial empty lkg branch"
   git push -u origin lkg
   ```
2. Add a [repository ruleset](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
   targeting the `lkg` branch with these rules enabled:
   - **Restrict deletions** — keeps the branch from being nuked.
   - **Block force pushes** — keeps history append-only.

   Do **not** enable "Restrict updates": that rule blocks all non-bypass
   pushes, but the default `GITHUB_TOKEN` that the `push-lkg` job uses is
   not exposed as a bypass actor in repository rulesets (only installed
   GitHub Apps appear in the bypass list, and `GITHUB_TOKEN` isn't one).
   Closing that gap requires either a PAT-as-secret or a dedicated
   GitHub App, both of which add operational cost (token rotation or
   App management) disproportionate to the benefit — the failure mode
   of a stray human push to `lkg` is just "the next nightly overwrites
   it", and the rules above already prevent the unrecoverable failures
   (deletion, history rewrite).

   Also do **not** enable "Require signed commits": `GITHUB_TOKEN`
   commits aren't GPG/SSH-signed and would be blocked.
3. (Optional) Seed the branch from a recent green nightly so the first
   PRs that land already get pinned installs — see below. If you skip
   this, the first nightly after merging the redesign will populate
   the branch.

### Seeding / manual recovery

`.github/workflows/seed_lkg_branch.py` is a manual recovery tool for
the rare cases where the auto-populating push job can't do the job
itself — e.g., the initial seed, or rolling the branch back to a
known-good run after a bad nightly.

The recipe (also in the script's module docstring):

```sh
gh run download <NIGHTLY_RUN_ID> --pattern "requirements-*" --dir /tmp/lkg-seed
git worktree add /tmp/lkg-worktree lkg
python .github/workflows/seed_lkg_branch.py /tmp/lkg-seed \
    --branch-worktree /tmp/lkg-worktree
cd /tmp/lkg-worktree
git add -A && git commit -s -m "Seed lkg branch from run <NIGHTLY_RUN_ID>"
git push origin lkg
```

### Adding a new matrix cell

Just add it to `ci.yml` (or `publish-*.yml`). The first nightly that
runs the new cell will upload its freeze artifact, and `push-lkg`
will create the per-cell file automatically. PRs that touch the new
cell before the first nightly run will fall back to a floating
install (the bootstrap path).

### What to do when a cell's pins are wedged

If a cell's pinned versions become un-installable (e.g., a yanked
release), delete just that cell's file from the `lkg` branch via a
direct commit. The next CI run for that cell will fall back to a
floating install, and the following nightly will repopulate the file.
