# Contributing to EconML

Thanks for your interest in contributing! This document covers how to set up a
development environment, find something to work on, run the tests, and build the
documentation.

Tasks that only maintainers perform -- cutting a release, and operating the
dependency-pinning infrastructure -- live in [MAINTAINING.md](MAINTAINING.md).

## Getting set up

You can get started by cloning this repository. We use 
[setuptools](https://setuptools.readthedocs.io/en/latest/index.html) for building and distributing our package.
We rely on some recent features of setuptools, so make sure to upgrade to a recent version with
`pip install setuptools --upgrade`.  Then from your local copy of the repository you can run `pip install -e .` to get started (but depending on what you're doing you might want to install with extras instead, like `pip install -e .[plt]` if you want to use matplotlib integration, or you can use  `pip install -e .[all]` to include all extras).

## Pre-commit hooks

We use the [pre-commit](https://pre-commit.com/) framework to enforce code style and run checks before every commit. To install the pre-commit hooks, make sure you have pre-commit installed (`pip install pre-commit`) and then run `pre-commit install` in the root of the repository. This will install the hooks and run them automatically before every commit. If you want to run the hooks manually, you can run `pre-commit run --all-files`.

## Finding issues to help with

If you're looking to contribute to the project, we have a number of issues tagged with the [`up for grabs`](https://github.com/py-why/EconML/issues?q=is%3Aopen+is%3Aissue+label%3A%22up+for+grabs%22) and [`help wanted`](https://github.com/py-why/EconML/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) labels. "Up for grabs" issues are ones that we think that people without a lot of experience in our codebase may be able to help with, while "Help wanted" issues are valuable improvements to the library that our team currently does not have time to prioritize where we would greatly appreciate community-initiated PRs, but which might be more involved.

## Running the tests

This project uses [pytest](https://docs.pytest.org/) to run tests for continuous integration.  It is also possible to use `pytest` to run tests locally, but this isn't recommended because it will take an extremely long time and some tests are specific to certain environments or scenarios that have additional dependencies.  However, if you'd like to do this anyway, to run all tests locally after installing the package you can use `pip install pytest pytest-xdist pytest-cov coverage[toml]` (as well as `pip install jupyter jupyter-client nbconvert nbformat seaborn xgboost tqdm` for the dependencies to run all of our notebooks as tests) followed by `python -m pytest`.

Because running all tests can be very time-consuming, we recommend running only the relevant subset of tests when developing locally.  The easiest way to do this is to rely on `pytest`'s compatibility with `unittest`, so you can just run `python -m unittest econml.tests.test_module` to run all tests in a given module, or `python -m unittest econml.tests.test_module.TestClass` to run all tests in a given class.  You can also run `python -m unittest econml.tests.test_module.TestClass.test_method` to run a single test method.

Some of our tests exercise plotting code that imports `matplotlib`.  Our CI sets `MPLBACKEND=Agg` so that matplotlib selects a non-interactive backend; if you run these tests locally (particularly on Windows, where matplotlib's default Tk backend can fail to initialize), you may want to do the same, e.g. `$env:MPLBACKEND = "Agg"` in PowerShell or `export MPLBACKEND=Agg` in bash before invoking `pytest`/`unittest`.

## Working with scikit-learn version differences

EconML deliberately supports a wide range of scikit-learn versions (the exact bounds live in the `scikit-learn` dependency constraint in `pyproject.toml`, kept there as the single source of truth). Many users have non-trivial environments where pinning the latest sklearn would conflict with unrelated dependencies, so keeping the supported range broad is a real benefit to the user base. The cost is that EconML wraps a number of sklearn estimators (in `econml.sklearn_extensions`) and uses a handful of sklearn internals, and those code paths sometimes have to branch on sklearn version. Two pieces of internal infrastructure exist to keep that bounded and consistent:

- **`econml/_sklearn_compat.py`** — the single home for sklearn version flags (`SKLEARN_GE_17`, `SKLEARN_GE_18`, ...) and any compatibility shims (e.g. `one_hot_encoder`, `ensure_finite_kwargs`). When you need to branch on a sklearn version, add a flag here rather than re-implementing `parse(sklearn.__version__) >= parse("X.Y")` at the call site. The module docstring contains the canonical recipe for writing a wrapper that handles a renamed/removed constructor argument across versions (including the easy-to-miss step of reassigning **only** the specific deprecated arg the parent silently overwrites, and — if the parent removed the arg entirely — adding a matching `get_params` override so the removed name doesn't leak back into sklearn's internals).

- **`econml/tests/_sklearn_compat_helpers.py`** — two test helpers that wrapper PRs should use:
  - `assert_sklearn_roundtrip(cls, **kwargs)` constructs the estimator and asserts that `get_params` reports back exactly the kwargs the user passed, and that `clone` preserves them. This is the assertion that catches the common bug where a wrapper drops an arg from `super().__init__()` and the parent silently writes a `"deprecated"` sentinel onto `self`.
  - `no_sklearn_future_warnings()` is a context manager that promotes sklearn-originated `FutureWarning`/`DeprecationWarning` to errors, so a wrapper's happy-path fit/predict tests fail loudly when an upstream deprecation starts firing.

The kinds of sklearn changes EconML has had to absorb so far, and the standard way each is handled:

| Kind of change | Example sklearn change | How EconML handles it | Codebase example |
|---|---|---|---|
| Renamed constructor kwarg | `OneHotEncoder(sparse=...)` → `OneHotEncoder(sparse_output=...)` (1.2) | A small wrapper in `_sklearn_compat.py` that splats the right kwarg name based on a version flag | [`one_hot_encoder()` as of `927ac261`](https://github.com/py-why/EconML/blob/927ac261/econml/_sklearn_compat.py) (the 1.2 branch was removed once the floor reached 1.6) |
| Renamed function kwarg | `force_all_finite=` → `ensure_all_finite=` on `check_array`/`check_X_y` (1.6; old name removed in 1.8) | A helper that returns the correct kwargs dict to splat into the call | [`ensure_finite_kwargs()` as of `927ac261`](https://github.com/py-why/EconML/blob/927ac261/econml/_sklearn_compat.py) (the 1.6 branch was removed once the floor reached 1.6) |
| Symbol moved to a different (often private) submodule | `_get_column_indices` moved from `sklearn.utils` to `sklearn.utils._indexing` (1.5); `_print_elapsed_time` moved to `sklearn.utils._user_interface` (1.5) | A single conditional `import` in `_sklearn_compat.py` re-exports the symbol under a stable name; call sites import from there | [`get_column_indices` / `print_elapsed_time` as of `927ac261`](https://github.com/py-why/EconML/blob/927ac261/econml/_sklearn_compat.py) (the 1.5 branches were removed once the floor reached 1.6) |
| Deprecated/removed constructor arg with sentinel default | `n_alphas` on `LassoCV` becomes a `"deprecated"` sentinel in 1.7+ | Branch the wrapper's `super().__init__()` on `SKLEARN_GE_*`. **Do NOT reassign `self.<deprecated_arg>` back to the user's value** — sklearn's parent `fit` may check that attribute against the sentinel and warn if it was overwritten (this was the bug behind [PR #1031's regression](https://github.com/py-why/EconML/pull/1031); fixed narrowly in [#1042](https://github.com/py-why/EconML/pull/1042) then again more comprehensively as part of the sklearn-compat overhaul). Instead, emit your own wrapper-level `FutureWarning` when the user explicitly passes the deprecated arg (nudging them to the modern name). Test with `assert_sklearn_roundtrip(cls, **kwargs)` for the modern name only. | `WeightedLassoCV.__init__` + `_warn_n_alphas_deprecated` in `econml/sklearn_extensions/linear_model.py` |
| Wrapper `__init__` still accepts an arg the parent removed | `n_alphas` fully removed from `LassoCV`/`lasso_path` in 1.9 (but our wrapper still exposes it as a legacy kwarg) | After parent init, use `hasattr`: preserve the parent's deprecation sentinel when present, or restore that same sentinel after the parent removes the attribute so sklearn's `BaseEstimator.get_params()` can inspect our static wrapper signature. Add BOTH a `get_params` and paired `set_params` override: `get_params` drops the removed arg before sklearn internal calls can consume it, while `set_params` translates the legacy name so `GridSearchCV` / `Pipeline` parameter grids keep working. | `WeightedLassoCV.__init__` / `get_params` / `set_params` in `econml/sklearn_extensions/linear_model.py` (pattern originally suggested in PR #1046) |
| Private-helper signature change | `verbose` argument removed from `_fit_and_predict` (1.4); `_preprocess_data` started returning `sqrt_weights` (1.8) | Branch the call site on the version flag with an inline comment citing the upstream change | `econml/sklearn_extensions/model_selection.py` (1.4 branch around `_fit_and_predict`); `econml/sklearn_extensions/linear_model.py` (1.8 branches around `_preprocess_data`) |
| Behavior change with no API rename | `_preprocess_data` started auto-rescaling by `sqrt(sample_weight)` (1.8); wrappers must pass `rescale_with_sw=False` to keep historical behavior | Branch the kwargs passed at the call site on the version flag; inline comment explains the change | `econml/sklearn_extensions/linear_model.py` (the `_preprocess_data` calls) |

If you're adding or modifying a `sklearn_extensions` wrapper, read the recipe in `econml/_sklearn_compat.py`'s module docstring and use both helpers above when adding tests. **When a new sklearn `FutureWarning`/`DeprecationWarning` first surfaces in CI, treat the removal version it names (e.g. "will be removed in 1.11") as a migration deadline** and open a follow-up issue against that version, rather than silencing the warning and forgetting it — that's how #1032's hard break on sklearn 1.9 got past the earlier deprecation cycle.

## Generating the documentation

This project's documentation is generated via [Sphinx](https://www.sphinx-doc.org/en/main/index.html).  Note that we use [graphviz](https://graphviz.org/)'s 
`dot` application to produce some of the images in our documentation, so you should make sure that `dot` is installed and in your path.

To generate a local copy of the documentation from a clone of this repository, just run `python setup.py build_sphinx -W -E -a`, which will build the documentation and place it under the `build/sphinx/html` path. 

The reStructuredText files that make up the documentation are stored in the [docs directory](https://github.com/py-why/EconML/tree/main/doc); module documentation is automatically generated by the Sphinx build process.
