# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the sklearn-compat test helpers themselves.

These are meta-tests: they validate that ``assert_sklearn_roundtrip`` and
``no_sklearn_future_warnings`` correctly identify the failure modes they
exist to catch (and don't false-positive on healthy inputs).
"""

import unittest
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso

from econml.sklearn_extensions.linear_model import (
    DebiasedLasso,
    WeightedLasso,
    WeightedLassoCV,
    WeightedMultiTaskLassoCV,
)
from econml.tests._sklearn_compat_helpers import (
    assert_no_sklearn_future_warnings,
    assert_sklearn_roundtrip,
    no_sklearn_future_warnings,
)


class _GoodWrapper(BaseEstimator):
    """A well-behaved sklearn-like estimator: get_params round-trips perfectly."""

    def __init__(self, a=1, b="hello", c=None):
        self.a = a
        self.b = b
        self.c = c


class _DriftingWrapper(BaseEstimator):
    """A buggy wrapper: simulates the PR-#1031 failure mode.

    The constructor accepts ``a`` but forgets to assign it onto ``self``
    (or assigns a sentinel instead), so ``get_params()`` returns the
    sentinel rather than what the user passed. ``clone()`` itself does not
    raise — the helper is what catches the divergence.
    """

    def __init__(self, a=1, b="hello"):
        self.a = "deprecated"  # bug: should be ``self.a = a``
        self.b = b


class _ArrayParamWrapper(BaseEstimator):
    """Estimator whose params include numpy arrays (exercise array compare path)."""

    def __init__(self, alphas=None):
        self.alphas = alphas


class TestAssertSklearnRoundtrip(unittest.TestCase):

    def test_passes_for_well_behaved_estimator_no_kwargs(self):
        # legacy instance-only form: just checks clone equivalence
        assert_sklearn_roundtrip(_GoodWrapper())

    def test_passes_for_well_behaved_estimator_with_kwargs(self):
        # primary form: pass class + kwargs to also check input-preservation
        assert_sklearn_roundtrip(_GoodWrapper, a=42, b="world", c=[1, 2, 3])

    def test_passes_for_real_sklearn_lasso(self):
        # sanity: vanilla sklearn estimators must always pass
        assert_sklearn_roundtrip(Lasso, alpha=0.5, fit_intercept=False)

    def test_passes_for_econml_weighted_wrappers(self):
        # These currently round-trip correctly with their defaults; the helper
        # should not regress them. (Calling with explicit kwargs would catch
        # PR #1031 for WeightedLassoCV / WeightedMultiTaskLassoCV — that test
        # belongs to a separate fix-PR-1031 commit.)
        for cls in [
            WeightedLasso,
            WeightedLassoCV,
            WeightedMultiTaskLassoCV,
            DebiasedLasso,
        ]:
            with self.subTest(estimator=cls.__name__):
                assert_sklearn_roundtrip(cls())

    def test_detects_sentinel_overwrite_drift(self):
        # The canonical PR-#1031 failure mode: user passes a=99 but the
        # wrapper silently stores "deprecated" instead.
        with self.assertRaises(AssertionError) as ctx:
            assert_sklearn_roundtrip(_DriftingWrapper, a=99)
        msg = str(ctx.exception)
        self.assertIn("_DriftingWrapper", msg)
        self.assertIn("a:", msg)
        self.assertIn("99", msg)
        self.assertIn("deprecated", msg)
        # the error message should mention the fix-up location
        self.assertIn("_sklearn_compat", msg)

    def test_detects_dropped_kwarg(self):
        # if a wrapper accepts a kwarg via **kwargs and doesn't store it,
        # get_params won't expose it at all
        class _Dropper(BaseEstimator):
            def __init__(self, a=1, **kwargs):  # noqa: D401
                self.a = a
                # b is accepted via **kwargs but silently discarded

        with self.assertRaises(AssertionError) as ctx:
            assert_sklearn_roundtrip(_Dropper, a=2, b=3)
        self.assertIn("does not expose", str(ctx.exception))
        self.assertIn("b", str(ctx.exception))

    def test_rejects_non_callable_with_kwargs(self):
        with self.assertRaises(TypeError) as ctx:
            assert_sklearn_roundtrip(_GoodWrapper(), a=1)
        self.assertIn("class or callable", str(ctx.exception))

    def test_rejects_class_without_kwargs(self):
        with self.assertRaises(TypeError) as ctx:
            assert_sklearn_roundtrip(_GoodWrapper)
        self.assertIn("already-built instance", str(ctx.exception))

    def test_compares_numpy_array_params(self):
        # equal arrays pass through both kwarg-form and instance-form
        assert_sklearn_roundtrip(_ArrayParamWrapper, alphas=np.array([0.1, 0.2]))
        assert_sklearn_roundtrip(_ArrayParamWrapper())

    def test_returns_constructed_estimator(self):
        est = assert_sklearn_roundtrip(_GoodWrapper, a=7)
        self.assertIsInstance(est, _GoodWrapper)
        self.assertEqual(est.a, 7)


class TestNoSklearnFutureWarnings(unittest.TestCase):

    def test_promotes_sklearn_future_warning(self):
        with self.assertRaises(FutureWarning), no_sklearn_future_warnings():
            warnings.warn_explicit(
                "fake sklearn deprecation",
                FutureWarning,
                filename="<test>",
                lineno=1,
                module="sklearn.linear_model._something",
            )

    def test_promotes_sklearn_deprecation_warning(self):
        with self.assertRaises(DeprecationWarning), no_sklearn_future_warnings():
            warnings.warn_explicit(
                "fake sklearn deprecation",
                DeprecationWarning,
                filename="<test>",
                lineno=1,
                module="sklearn.utils._something",
            )

    def test_ignores_non_sklearn_future_warning(self):
        # FutureWarnings from outside sklearn should NOT be promoted to errors.
        # The absence of an exception below is the test; we don't shadow the
        # outer filter stack with an inner catch_warnings so we know our
        # promotion rule was actually consulted.
        with no_sklearn_future_warnings():
            warnings.warn_explicit(
                "not sklearn",
                FutureWarning,
                filename="<test>",
                lineno=1,
                module="someotherlib.submodule",
            )

    def test_respects_custom_categories(self):
        # opt out of DeprecationWarning by passing categories=(FutureWarning,)
        with no_sklearn_future_warnings(categories=(FutureWarning,)):
            warnings.warn_explicit(
                "sklearn deprecation should not error here",
                DeprecationWarning,
                filename="<test>",
                lineno=1,
                module="sklearn.foo",
            )

    def test_respects_extra_modules(self):
        with self.assertRaises(FutureWarning), no_sklearn_future_warnings(extra_modules=("imblearn",)):
            warnings.warn_explicit(
                "imblearn deprecation",
                FutureWarning,
                filename="<test>",
                lineno=1,
                module="imblearn.utils",
            )

    def test_does_not_leak_filters(self):
        # After the context manager exits, the global filter state should be restored
        before = list(warnings.filters)
        with no_sklearn_future_warnings():
            pass
        after = list(warnings.filters)
        self.assertEqual(before, after)


class TestAssertNoSklearnFutureWarnings(unittest.TestCase):

    def test_returns_callable_result(self):
        result = assert_no_sklearn_future_warnings(lambda x: x * 2, 21)
        self.assertEqual(result, 42)

    def test_promotes_warning_from_callable(self):
        def _emit():
            warnings.warn_explicit(
                "fake",
                FutureWarning,
                filename="<test>",
                lineno=1,
                module="sklearn.foo",
            )

        with self.assertRaises(FutureWarning):
            assert_no_sklearn_future_warnings(_emit)


if __name__ == "__main__":
    unittest.main()
