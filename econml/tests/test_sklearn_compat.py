# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for production scikit-learn compatibility shims."""

import unittest

from econml._sklearn_compat import (
    SKLEARN_GE_12,
    SKLEARN_GE_16,
    ensure_finite_kwargs,
    one_hot_encoder,
)


class TestSklearnCompat(unittest.TestCase):
    """Test compatibility helpers against the installed sklearn version."""

    def test_ensure_finite_kwargs(self):
        kwargs = ensure_finite_kwargs("allow-nan")
        expected_key = "ensure_all_finite" if SKLEARN_GE_16 else "force_all_finite"
        self.assertEqual(kwargs, {expected_key: "allow-nan"})

    def test_one_hot_encoder(self):
        encoder = one_hot_encoder(sparse=False, handle_unknown="ignore")
        params = encoder.get_params()
        sparse_key = "sparse_output" if SKLEARN_GE_12 else "sparse"
        self.assertFalse(params[sparse_key])
        self.assertEqual(params["handle_unknown"], "ignore")


if __name__ == "__main__":
    unittest.main()
