# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for production scikit-learn compatibility shims."""

import unittest

from econml._sklearn_compat import ensure_finite_kwargs, one_hot_encoder


class TestSklearnCompat(unittest.TestCase):
    """Test compatibility helpers against the installed sklearn version."""

    def test_ensure_finite_kwargs(self):
        # `force_all_finite` was renamed to `ensure_all_finite` in sklearn 1.6,
        # which is our declared floor, so the new spelling is always the right one.
        self.assertEqual(ensure_finite_kwargs("allow-nan"), {"ensure_all_finite": "allow-nan"})

    def test_one_hot_encoder(self):
        # `sparse` was renamed to `sparse_output` in sklearn 1.2, below our floor.
        encoder = one_hot_encoder(sparse=False, handle_unknown="ignore")
        params = encoder.get_params()
        self.assertFalse(params["sparse_output"])
        self.assertEqual(params["handle_unknown"], "ignore")


if __name__ == "__main__":
    unittest.main()
