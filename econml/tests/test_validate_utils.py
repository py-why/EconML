import unittest
import warnings
from unittest.mock import Mock

import numpy as np

from econml.validate.drtester import DRTester
from econml.validate.utils import _calculate_dr_outcomes_with_diagnostics, calculate_dr_outcomes


class TestValidateUtils(unittest.TestCase):
    def test_dr_outcome_clipping_configuration_and_warning(self):
        D = np.array([0, 1, 0, 1])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        reg_preds = np.zeros((4, 2))
        prop_preds = np.array(
            [
                [0.005, 0.995],
                [0.995, 0.005],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )

        with self.assertWarnsRegex(UserWarning, r"2 values.*0: 1, 1: 1"):
            actual = calculate_dr_outcomes(D, y, reg_preds, prop_preds)

        y_dr_0 = (D == 0) / np.clip(prop_preds[:, 0], 0.01, np.inf) * y
        y_dr_1 = (D == 1) / np.clip(prop_preds[:, 1], 0.01, np.inf) * y
        np.testing.assert_array_equal(actual[:, 0], y_dr_1 - y_dr_0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            custom = calculate_dr_outcomes(D, y, reg_preds, prop_preds, min_propensity=0.1, warn_on_clip=False)
        custom_y_dr_0 = (D == 0) / np.clip(prop_preds[:, 0], 0.1, np.inf) * y
        custom_y_dr_1 = (D == 1) / np.clip(prop_preds[:, 1], 0.1, np.inf) * y
        np.testing.assert_array_equal(custom[:, 0], custom_y_dr_1 - custom_y_dr_0)

    def test_dr_outcome_default_is_unchanged_without_clipping(self):
        D = np.array([0, 1, 2])
        y = np.array([1.0, 2.0, 3.0])
        reg_preds = np.arange(9, dtype=float).reshape(3, 3) / 10
        prop_preds = np.array(
            [
                [0.5, 0.3, 0.2],
                [0.2, 0.6, 0.2],
                [0.3, 0.2, 0.5],
            ]
        )

        d0_mask = np.where(D == 0, 1, 0)
        y_dr_0 = reg_preds[:, 0] + (d0_mask / np.clip(prop_preds[:, 0], 0.01, np.inf)) * (y - reg_preds[:, 0])
        expected = []
        for k in (1, 2):
            dk_mask = np.where(D == k, 1, 0)
            y_dr_k = reg_preds[:, k] + (dk_mask / np.clip(prop_preds[:, k], 0.01, np.inf)) * (y - reg_preds[:, k])
            expected.append(y_dr_k - y_dr_0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = calculate_dr_outcomes(D, y, reg_preds, prop_preds)
        np.testing.assert_array_equal(actual, np.column_stack(expected))

    def test_unassigned_low_propensities_do_not_warn(self):
        D = np.array([0, 0, 1, 1])
        y = np.ones(4)
        reg_preds = np.zeros((4, 2))
        prop_preds = np.array(
            [
                [0.9, 0.001],
                [0.9, 0.001],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            calculate_dr_outcomes(D, y, reg_preds, prop_preds)

        _, diagnostics = _calculate_dr_outcomes_with_diagnostics(D, y, reg_preds, prop_preds)
        self.assertEqual(diagnostics["below_threshold_by_treatment"], {0: 0, 1: 2})
        self.assertEqual(diagnostics["n_below_threshold"], 2)
        self.assertEqual(diagnostics["clipped_by_treatment"], {0: 0, 1: 0})
        self.assertEqual(diagnostics["n_clipped"], 0)

    def test_dr_outcome_input_validation(self):
        D = np.array([0, 1])
        y = np.array([1.0, 2.0])
        reg_preds = np.zeros((2, 2))
        prop_preds = np.full((2, 2), 0.5)

        for threshold in (0, 0.5, np.nan, np.inf, True, "0.01"):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "min_propensity"):
                calculate_dr_outcomes(D, y, reg_preds, prop_preds, min_propensity=threshold)
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            calculate_dr_outcomes(D, y[:-1], reg_preds, prop_preds)
        with self.assertRaisesRegex(ValueError, "same number of treatment columns"):
            calculate_dr_outcomes(D, y, reg_preds[:, :1], prop_preds)
        with self.assertRaisesRegex(ValueError, "finite values"):
            calculate_dr_outcomes(D, y, reg_preds, np.array([[np.nan, 0.5], [0.5, 0.5]]))
        with self.assertRaisesRegex(ValueError, "column for every treatment label"):
            calculate_dr_outcomes(np.array([0, 2]), y, reg_preds, prop_preds)

    def test_clip_diagnostics_distinguish_train_and_validation(self):
        Dtrain = np.array([0, 1, 2])
        Dval = np.array([0, 1])
        ytrain = np.ones(3)
        yval = np.ones(2)
        train_preds = (
            np.zeros((3, 3)),
            np.array([[0.005, 0.5, 0.5], [0.5, 0.004, 0.5], [0.5, 0.5, 0.003]]),
        )
        validation_preds = (
            np.zeros((2, 3)),
            np.array([[0.002, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        )
        tester = DRTester(model_regression=None, model_propensity=None, cate=None)
        tester.fit_nuisance_cv = Mock(return_value=train_preds)
        tester.fit_nuisance_train = Mock(return_value=validation_preds)

        with self.assertWarnsRegex(UserWarning, r"train \(0: 1, 1: 1, 2: 1\); validation \(0: 1\)"):
            tester.fit_nuisance(
                np.zeros((2, 1)),
                Dval,
                yval,
                np.zeros((3, 1)),
                Dtrain,
                ytrain,
            )

        self.assertEqual(set(tester.clip_diagnostics_), {"train", "validation"})
        self.assertEqual(tester.clip_diagnostics_["train"]["treatment_labels"], (0, 1, 2))
        self.assertEqual(tester.clip_diagnostics_["validation"]["treatment_labels"], (0, 1))
        self.assertNotIn(2, tester.clip_diagnostics_["validation"]["clipped_by_treatment"])

    def test_clip_diagnostics_are_available_when_warning_is_disabled(self):
        Dval = np.array([0, 1])
        tester = DRTester(
            model_regression=None,
            model_propensity=None,
            cate=None,
            min_propensity=0.1,
            warn_on_clip=False,
        )
        tester.fit_nuisance_cv = Mock(
            return_value=(
                np.zeros((2, 2)),
                np.array([[0.05, 0.95], [0.95, 0.05]]),
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tester.fit_nuisance(np.zeros((2, 1)), Dval, np.ones(2))

        self.assertEqual(set(tester.clip_diagnostics_), {"validation"})
        self.assertEqual(tester.clip_diagnostics_["validation"]["n_clipped"], 2)
        self.assertEqual(tester.clip_diagnostics_["validation"]["min_propensity"], 0.1)
