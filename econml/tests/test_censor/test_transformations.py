"""Tests for censoring unbiased transformations (CUT) and UIF transforms.

Tests verify:
  - Helpers produce correct shapes and values
  - All 12 transforms run without error and return finite (n,) arrays
  - No-censoring sanity check: IPCW with G=1 recovers min(time, tau)
  - Double-robustness: AIPCW ≈ IPCW when G is correct
"""

import unittest
import numpy as np

from econml.censor._transformations import (
    # helpers
    _setup_time_grid,
    _clamp_survival,
    _incremental_hazard,
    _compute_cif_matrix,
    _cumulative_integral,
    _interpolate_to_tau,
    _build_indicators,
    _ipcw_weight_matrix,
    _extract_at_time,
    # RMST
    ipcw_cut_rmst,
    bj_cut_rmst,
    aipcw_cut_rmst,
    uif_diff_rmst,
    # RMTL
    ipcw_cut_rmtlj,
    bj_cut_rmtlj,
    aipcw_cut_rmtlj,
    aipcw_cut_rmtlj_sep_direct_astar1,
    aipcw_cut_rmtlj_sep_indirect_astar1,
    uif_diff_rmtlj,
    uif_diff_rmtlj_sep_direct_astar1,
    uif_diff_rmtlj_sep_indirect_astar1,
)


def _make_survival_data(n=200, seed=42):
    """Generate simple survival test data with known survival matrices."""
    rng = np.random.RandomState(seed)

    # Covariates not needed for transformation tests — only (a, time, event)
    a = rng.binomial(1, 0.5, size=n).astype(float)
    time_latent = rng.exponential(5.0, size=n)
    cens_time = rng.exponential(8.0, size=n)
    admin_cens = 10.0
    obs_time = np.minimum(time_latent, np.minimum(cens_time, admin_cens))
    event = (time_latent <= np.minimum(cens_time, admin_cens)).astype(float)

    # Build a time grid from unique observed times
    s = np.sort(np.unique(obs_time))
    ns = len(s)

    # Synthetic survival matrices: S(t) = exp(-0.15 * t) for all subjects
    lam = 0.15
    S_mat = np.exp(-lam * s[np.newaxis, :]).repeat(n, axis=0)
    # Censoring survival: G(t) = exp(-0.1 * t)
    lam_c = 0.1
    G_mat = np.exp(-lam_c * s[np.newaxis, :]).repeat(n, axis=0)

    return dict(a=a, time=obs_time, event=event, s=s, S=S_mat, G=G_mat,
                n=n, ns=ns, admin_cens=admin_cens, tau=4.0)


def _make_competing_data(n=200, seed=42):
    """Generate competing-risk test data with known survival matrices."""
    rng = np.random.RandomState(seed)

    a = rng.binomial(1, 0.5, size=n).astype(float)
    # Two cause times + censoring
    t_cause1 = rng.exponential(6.0, size=n)
    t_cause2 = rng.exponential(8.0, size=n)
    cens_time = rng.exponential(10.0, size=n)
    admin_cens = 10.0

    t_event = np.minimum(t_cause1, t_cause2)
    obs_time = np.minimum(t_event, np.minimum(cens_time, admin_cens))

    # Event code: 0=censored, 1=cause1, 2=cause2
    event = np.zeros(n, dtype=int)
    not_censored = t_event <= np.minimum(cens_time, admin_cens)
    is_cause1 = (t_cause1 <= t_cause2) & not_censored
    is_cause2 = (t_cause2 < t_cause1) & not_censored
    event[is_cause1] = 1
    event[is_cause2] = 2

    s = np.sort(np.unique(obs_time))
    ns = len(s)

    # Synthetic matrices
    S_mat = np.exp(-0.12 * s[np.newaxis, :]).repeat(n, axis=0)      # overall
    Sj_mat = np.exp(-0.06 * s[np.newaxis, :]).repeat(n, axis=0)     # cause 1
    Sjbar_mat = np.exp(-0.06 * s[np.newaxis, :]).repeat(n, axis=0)  # cause 2
    G_mat = np.exp(-0.08 * s[np.newaxis, :]).repeat(n, axis=0)      # censoring

    return dict(a=a, time=obs_time, event=event, s=s,
                S=S_mat, Sj=Sj_mat, Sjbar=Sjbar_mat, G=G_mat,
                n=n, ns=ns, admin_cens=admin_cens, tau=4.0)


class TestHelpers(unittest.TestCase):

    def test_setup_time_grid_unique(self):
        time = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        s, ds, ind = _setup_time_grid(time, tau=3.5)
        np.testing.assert_array_equal(s, [1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(ds, [1, 1, 1, 1, 1])
        self.assertEqual(ind, 2)  # s[2]=3 < 3.5

    def test_setup_time_grid_freq(self):
        time = np.array([1.0, 2.0])
        s, ds, ind = _setup_time_grid(time, tau=2.5, time_grid=1.0, admin_cens=5.0)
        np.testing.assert_array_almost_equal(s, [1, 2, 3, 4, 5])
        self.assertEqual(ind, 1)  # s[1]=2 < 2.5 < s[2]=3

    def test_clamp_survival(self):
        S = np.array([[0.0001, 0.5, np.nan],
                      [0.9, 0.0, 0.3]])
        result = _clamp_survival(S.copy())
        self.assertTrue(np.all(result >= 1e-3))
        self.assertFalse(np.any(np.isnan(result)))
        # NaN at [0,2] should be forward-filled from [0,1]=0.5
        self.assertAlmostEqual(result[0, 2], 0.5)

    def test_incremental_hazard_shape(self):
        S = np.exp(-0.1 * np.arange(1, 11))[np.newaxis, :].repeat(5, axis=0)
        dH = _incremental_hazard(S)
        self.assertEqual(dH.shape, S.shape)
        # For constant rate, increments should be approximately equal
        np.testing.assert_array_almost_equal(dH[0], 0.1 * np.ones(10), decimal=5)

    def test_compute_cif_matrix(self):
        n, ns = 3, 5
        S = np.exp(-0.1 * np.arange(1, ns + 1))[np.newaxis, :].repeat(n, axis=0)
        Sj = np.exp(-0.05 * np.arange(1, ns + 1))[np.newaxis, :].repeat(n, axis=0)
        dH_j = _incremental_hazard(Sj)
        Fj = _compute_cif_matrix(S, dH_j)
        self.assertEqual(Fj.shape, (n, ns))
        # CIF should be non-decreasing
        self.assertTrue(np.all(np.diff(Fj, axis=1) >= -1e-10))

    def test_cumulative_integral(self):
        vals = np.ones((2, 5))
        ds = np.array([1, 1, 1, 1, 1], dtype=float)
        result = _cumulative_integral(vals, ds)
        np.testing.assert_array_almost_equal(result[0], [1, 2, 3, 4, 5])

    def test_interpolate_to_tau(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        matrix = np.array([[10, 20, 30, 40, 50]], dtype=float)
        result = _interpolate_to_tau(matrix, s, tau=2.5, ind=1)
        self.assertAlmostEqual(result[0], 25.0)

    def test_build_indicators_shape(self):
        time = np.array([1.5, 2.5, 3.5])
        event = np.array([1, 0, 2])
        s = np.array([1.0, 2.0, 3.0, 4.0])
        ind = _build_indicators(time, event, s, cause=1)
        self.assertEqual(ind['Yt'].shape, (3, 4))
        self.assertEqual(ind['dNjt'].shape, (3, 4))
        self.assertEqual(ind['dNjbart'].shape, (3, 4))

    def test_extract_at_time(self):
        s = np.array([1.0, 2.0, 3.0])
        matrix = np.array([[10, 20, 30],
                           [40, 50, 60]], dtype=float)
        time = np.array([2.0, 3.0])
        result = _extract_at_time(matrix, time, s)
        np.testing.assert_array_almost_equal(result, [20, 60])


class TestRMSTTransformations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = _make_survival_data(n=200)

    def test_ipcw_cut_rmst(self):
        d = self.d
        result = ipcw_cut_rmst(d['a'], d['time'], d['event'], d['tau'],
                               d['G'], d['G'], time_grid=d['s'])
        self.assertEqual(result.shape, (d['n'],))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_bj_cut_rmst(self):
        d = self.d
        result = bj_cut_rmst(d['a'], d['time'], d['event'], d['tau'],
                             d['S'], d['S'], time_grid=d['s'])
        self.assertEqual(result.shape, (d['n'],))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_aipcw_cut_rmst(self):
        d = self.d
        result = aipcw_cut_rmst(d['a'], d['time'], d['event'], d['tau'],
                                d['G'], d['G'], d['S'], d['S'],
                                time_grid=d['s'])
        self.assertEqual(result.shape, (d['n'],))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_uif_diff_rmst(self):
        d = self.d
        n = d['n']
        bw = np.ones(n)
        tilt = np.ones(n)
        result = uif_diff_rmst(d['a'], d['time'], d['event'], d['tau'],
                               bw, tilt, d['G'], d['G'], d['S'], d['S'],
                               time_grid=d['s'])
        self.assertEqual(result.shape, (n,))
        self.assertTrue(np.all(np.isfinite(result)))


class TestRMTLTransformations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = _make_competing_data(n=200)

    def test_ipcw_cut_rmtlj(self):
        d = self.d
        result = ipcw_cut_rmtlj(d['a'], d['time'], d['event'], d['tau'],
                                d['G'], d['G'], cause=1, time_grid=d['s'])
        self.assertEqual(result.shape, (d['n'],))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_bj_cut_rmtlj(self):
        d = self.d
        result = bj_cut_rmtlj(d['a'], d['time'], d['event'], d['tau'],
                               d['S'], d['S'], d['Sj'], d['Sj'],
                               cause=1, time_grid=d['s'])
        self.assertEqual(result.shape, (d['n'],))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_aipcw_cut_rmtlj(self):
        d = self.d
        result = aipcw_cut_rmtlj(d['a'], d['time'], d['event'], d['tau'],
                                  d['G'], d['G'], d['S'], d['S'],
                                  d['Sj'], d['Sj'],
                                  cause=1, time_grid=d['s'])
        self.assertEqual(result.shape, (d['n'],))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_aipcw_sep_direct(self):
        d = self.d
        result = aipcw_cut_rmtlj_sep_direct_astar1(
            d['a'], d['time'], d['event'], d['tau'],
            d['G'], d['G'], d['S'], d['S'],
            d['Sj'], d['Sj'], d['Sjbar'], d['Sjbar'],
            cause=1, time_grid=d['s'])
        self.assertEqual(result.shape, (d['n'],))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_aipcw_sep_indirect(self):
        d = self.d
        result = aipcw_cut_rmtlj_sep_indirect_astar1(
            d['a'], d['time'], d['event'], d['tau'],
            d['G'], d['G'], d['S'], d['S'],
            d['Sj'], d['Sj'], d['Sjbar'], d['Sjbar'],
            cause=1, time_grid=d['s'])
        self.assertEqual(result.shape, (d['n'],))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_uif_diff_rmtlj(self):
        d = self.d
        n = d['n']
        bw = np.ones(n)
        tilt = np.ones(n)
        result = uif_diff_rmtlj(d['a'], d['time'], d['event'], d['tau'],
                                bw, tilt,
                                d['G'], d['G'], d['S'], d['S'],
                                d['Sj'], d['Sj'],
                                cause=1, time_grid=d['s'])
        self.assertEqual(result.shape, (n,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_uif_sep_direct(self):
        d = self.d
        n = d['n']
        bw = np.ones(n)
        tilt = np.ones(n)
        ps = 0.5 * np.ones(n)
        result = uif_diff_rmtlj_sep_direct_astar1(
            d['a'], ps, d['time'], d['event'], d['tau'],
            bw, tilt,
            d['G'], d['G'], d['S'], d['S'],
            d['Sj'], d['Sj'], d['Sjbar'], d['Sjbar'],
            cause=1, time_grid=d['s'])
        self.assertEqual(result.shape, (n,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_uif_sep_indirect(self):
        d = self.d
        n = d['n']
        bw = np.ones(n)
        tilt = np.ones(n)
        ps = 0.5 * np.ones(n)
        result = uif_diff_rmtlj_sep_indirect_astar1(
            d['a'], ps, d['time'], d['event'], d['tau'],
            bw, tilt,
            d['G'], d['G'], d['S'], d['S'],
            d['Sj'], d['Sj'], d['Sjbar'], d['Sjbar'],
            cause=1, time_grid=d['s'])
        self.assertEqual(result.shape, (n,))
        self.assertTrue(np.all(np.isfinite(result)))


class TestNoCensoring(unittest.TestCase):
    """When G(t) = 1 for all t, IPCW-RMST should return min(time, tau)."""

    def test_ipcw_rmst_no_censoring(self):
        n = 100
        rng = np.random.RandomState(123)
        time = rng.exponential(3.0, size=n)
        event = np.ones(n)  # no censoring
        a = rng.binomial(1, 0.5, size=n).astype(float)
        tau = 4.0

        s = np.sort(np.unique(time))
        ns = len(s)
        G_perfect = np.ones((n, ns))  # no censoring

        result = ipcw_cut_rmst(a, time, event, tau, G_perfect, G_perfect,
                               time_grid=s)
        expected = np.minimum(time, tau)
        np.testing.assert_array_almost_equal(result, expected, decimal=2)


class TestDoubleRobustness(unittest.TestCase):
    """AIPCW with correct G should approximate IPCW."""

    def test_aipcw_approx_ipcw(self):
        d = _make_survival_data(n=300, seed=99)
        # Use correct G and some S for AIPCW
        ipcw_result = ipcw_cut_rmst(d['a'], d['time'], d['event'], d['tau'],
                                    d['G'], d['G'], time_grid=d['s'])
        aipcw_result = aipcw_cut_rmst(d['a'], d['time'], d['event'], d['tau'],
                                      d['G'], d['G'], d['S'], d['S'],
                                      time_grid=d['s'])
        # With correct G, AIPCW augmentation terms should not drastically change result.
        # They're not identical due to the augmentation, but should be correlated.
        corr = np.corrcoef(ipcw_result, aipcw_result)[0, 1]
        self.assertGreater(corr, 0.5)


if __name__ == '__main__':
    unittest.main()