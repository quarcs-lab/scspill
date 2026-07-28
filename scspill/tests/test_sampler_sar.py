"""Tests for the Step-2 SAR sampler (sampler_sar.py) and its Geweke seams."""

import numpy as np
import pytest

from scspill.utils.scspill_helpers.sar._kernels import resolve_backend
from scspill.utils.scspill_helpers.sar.sampler_sar import (
    SARState,
    draw_prior_state,
    initial_state,
    make_sar_data,
    one_sweep,
    rho_stability_bound,
    sar_step2_sampler,
)
from scspill.utils.scspill_helpers.structures import SARPosterior

from .conftest import assert_chains_reproducible, make_sar_panel, paper_alpha

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _sar_pre_data(N=16, T0=40, rho=0.4, sigma2=0.1, K=0, seed=0):
    """Simulated pre-period SAR data with known (alpha, rho)."""
    panel = make_sar_panel(N=N, T0=T0, T1=2, rho=rho, sigma2=sigma2, K=K, seed=seed)
    truth = panel["truth"]
    Yc_pre = truth["Yc"][:T0]
    X = None
    if K > 0:
        # Rebuild the covariate cube from the panel.
        df = panel["df"]
        labels = sorted(panel["units"])
        cubes = []
        for c in panel["covariates"]:
            cw = df.pivot(index="time", columns="unit", values=c)[labels]
            cubes.append(cw.to_numpy()[:T0])
        X = np.stack(cubes, axis=2)
    return Yc_pre, truth, X


# ---------------------------------------------------------------------------
# SAMPLER
# ---------------------------------------------------------------------------


def test_rho_recovery_positive():
    Yc_pre, truth, _ = _sar_pre_data(rho=0.4, seed=2)
    post = sar_step2_sampler(
        np.random.default_rng(1),
        Yc_pre,
        truth["alpha"],
        truth["wn"],
        truth["Wn"],
        2000,
        1000,
        p=0,
    )
    assert isinstance(post, SARPosterior)
    assert post.rho_hat == pytest.approx(0.4, abs=0.10)
    lo, hi = np.quantile(post.rho, [0.025, 0.975])
    assert lo <= 0.4 <= hi
    assert truth["sigma2"] * 0.7 < post.sigma2.mean() < truth["sigma2"] * 1.3


def test_rho_recovery_negative():
    Yc_pre, truth, _ = _sar_pre_data(rho=-0.5, seed=4)
    post = sar_step2_sampler(
        np.random.default_rng(3),
        Yc_pre,
        truth["alpha"],
        truth["wn"],
        truth["Wn"],
        2000,
        1000,
        p=0,
    )
    assert post.rho_hat == pytest.approx(-0.5, abs=0.10)


def test_support_and_acceptance():
    Yc_pre, truth, _ = _sar_pre_data(seed=5)
    post = sar_step2_sampler(
        np.random.default_rng(6),
        Yc_pre,
        truth["alpha"],
        truth["wn"],
        truth["Wn"],
        1500,
        700,
        p=0,
    )
    bnd = rho_stability_bound(truth["Wn"])
    assert bnd == pytest.approx(0.95)  # row-stochastic W
    assert np.all(np.abs(post.rho) < bnd)
    assert 0.05 < post.acc_rho < 0.8


def test_adaptation_moves_step_and_freezes():
    Yc_pre, truth, _ = _sar_pre_data(seed=7)
    # Absurdly large initial step: adaptation must shrink it toward the target.
    adapted = sar_step2_sampler(
        np.random.default_rng(8),
        Yc_pre,
        truth["alpha"],
        truth["wn"],
        truth["Wn"],
        2000,
        1000,
        p=0,
        step_rho=0.9,
        adapt_rho=True,
    )
    assert adapted.step_rho_final < 0.9
    assert 0.2 < adapted.acc_rho < 0.7  # near the 0.44 target
    fixed = sar_step2_sampler(
        np.random.default_rng(8),
        Yc_pre,
        truth["alpha"],
        truth["wn"],
        truth["Wn"],
        800,
        400,
        p=0,
        step_rho=0.9,
        adapt_rho=False,
    )
    assert fixed.step_rho_final == pytest.approx(0.9)


def test_explicit_rho_support_respected():
    Yc_pre, truth, _ = _sar_pre_data(rho=0.4, seed=9)
    post = sar_step2_sampler(
        np.random.default_rng(10),
        Yc_pre,
        truth["alpha"],
        truth["wn"],
        truth["Wn"],
        800,
        400,
        p=0,
        rho_support=(-0.2, 0.2),
    )
    assert np.all(post.rho > -0.2) and np.all(post.rho < 0.2)


def test_factor_and_covariate_branches_run():
    Yc_pre, truth, X = _sar_pre_data(K=2, seed=11)
    for beta_prior in ("horseshoe", "ridge"):
        post = sar_step2_sampler(
            np.random.default_rng(12),
            Yc_pre,
            truth["alpha"],
            truth["wn"],
            truth["Wn"],
            400,
            200,
            X=X,
            p=1,
            beta_prior=beta_prior,
        )
        assert post.beta.shape == (200, 2)
        assert np.all(np.isfinite(post.beta))
        assert post.beta_prior == beta_prior
        # beta ~ 1.0 in the DGP; even short chains should be in the vicinity.
        assert np.allclose(post.beta.mean(axis=0), 1.0, atol=0.5)


def test_logdet_matches_slogdet():
    """Eigen-cached log-determinant equals dense slogdet of (I - rho A)."""
    from scspill.utils.scspill_helpers.sar._kernels import _rho_loglik

    _Yc_pre, truth, _ = _sar_pre_data(seed=13)
    alpha = truth["alpha"]
    A = truth["Wn"] + np.outer(truth["wn"], alpha)
    evA = np.linalg.eigvals(A).astype(np.complex128)
    N = A.shape[0]
    for rho in (-0.8, -0.3, 0.0, 0.3, 0.8):
        cached = _rho_loglik(rho, evA, 0.0, 0.0, 0.0, 1.0, 1, N, -0.95, 0.95)
        _, logdet = np.linalg.slogdet(np.eye(N) - rho * A)
        assert cached == pytest.approx(logdet, abs=1e-8)


def test_seed_reproducibility():
    Yc_pre, truth, _ = _sar_pre_data(seed=14)
    kw = dict(X=None, p=1)
    a = sar_step2_sampler(
        np.random.default_rng(0), Yc_pre, truth["alpha"], truth["wn"], truth["Wn"], 800, 300, **kw
    )
    b = sar_step2_sampler(
        np.random.default_rng(0), Yc_pre, truth["alpha"], truth["wn"], truth["Wn"], 800, 300, **kw
    )
    assert_chains_reproducible(a.rho, b.rho, atol=0.05)
    assert_chains_reproducible(a.sigma2, b.sigma2, atol=0.05)


# ---------------------------------------------------------------------------
# GEWEKE SEAMS: one_sweep parity and prior draws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K,p,beta_hs", [(0, 0, True), (2, 1, True), (2, 1, False)])
def test_one_sweep_matches_batch_loop(K, p, beta_hs):
    """one_sweep must replicate the batch loop draw-for-draw (fixed step)."""
    Yc_pre, truth, X = _sar_pre_data(N=9, T0=15, K=K, seed=15)
    data = make_sar_data(
        Yc_pre,
        truth["alpha"],
        truth["wn"],
        truth["Wn"],
        X=X,
        p=p,
        step_rho=0.05,
        beta_prior="horseshoe" if beta_hs else "ridge",
    )
    n_sweeps = 6
    kernels = resolve_backend("numpy")
    rng_loop = np.random.default_rng(99)
    rho_d, s2_d, beta_d, _, _ = kernels.sar_step2_loop(
        rng_loop,
        data.Yc,
        data.AYc,
        data.evA,
        data.X2,
        data.K,
        data.p,
        n_sweeps,
        0,  # burn=0: keep every draw
        data.step_rho,
        False,  # adapt off: fixed transition kernel
        0.44,
        data.rho_lo,
        data.rho_hi,
        data.a0,
        data.b0,
        data.beta_hs,
    )
    rng_sweep = np.random.default_rng(99)
    state = initial_state(data)
    for m in range(n_sweeps):
        one_sweep(state, data, rng_sweep)
        # Tolerance instead of equality: some BLAS builds (Apple Accelerate)
        # select kernels by memory alignment, and the two code paths allocate
        # their temporaries differently.
        assert state.rho == pytest.approx(rho_d[m], rel=1e-9, abs=1e-12)
        assert state.s2 == pytest.approx(s2_d[m], rel=1e-9, abs=1e-12)
        if K > 0:
            assert np.allclose(state.beta, beta_d[m], rtol=1e-9, atol=1e-12)


def test_initial_state_shapes():
    Yc_pre, truth, X = _sar_pre_data(N=9, T0=15, K=2, seed=16)
    data = make_sar_data(Yc_pre, truth["alpha"], truth["wn"], truth["Wn"], X=X, p=2)
    state = initial_state(data)
    assert isinstance(state, SARState)
    assert state.beta.shape == (2,)
    assert state.Eta.shape == (9, 2)
    assert state.Gamma.shape == (2, 15)
    assert state.omega.shape == (2,)


def test_draw_prior_state_distributions():
    Yc_pre, truth, X = _sar_pre_data(N=9, T0=15, K=2, seed=17)
    data = make_sar_data(Yc_pre, truth["alpha"], truth["wn"], truth["Wn"], X=X, p=1)
    rng = np.random.default_rng(5)
    rhos, s2s, phis = [], [], []
    for _ in range(400):
        st = draw_prior_state(data, rng)
        rhos.append(st.rho)
        s2s.append(st.s2)
        phis.append(st.phi_g)
        assert st.Gamma.shape == (1, 15)
        assert st.Eta.shape == (9, 1)
        assert st.beta.shape == (2,)
    rhos = np.array(rhos)
    # rho ~ U(rho_lo, rho_hi): mean near 0, full spread.
    assert abs(rhos.mean()) < 0.1
    assert rhos.min() < -0.7 and rhos.max() > 0.7
    assert np.all(np.abs(phis) < 1.0)
    assert np.all(np.array(s2s) > 0)


def test_paper_alpha_helper():
    alpha = paper_alpha(16)
    assert alpha[0] == 0.5 and alpha[1] == -0.2
    assert np.all(alpha[10:] == 0)
