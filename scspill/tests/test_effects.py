"""Tests for the identification-formula effects (effects.py)."""

import numpy as np
import pandas as pd
import pytest

from scspill.exceptions import ScspillEstimationError
from scspill.utils.scspill_helpers.effects import (
    RhoSolver,
    counterfactual_controls,
    posterior_effects,
    spillover_effects,
    treated_counterfactual,
)
from scspill.utils.scspill_helpers.setup import prepare_scspill_inputs
from scspill.utils.scspill_helpers.structures import SCSPILLEffects

from .conftest import make_sar_panel

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small():
    panel = make_sar_panel(N=9, T0=15, T1=5, rho=0.3, seed=21)
    truth = panel["truth"]
    return truth


@pytest.fixture(scope="module")
def inputs():
    panel = make_sar_panel(N=9, T0=15, T1=5, rho=0.3, seed=21)
    return prepare_scspill_inputs(
        panel["df"],
        "y",
        "treat",
        "unit",
        "time",
        panel["spatial_W"],
        panel["spatial_w"],
        covariates=None,
    )


# ---------------------------------------------------------------------------
# IDENTIFICATION FORMULAS
# ---------------------------------------------------------------------------


def test_rho_zero_collapses_to_plain_sc(small):
    """At rho = 0 the counterfactual is exactly Yc @ alpha and spillovers vanish."""
    Y0, Yc = small["Y0"], small["Yc"]
    alpha = small["alpha"]
    cf = treated_counterfactual(Y0, Yc, small["Wn"], small["wn"], alpha, 0.0)
    assert np.allclose(cf, Yc @ alpha, atol=1e-12)
    sp = spillover_effects(Y0, Yc, small["Wn"], small["wn"], alpha, 0.0)
    assert np.allclose(sp, 0.0, atol=1e-12)


def test_truth_recovers_planted_effects(small):
    """At the true (alpha, rho), the identified effects equal the realized taus."""
    T0 = small["T0"]
    cf = treated_counterfactual(
        small["Y0"][T0:],
        small["Yc"][T0:],
        small["Wn"],
        small["wn"],
        small["alpha"],
        small["rho"],
    )
    te = small["Y0"][T0:] - cf
    assert np.allclose(te, small["taus"], atol=1e-10)


def test_hand_computed_two_unit_case():
    """Closed-form check on a 2-control system."""
    Wn = np.array([[0.0, 1.0], [1.0, 0.0]])
    wn = np.array([0.7, 0.3])
    alpha = np.array([0.6, 0.4])
    rho = 0.25
    Y0 = np.array([2.0])
    Yc = np.array([[1.0, 3.0]])
    I2 = np.eye(2)
    Mmat = I2 - rho * Wn - rho * np.outer(wn, alpha)
    RHS = (I2 - rho * Wn) @ Yc.T - rho * np.outer(wn, Y0)
    expected = np.linalg.solve(Mmat, RHS).T
    got = counterfactual_controls(Y0, Yc, Wn, wn, alpha, rho)
    assert np.allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# RHO SOLVER (Sherman-Morrison fast path)
# ---------------------------------------------------------------------------


def test_solver_matches_direct_solve(small):
    solver = RhoSolver(small["Wn"], small["wn"])
    assert solver.cond_ok
    rng = np.random.default_rng(3)
    for _ in range(10):
        rho = float(rng.uniform(-0.9, 0.9))
        alpha = rng.standard_normal(9) * 0.3
        direct = counterfactual_controls(
            small["Y0"], small["Yc"], small["Wn"], small["wn"], alpha, rho
        )
        fast = solver.counterfactual_controls(small["Y0"], small["Yc"], alpha, rho)
        assert np.allclose(fast, direct, atol=1e-9)


def test_solver_cache_reuse(small):
    solver = RhoSolver(small["Wn"], small["wn"])
    cache = {}
    a = solver.counterfactual_controls(small["Y0"], small["Yc"], small["alpha"], 0.2, _cache=cache)
    assert "PinvYcT" in cache
    b = solver.counterfactual_controls(small["Y0"], small["Yc"], small["alpha"], 0.2, _cache=cache)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# POSTERIOR SWEEP
# ---------------------------------------------------------------------------


def _fake_draws(inputs, M=50, seed=0):
    rng = np.random.default_rng(seed)
    alpha_hat = np.linspace(-0.2, 0.5, inputs.N)
    alpha_draws = alpha_hat[None, :] + 0.01 * rng.standard_normal((M, inputs.N))
    rho_draws = 0.3 + 0.02 * rng.standard_normal(M)
    return alpha_draws, rho_draws, alpha_hat


def test_posterior_effects_output(inputs):
    alpha_draws, rho_draws, alpha_hat = _fake_draws(inputs)
    eff = posterior_effects(inputs, alpha_draws, rho_draws, alpha_hat, 0.3)
    assert isinstance(eff, SCSPILLEffects)
    assert eff.cf_mean.shape == (inputs.T,)
    assert eff.att_draws.shape == (50,)
    assert eff.att_ci[0] <= eff.att <= eff.att_ci[1]
    assert np.all(eff.cf_lower <= eff.cf_upper)
    assert isinstance(eff.spill_mean, pd.DataFrame)
    assert eff.spill_mean.shape == (inputs.T, inputs.N)
    assert list(eff.spill_mean.columns) == list(inputs.control_labels)
    assert eff.n_draws_used == 50
    assert eff.propagate_alpha is True


def test_posterior_effects_deterministic(inputs):
    """No randomness in the sweep: repeated calls agree to rounding error."""
    alpha_draws, rho_draws, alpha_hat = _fake_draws(inputs)
    a = posterior_effects(inputs, alpha_draws, rho_draws, alpha_hat, 0.3)
    b = posterior_effects(inputs, alpha_draws, rho_draws, alpha_hat, 0.3)
    assert np.allclose(a.att_draws, b.att_draws, rtol=1e-9, atol=1e-12)


def test_propagate_alpha_false_ignores_alpha_draws(inputs):
    alpha_draws, rho_draws, alpha_hat = _fake_draws(inputs)
    garbage = np.full_like(alpha_draws, 99.0)
    a = posterior_effects(inputs, garbage, rho_draws, alpha_hat, 0.3, propagate_alpha=False)
    b = posterior_effects(inputs, alpha_draws, rho_draws, alpha_hat, 0.3, propagate_alpha=False)
    assert np.array_equal(a.att_draws, b.att_draws)
    assert a.propagate_alpha is False


def test_unequal_draw_counts_raise(inputs):
    alpha_draws, rho_draws, alpha_hat = _fake_draws(inputs)
    with pytest.raises(ScspillEstimationError, match="differ in length"):
        posterior_effects(inputs, alpha_draws[:30], rho_draws, alpha_hat, 0.3)


def test_max_draws_thinning(inputs):
    alpha_draws, rho_draws, alpha_hat = _fake_draws(inputs)
    eff = posterior_effects(inputs, alpha_draws, rho_draws, alpha_hat, 0.3, max_draws=10)
    assert eff.n_draws_used == 10


def test_att_scm_is_rho_zero_case(inputs):
    alpha_draws, rho_draws, alpha_hat = _fake_draws(inputs)
    eff = posterior_effects(inputs, alpha_draws, rho_draws, alpha_hat, 0.3)
    expected = float((inputs.Y0[inputs.T0 :] - inputs.Yc[inputs.T0 :] @ alpha_hat).mean())
    assert eff.att_scm == pytest.approx(expected, abs=1e-12)


def test_effects_frozen(inputs):
    alpha_draws, rho_draws, alpha_hat = _fake_draws(inputs)
    eff = posterior_effects(inputs, alpha_draws, rho_draws, alpha_hat, 0.3)
    with pytest.raises(AttributeError):
        eff.att = 0.0
    assert not eff.att_draws.flags.writeable
