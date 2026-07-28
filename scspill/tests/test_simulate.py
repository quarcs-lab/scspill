"""Tests for the Monte Carlo simulation engine (scspill.simulate)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scspill.exceptions import ScspillConfigError, ScspillDataError
from scspill.simulate import (
    SimRunResult,
    load_r_mc_reference,
    make_w,
    mc_grid,
    rook_W,
    run_many_sim,
    run_one_sim,
    scspill_sim_dgp,
    summarize_many,
)
from scspill.simulate.dgp import paper_alpha

# ---------------------------------------------------------------------------
# LATTICE / WEIGHTS
# ---------------------------------------------------------------------------


def test_rook_W_2x3_hand_checked():
    W = rook_W(2, 3)
    # Unit 0 (row 0, col 0) borders unit 1 (right) and unit 3 (below).
    expected_row0 = np.array([0, 1, 0, 1, 0, 0], dtype=float)
    assert np.array_equal(W[0], expected_row0)
    assert np.array_equal(W, W.T)
    assert np.all(np.diag(W) == 0)


def test_rook_W_3x3_degrees():
    W = rook_W(3, 3)
    degrees = W.sum(axis=1)
    # Corners have 2 neighbors, edges 3, the center 4.
    assert sorted(degrees.tolist()) == [2, 2, 2, 2, 3, 3, 3, 3, 4]


def test_rook_W_normalized():
    W = rook_W(3, 3, normalize=True)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_make_w():
    w = make_w(6, treated=(0, 2))
    assert np.array_equal(w, [1, 0, 1, 0, 0, 0])
    w1 = make_w(6, treated=1)
    assert np.array_equal(w1, [0, 1, 0, 0, 0, 0])


def test_paper_alpha_requires_ten_units():
    with pytest.raises(ScspillDataError):
        paper_alpha(9)


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


def _dgp(rho=0.3, K=1, seed=5, **kw):
    N = 16
    return scspill_sim_dgp(
        T0=kw.pop("T0", 20),
        T1=kw.pop("T1", 6),
        N=N,
        W=rook_W(4, 4),
        w=make_w(N),
        rho=rho,
        sigma2=0.1,
        alpha=paper_alpha(N),
        K=K,
        beta=np.ones(K) if K > 0 else None,
        seed=seed,
        **kw,
    )


def test_dgp_shapes():
    dgp = _dgp()
    assert dgp.Y0_pre.shape == (20,) and dgp.Y0_post.shape == (6,)
    assert dgp.Yc_pre.shape == (20, 16) and dgp.Yc_post.shape == (6, 16)
    assert dgp.X_pre.shape == (20, 16, 1) and dgp.X_post.shape == (6, 16, 1)
    assert dgp.T0 == 20 and dgp.T1 == 6 and dgp.N == 16 and dgp.K == 1


def test_dgp_seed_reproducibility():
    a = _dgp(seed=9)
    b = _dgp(seed=9)
    assert np.allclose(a.Yc_pre, b.Yc_pre, rtol=1e-12)
    assert np.allclose(a.truth.tau_post, b.truth.tau_post, rtol=1e-12)
    c = _dgp(seed=10)
    assert not np.array_equal(a.Yc_pre, c.Yc_pre)


def test_dgp_perfect_fit_invariant():
    """Pre-treatment, the treated unit is exactly alpha' Yc."""
    dgp = _dgp()
    assert np.allclose(dgp.Y0_pre, dgp.Yc_pre @ dgp.truth.alpha, atol=1e-10)


def test_dgp_tau_definition():
    dgp = _dgp()
    assert np.array_equal(dgp.truth.tau_post, dgp.Y0_post - dgp.truth.y0_cf_post)


def test_dgp_post_sar_identity():
    """(I - rho Wn) Yc_post_t' == rho wn Y0_post_t + X_t beta + e_t."""
    dgp = _dgp(keep_internals=True)
    rho = dgp.truth.rho
    I_N = np.eye(dgp.N)
    for tt in range(dgp.T1):
        lhs = (I_N - rho * dgp.Wn) @ dgp.Yc_post[tt]
        rhs = (
            rho * dgp.wn * dgp.Y0_post[tt]
            + dgp.X_post[tt] @ dgp.truth.beta
            + dgp.errors[dgp.T0 + tt]
        )
        assert np.allclose(lhs, rhs, atol=1e-8)


def test_dgp_singular_rho_raises():
    # rho = 1 makes I - rho * Wn singular for a row-stochastic Wn.
    with pytest.raises(ScspillDataError, match="singular"):
        _dgp(rho=1.0)


def test_dgp_validation_errors():
    with pytest.raises(ScspillDataError):
        scspill_sim_dgp(
            T0=10,
            T1=2,
            N=16,
            W=rook_W(4, 4),
            w=make_w(16),
            rho=0.2,
            sigma2=0.1,
            alpha=paper_alpha(16),
            K=1,
            beta=None,
        )
    with pytest.raises(ScspillDataError):
        scspill_sim_dgp(
            T0=10,
            T1=2,
            N=16,
            W=np.zeros((4, 4)),
            w=make_w(16),
            rho=0.2,
            sigma2=0.1,
            alpha=paper_alpha(16),
        )


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------


def test_run_one_sim_metrics_frame():
    res = run_one_sim(
        dgp_args={
            "grid": (3, 3),
            "T0": 12,
            "T1": 4,
            "rho": 0.2,
            "sigma2": 0.1,
            "alpha": np.array([0.5, -0.2, 0.4, 0.4, 0, 0, 0, 0, 0.0]),
            "K": 0,
        },
        m_iter=300,
        burn=150,
        seed=3,
    )
    assert isinstance(res, SimRunResult)
    m = res.metrics
    assert list(m["method"]) == ["SCM", "BSCM", "SCSPILL"]
    for col in ("bias_ate", "mse_ate", "bias_point", "mse_point", "cover95_ate", "cover95_point"):
        assert col in m.columns
    # Coverage: NaN for SCM, in [0, 1] for the Bayesian methods.
    assert np.isnan(m.loc[0, "cover95_point"])
    for i in (1, 2):
        assert 0.0 <= m.loc[i, "cover95_point"] <= 1.0
        assert m.loc[i, "cover95_ate"] in (0.0, 1.0)
    assert res.truth is None and res.draws is None


def test_run_one_sim_keep_full():
    res = run_one_sim(
        dgp_args={"grid": (4, 4), "T0": 12, "T1": 4, "rho": 0.2, "sigma2": 0.1, "K": 0},
        m_iter=300,
        burn=150,
        seed=4,
        keep_full=True,
    )
    assert res.truth is not None
    assert res.draws["alpha_bscm"].shape == (150, 16)
    assert res.per_time["true"].shape == (4,)
    assert np.all(res.per_time["ci"]["scspill"]["lower"] <= res.per_time["ci"]["scspill"]["upper"])


def test_run_one_sim_requires_dgp_or_args():
    with pytest.raises(ScspillConfigError):
        run_one_sim()


def test_run_many_sim_seeds_and_order():
    args = {
        "grid": (3, 3),
        "T0": 10,
        "T1": 3,
        "rho": 0.1,
        "sigma2": 0.1,
        "alpha": np.array([0.5, -0.2, 0.4, 0.4, 0, 0, 0, 0, 0.0]),
        "K": 0,
    }
    with pytest.raises(ScspillConfigError):
        run_many_sim(3, args, seeds=[1, 2])  # wrong length
    results = run_many_sim(3, args, seeds=[1, 2, 3], m_iter=200, burn=100)
    assert len(results) == 3
    again = run_many_sim(3, args, seeds=[1, 2, 3], m_iter=200, burn=100)
    for a, b in zip(results, again, strict=True):
        # Same seeds -> statistically identical metrics (bit-exact only when
        # the platform BLAS is deterministic).
        assert np.allclose(
            a.metrics[["bias_point", "bias_ate"]].to_numpy(),
            b.metrics[["bias_point", "bias_ate"]].to_numpy(),
            atol=0.05,
        )


def test_run_many_sim_parallel_matches_serial():
    args = {
        "grid": (3, 3),
        "T0": 10,
        "T1": 3,
        "rho": 0.1,
        "sigma2": 0.1,
        "alpha": np.array([0.5, -0.2, 0.4, 0.4, 0, 0, 0, 0, 0.0]),
        "K": 0,
    }
    serial = run_many_sim(2, args, seeds=[7, 8], m_iter=200, burn=100, n_jobs=1)
    parallel = run_many_sim(2, args, seeds=[7, 8], m_iter=200, burn=100, n_jobs=2)
    assert len(parallel) == 2
    for a, b in zip(serial, parallel, strict=True):
        assert np.allclose(
            a.metrics["bias_point"].to_numpy(), b.metrics["bias_point"].to_numpy(), atol=0.05
        )


# ---------------------------------------------------------------------------
# SUMMARY / GRID
# ---------------------------------------------------------------------------


def _fake_result(bias_scspill: float) -> SimRunResult:
    m = pd.DataFrame(
        {
            "bias_ate": [0.5, 0.2, bias_scspill],
            "mse_ate": [0.25, 0.04, bias_scspill**2],
            "bias_point": [0.5, 0.2, bias_scspill],
            "mse_point": [0.25, 0.04, bias_scspill**2],
            "cover95_ate": [np.nan, 1.0, 1.0],
            "cover95_point": [np.nan, 0.9, 0.95],
            "method": ["SCM", "BSCM", "SCSPILL"],
        }
    )
    return SimRunResult(metrics=m)


def test_summarize_many_hand_checked():
    out = summarize_many([_fake_result(0.1), _fake_result(0.3)])
    assert list(out["method"]) == ["SCM", "BSCM", "SCSPILL"]
    row = out[out["method"] == "SCSPILL"].iloc[0]
    assert row["bias_point"] == pytest.approx(0.2)
    assert row["mse_point"] == pytest.approx((0.01 + 0.09) / 2)
    assert row["rmse_point"] == pytest.approx(np.sqrt(0.05))
    assert row["cover95_point"] == pytest.approx(0.95)
    assert row["n_sims"] == 2
    scm = out[out["method"] == "SCM"].iloc[0]
    assert np.isnan(scm["cover95_point"])


def test_summarize_many_empty_raises():
    with pytest.raises(ScspillDataError):
        summarize_many([])


def test_mc_grid_smoke_and_schema():
    out = mc_grid(
        Ns=(16,),
        T0s=(12,),
        T1=3,
        rhos=(0.3,),
        sims_per=2,
        K=0,
        m_iter=200,
        burn=100,
        seed=1,
    )
    assert list(out.columns) == [
        "N",
        "T0",
        "T1",
        "rho",
        "method",
        "bias_point",
        "rmse_point",
        "cover95_point",
    ]
    assert len(out) == 3  # one scenario x three methods
    assert set(out["method"]) == {"SCM", "BSCM", "SCSPILL"}


def test_mc_grid_rejects_nonsquare_N():
    with pytest.raises(ScspillConfigError):
        mc_grid(Ns=(15,), T0s=(12,), rhos=(0.1,), sims_per=1)


_R_MC_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "reference" / "mc_result"


@pytest.mark.skipif(not _R_MC_DIR.exists(), reason="frozen R reference CSVs not present")
def test_load_r_mc_reference():
    ref = load_r_mc_reference(_R_MC_DIR)
    assert set(ref.columns) >= {"N", "T0", "T1", "rho", "method", "bias_point", "rmse_point"}
    assert "SCSPILL" in set(ref["method"])


def test_load_r_mc_reference_missing_dir(tmp_path):
    with pytest.raises(ScspillDataError):
        load_r_mc_reference(tmp_path)


# ---------------------------------------------------------------------------
# RECOVERY (slow tier)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scspill_beats_bscm_under_spillovers():
    """At rho = 0.3, SCSPILL is nearly unbiased and beats BSCM on RMSE."""
    args = {"grid": (4, 4), "T0": 20, "T1": 6, "rho": 0.3, "sigma2": 0.1, "K": 0}
    results = run_many_sim(20, args, seeds=list(range(20)), m_iter=1500, burn=700)
    summary = summarize_many(results)
    sp = summary[summary["method"] == "SCSPILL"].iloc[0]
    bscm = summary[summary["method"] == "BSCM"].iloc[0]
    assert abs(sp["bias_point"]) < 0.05
    assert sp["rmse_point"] < bscm["rmse_point"]
    assert 0.80 <= sp["cover95_point"] <= 1.0
