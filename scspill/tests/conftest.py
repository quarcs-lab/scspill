"""Shared fixtures for the scspill test suite."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    """Never open interactive figure windows during tests."""
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def rook_lattice(side: int) -> np.ndarray:
    """Binary rook adjacency on a side x side lattice (row-major unit ids)."""
    N = side * side
    W = np.zeros((N, N))
    for i in range(N):
        r, c = divmod(i, side)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < side and 0 <= cc < side:
                W[i, rr * side + cc] = 1.0
    return W


def paper_alpha(N: int) -> np.ndarray:
    """The paper's planted synthetic weights (0.5, -0.2, 0.4, 0.4, 0.1/6 x6, 0...)."""
    alpha = np.zeros(N)
    alpha[0] = 0.5
    alpha[1] = -0.2
    alpha[2] = 0.4
    alpha[3] = 0.4
    alpha[4:10] = 0.1 / 6.0
    return alpha


def make_sar_panel(
    N=16,
    T0=30,
    T1=8,
    rho=0.4,
    sigma2=0.1,
    K=1,
    seed=7,
    mu_tau=1.0,
    sd_tau=1.0,
):
    """Simulate a spillover panel from the paper's SAR DGP.

    Pre-treatment: ``Yc_t = (I - rho W - rho w alpha')^{-1} (X_t beta + e_t)``
    and ``Y0_t = alpha' Yc_t`` (exact perfect fit). Post-treatment: the
    treated outcome gains ``tau_t ~ N(mu_tau, sd_tau)`` and the controls are
    regenerated through the SAR with the treated's treated outcome.

    Returns
    -------
    dict
        ``df`` (long panel with a ``treat`` column), ``spatial_w`` (Series),
        ``spatial_W`` (DataFrame), and ``truth`` (alpha, rho, sigma2, taus,
        arrays, normalized weights).
    """
    side = round(float(np.sqrt(N)))
    assert side * side == N, "N must be a perfect square for the rook lattice"
    rng = np.random.default_rng(seed)
    T = T0 + T1

    W = rook_lattice(side)
    w = np.zeros(N)
    w[:4] = 1.0
    alpha = paper_alpha(N)
    Wn = W / W.sum(axis=1, keepdims=True)
    wn = w / w.sum()
    Apre = np.eye(N) - rho * Wn - rho * np.outer(wn, alpha)
    Apost = np.eye(N) - rho * Wn
    Apre_inv = np.linalg.inv(Apre)

    beta = np.ones(K)
    X = rng.standard_normal((T, N, K)) if K > 0 else np.zeros((T, N, 0))
    E = np.sqrt(sigma2) * rng.standard_normal((T, N))

    Yc = np.empty((T, N))
    Y0 = np.empty(T)
    taus = []
    for t in range(T):
        drive = (X[t] @ beta if K > 0 else 0.0) + E[t]
        yc0 = Apre_inv @ drive
        y00 = float(alpha @ yc0)
        if t < T0:
            Yc[t] = yc0
            Y0[t] = y00
        else:
            tau = float(rng.normal(mu_tau, sd_tau))
            taus.append(tau)
            Y0[t] = y00 + tau
            Yc[t] = np.linalg.solve(Apost, rho * wn * Y0[t] + drive)

    units = [f"u{i + 1:02d}" for i in range(N)]
    rows = []
    for t in range(T):
        row = {"unit": "treated", "time": t, "y": Y0[t], "treat": int(t >= T0)}
        for k in range(K):
            row[f"x{k + 1}"] = 0.0
        rows.append(row)
        for i, u in enumerate(units):
            row = {"unit": u, "time": t, "y": Yc[t, i], "treat": 0}
            for k in range(K):
                row[f"x{k + 1}"] = X[t, i, k]
            rows.append(row)
    df = pd.DataFrame(rows)

    return {
        "df": df,
        "spatial_w": pd.Series(w, index=units),
        "spatial_W": pd.DataFrame(W, index=units, columns=units),
        "covariates": [f"x{k + 1}" for k in range(K)],
        "units": units,
        "truth": {
            "alpha": alpha,
            "rho": rho,
            "sigma2": sigma2,
            "beta": beta,
            "taus": np.array(taus),
            "Y0": Y0,
            "Yc": Yc,
            "Wn": Wn,
            "wn": wn,
            "T0": T0,
        },
    }


@pytest.fixture(scope="session")
def sar_panel():
    """The default planted-truth SAR panel (N=16, T0=30, rho=0.4)."""
    return make_sar_panel()


@pytest.fixture(scope="session")
def california():
    from scspill.data import load_california

    return load_california()


@pytest.fixture(scope="session")
def sudan():
    from scspill.data import load_sudan

    return load_sudan()


def assert_chains_reproducible(a: np.ndarray, b: np.ndarray, atol: float = 0.05) -> None:
    """Assert two same-seed chains are reproducible.

    Bit-exact equality is asserted when the platform's BLAS is deterministic.
    Some BLAS builds (notably Apple's Accelerate) select kernels by memory
    alignment, injecting floating-point-rounding differences that chaotically
    diverge an MCMC chain; in that case the chains must still be
    *statistically* identical, so the fallback asserts their posterior means
    and standard deviations agree.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    if np.array_equal(a, b):
        return
    assert np.allclose(np.mean(a, axis=0), np.mean(b, axis=0), atol=atol)
    assert np.allclose(np.std(a, axis=0), np.std(b, axis=0), atol=atol)


def base_config_kwargs(panel, **overrides):
    """Config kwargs for the synthetic SAR panel, with fast MCMC defaults."""
    kwargs = {
        "df": panel["df"],
        "outcome": "y",
        "treat": "treat",
        "unitid": "unit",
        "time": "time",
        "spatial_w": panel["spatial_w"],
        "spatial_W": panel["spatial_W"],
        "covariates": panel["covariates"],
        "m_iter": 600,
        "burn": 300,
        "seed": 11,
        "display_graphs": False,
        "p_factors": 0,
    }
    kwargs.update(overrides)
    return kwargs
