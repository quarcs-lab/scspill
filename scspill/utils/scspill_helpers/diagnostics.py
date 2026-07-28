"""MCMC convergence diagnostics for SCSPILL chains.

Pure functions mirroring the R package's ``diagnostics()`` internals:
effective sample size via the initial-positive-sequence autocorrelation
estimator (Geyer 1992, FFT-based), split-chain R-hat, Monte Carlo standard
error, and a Geweke z-score comparing early and late chain segments -- plus
:func:`mcmc_summary`, which assembles them into the standard posterior
summary table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ess_acf(x: np.ndarray, max_lag: int | None = None) -> float:
    """Effective sample size of a chain via initial-positive-sequence ACF.

    ``n / (1 + 2 sum(paired positive autocorrelations))`` with the sum
    truncated at the first negative pair (Geyer 1992). Returns ``n`` for
    white noise and shrinks toward 1 as the chain becomes sticky.

    Parameters
    ----------
    x : np.ndarray
        The chain, shape ``(n,)``.
    max_lag : int, optional
        Truncate the autocorrelation sum at this lag.

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = float(x @ x / n)
    if var <= 0:
        return float(n)
    # autocovariances via FFT
    m = 1
    while m < 2 * n:
        m *= 2
    f = np.fft.rfft(x, m)
    acov = np.fft.irfft(f * np.conjugate(f), m)[:n].real / n
    rho = acov / acov[0]
    stop = n - 1 if max_lag is None else min(max_lag, n - 1)
    s = 0.0
    for k in range(1, stop, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        s += pair
    tau = 1.0 + 2.0 * s
    return float(n / max(tau, 1.0))


def split_rhat(x: np.ndarray) -> float:
    """Split-chain R-hat of a single chain (halves as pseudo-chains).

    The Gelman-Rubin potential scale reduction factor computed on the two
    halves of the chain; values near 1 indicate the halves agree in mean and
    variance.

    Parameters
    ----------
    x : np.ndarray
        The chain, shape ``(n,)``.

    Returns
    -------
    float
        ``nan`` when the chain is too short or degenerate.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size // 2
    if n < 2:
        return float("nan")
    chains = np.stack([x[:n], x[n : 2 * n]])
    means = chains.mean(axis=1)
    variances = chains.var(axis=1, ddof=1)
    W = float(variances.mean())
    B = float(n * means.var(ddof=1))
    if W <= 0:
        return float("nan")
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def mcse_from_ess(x: np.ndarray, ess: float) -> float:
    """Monte Carlo standard error ``sd(x) / sqrt(ess)``.

    Parameters
    ----------
    x : np.ndarray
        The chain.
    ess : float
        Its effective sample size.

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2 or ess <= 0:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(ess))


def geweke_z(x: np.ndarray, frac_first: float = 0.1, frac_last: float = 0.5) -> float:
    """Geweke convergence z-score comparing early and late chain segments.

    Compares the means of the first ``frac_first`` and last ``frac_last``
    fractions of the chain, with segment variances scaled by their own
    effective sample sizes (an ESS-based stand-in for the spectral-density
    estimator of the classical statistic).

    Parameters
    ----------
    x : np.ndarray
        The chain.
    frac_first, frac_last : float
        Fractions defining the two segments.

    Returns
    -------
    float
        ``nan`` when the segments are too short.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    n1 = int(np.floor(frac_first * n))
    n2 = int(np.floor(frac_last * n))
    if n1 < 5 or n2 < 5:
        return float("nan")
    a, b = x[:n1], x[n - n2 :]
    va = a.var(ddof=1) / max(ess_acf(a), 1.0)
    vb = b.var(ddof=1) / max(ess_acf(b), 1.0)
    denom = np.sqrt(va + vb)
    if denom <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / denom)


def mcmc_summary(
    chains: dict[str, np.ndarray],
    probs: tuple = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """Posterior summary table for a set of named chains.

    Parameters
    ----------
    chains : dict of str -> np.ndarray
        Named chains (each 1-D).
    probs : tuple of float, default (0.025, 0.5, 0.975)
        Quantiles reported as ``q025`` / ``q50`` / ``q975``-style columns.

    Returns
    -------
    pd.DataFrame
        Indexed by parameter name with columns ``mean``, ``sd``, one per
        quantile, ``ess``, ``rhat_split``, ``mcse``, ``geweke_z``.
    """
    rows = {}

    def _qname(p: float) -> str:
        digits = f"{p:g}".split(".")[1] if "." in f"{p:g}" else "0"
        if len(digits) == 1:
            digits += "0"  # 0.5 -> q50
        return f"q{digits}"

    qcols = [_qname(p) for p in probs]
    for name, chain in chains.items():
        chain = np.asarray(chain, dtype=float).ravel()
        ess = ess_acf(chain)
        qs = np.quantile(chain, probs)
        row = {
            "mean": float(chain.mean()),
            "sd": float(chain.std(ddof=1)) if chain.size > 1 else float("nan"),
        }
        row.update(dict(zip(qcols, map(float, qs), strict=True)))
        row.update(
            {
                "ess": ess,
                "rhat_split": split_rhat(chain),
                "mcse": mcse_from_ess(chain, ess),
                "geweke_z": geweke_z(chain),
            }
        )
        rows[name] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "parameter"
    return out
