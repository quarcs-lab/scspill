"""Statistical parity between the NumPy and numba sampler backends.

Runs only when the optional numba extra is installed. The two backends
compile the same source functions, so they must agree statistically (they are
not bit-identical: JIT compilation may reassociate floating-point
arithmetic).
"""

import numpy as np
import pytest

from scspill import SCSPILL
from scspill.utils.scspill_helpers._kernels import resolve_backend

from .conftest import base_config_kwargs

numba = pytest.importorskip("numba")


@pytest.fixture(scope="module")
def kernels():
    return resolve_backend("numba")


def test_backend_resolves(kernels):
    assert kernels.name == "numba"


def test_alpha_loop_statistical_parity(kernels):
    rng_np = np.random.default_rng(0)
    T0, N = 60, 8
    X = rng_np.standard_normal((T0, N))
    y = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.05 * rng_np.standard_normal(T0)
    sx = np.maximum(X.std(axis=0, ddof=1), 1e-8)
    sy = max(float(y.std(ddof=1)), 1e-8)
    numpy_k = resolve_backend("numpy")
    a_np, _ = numpy_k.hs_alpha_loop(np.random.default_rng(1), y / sy, X / sx, 2000, 1000)
    a_nb, _ = kernels.hs_alpha_loop(np.random.default_rng(1), y / sy, X / sx, 2000, 1000)
    assert a_nb.shape == a_np.shape
    # Same algorithm, same seed-family: posterior means agree within MC error.
    assert np.allclose(a_np.mean(axis=0), a_nb.mean(axis=0), atol=0.05)


@pytest.mark.slow
def test_estimator_backend_parity(sar_panel):
    res_np = SCSPILL(base_config_kwargs(sar_panel, m_iter=1500, burn=700, backend="numpy")).fit()
    res_nb = SCSPILL(base_config_kwargs(sar_panel, m_iter=1500, burn=700, backend="numba")).fit()
    assert res_nb.rho_hat == pytest.approx(res_np.rho_hat, abs=0.05)
    assert res_nb.att == pytest.approx(res_np.att, abs=0.1)
