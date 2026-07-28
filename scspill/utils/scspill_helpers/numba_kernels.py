"""Optional numba-compiled sampler kernels.

Compiles the very same loop functions defined in
:mod:`scspill.utils.scspill_helpers._kernels` with ``numba.njit`` -- the two
backends share one source, so they cannot drift algorithmically (they are not
bit-identical, because JIT compilation may reassociate floating-point
operations).

This module is imported lazily by :func:`_kernels.resolve_backend`; importing
it without numba installed raises ``ImportError``, which the resolver
translates (``backend="numba"``) or absorbs (``backend="auto"``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def build_kernels() -> SimpleNamespace:
    """Compile and warm up the numba versions of the sampler loops.

    The warm-up calls run both kernels on tiny inputs so that any unsupported
    numba feature surfaces here (at resolve time) rather than mid-fit.

    Returns
    -------
    types.SimpleNamespace
        With attributes ``name``, ``hs_alpha_loop``, ``sar_step2_loop``.

    Raises
    ------
    ImportError
        If numba is not installed.
    Exception
        If numba fails to compile the kernels (e.g. an old numba without
        ``np.random.Generator`` support).
    """
    from numba import njit

    from . import _kernels

    hs = njit(cache=True)(_kernels.hs_alpha_loop)
    sar = njit(cache=True)(_kernels.sar_step2_loop)

    # Warm-up: exercise every runtime branch (factors on/off, covariates
    # on/off, horseshoe and ridge beta priors, adaptation on/off).
    rng = np.random.default_rng(0)
    ys = rng.standard_normal(6)
    Xs = rng.standard_normal((6, 4))
    hs(rng, ys, Xs, 4, 2)

    Yc = rng.standard_normal((6, 4))
    A = 0.1 * rng.standard_normal((4, 4))
    AYc = (A @ Yc.T).T
    evA = np.linalg.eigvals(A).astype(np.complex128)
    X2 = rng.standard_normal((24, 1))
    sar(rng, Yc, AYc, evA, X2, 1, 1, 4, 2, 0.05, True, 0.44, -0.95, 0.95, 1.0, 1.0, True)
    sar(rng, Yc, AYc, evA, X2, 1, 0, 4, 2, 0.05, False, 0.44, -0.95, 0.95, 1.0, 1.0, False)
    sar(
        rng,
        Yc,
        AYc,
        evA,
        np.zeros((0, 0)),
        0,
        0,
        4,
        2,
        0.05,
        False,
        0.44,
        -0.95,
        0.95,
        1.0,
        1.0,
        False,
    )

    return SimpleNamespace(name="numba", hs_alpha_loop=hs, sar_step2_loop=sar)
