"""Build rook-contiguity weights for Cao & Dowd's 50-states-plus-DC panel.

The ``sar`` model needs spatial weights; scspill bundles them only for the
39-state California panel (``load_california``). This script builds the same
kind of weights for the 51 units of ``load_california_cd(51)``, from the same
source and with the same rule, for the article ``docs/articles/cd-results.qmd``:

* source: the U.S. Census Bureau TIGER/Line 2024 state shapefile,
  https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip;
* rule: rook contiguity (a shared boundary, not just a corner), as R's
  ``spdep::poly2nb(queen = FALSE)`` builds the bundled 39-state weights.

It runs outside scspill's environment (it needs geopandas and libpysal)::

    python tools/build_cd_contiguity.py path/to/tl_2024_us_state.shp

and writes ``docs/articles/data/cd_rook_W.csv``: a binary 51 x 51 matrix keyed
by the postal codes the Cao-Dowd panel uses. Before writing, it checks that
its 39-state subset reproduces the bundled ``california_W_matrix.csv`` and
``california_w_vector.csv`` exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Rook

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "scspill" / "data" / "files"
OUT = ROOT / "docs" / "articles" / "data" / "cd_rook_W.csv"


def main(shapefile: str) -> None:
    """Build, check and write the 51-unit rook adjacency."""
    states = gpd.read_file(shapefile)
    panel = pd.read_csv(FILES / "cao_dowd" / "cigs_consumption.csv")
    codes = sorted(panel["state"].unique())
    states = states.set_index("STUSPS").loc[codes].reset_index()
    rook = Rook.from_dataframe(states, use_index=False, silence_warnings=True)
    W = pd.DataFrame(0, index=codes, columns=codes, dtype=int)
    for i, neighbours in rook.neighbors.items():
        for j in neighbours:
            W.iloc[i, j] = 1
    assert (W.to_numpy() == W.to_numpy().T).all(), "rook adjacency must be symmetric"
    assert np.all(np.diag(W.to_numpy()) == 0)

    # Self-check against the bundled 39-state weights (full state names).
    names = dict(zip(states["STUSPS"], states["NAME"], strict=True))
    bundled_W = pd.read_csv(FILES / "california_W_matrix.csv").set_index("state")
    bundled_w = pd.read_csv(FILES / "california_w_vector.csv").set_index("state")["adj"]
    by_name = W.rename(index=names, columns=names)
    donors = list(bundled_W.index)
    sub = by_name.loc[donors, donors]
    assert (sub.to_numpy() == bundled_W.loc[donors, donors].to_numpy()).all(), (
        "39-state subset differs from california_W_matrix.csv"
    )
    assert (
        by_name.loc["California", donors].to_numpy() == bundled_w.loc[donors].to_numpy()
    ).all(), "California's row differs from california_w_vector.csv"
    print(f"self-check passed: the {len(donors)}-donor subset matches the bundled weights")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    W.index.name = "state"
    W.to_csv(OUT)
    print(f"wrote {OUT}: {W.shape[0]} units, {int(W.to_numpy().sum()) // 2} borders")
    print("California borders:", list(W.columns[W.loc["CA"] == 1]))
    print("no neighbours:", list(W.index[W.sum(axis=1) == 0]))


if __name__ == "__main__":
    main(sys.argv[1])
