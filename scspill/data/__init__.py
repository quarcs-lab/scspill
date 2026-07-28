"""Bundled case-study datasets for scspill.

Ships the two empirical applications of Sakaguchi & Tagawa inside the wheel,
each as a long panel plus its spatial-weight specification:

* :func:`load_california` -- the California Proposition 99 tobacco panel
  (Abadie, Diamond & Hainmueller 2010): 39 states, 1970-2000, treatment in
  1988; rook-contiguity spatial weights.
* :func:`load_sudan` -- the 2011 Sudan secession panel: 34 African countries,
  2000-2015, treatment in 2011; average bilateral-trade spatial weights.

The CSV files are verbatim copies of the replication package's nonproprietary
data exports. Both weight objects are stored raw (unnormalized) --
row-normalization of ``W`` and sum-normalization of ``w`` happen inside the
estimator, exactly as in the R package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import pandas as pd

from ..exceptions import ScspillDataError

__all__ = ["SpillPanel", "load_california", "load_sudan"]


#: Mapping from the R-mangled WDI column headers in ``sudan_panel.csv`` to the
#: clean snake_case names used by :func:`load_sudan` by default.
SUDAN_COLUMN_MAP = {
    "GDP.per.capita..constant.2015.US..": "gdp_pc",
    "Exports.of.goods.and.services....of.GDP.": "exports_gdp_share",
    "Merchandise.trade....of.GDP.": "merchandise_trade_gdp_share",
    "Trade....of.GDP.": "trade_gdp_share",
    "Access.to.clean.fuels.and.technologies.for.cooking....of.population.": "clean_fuels_access",
    "Inflation..consumer.prices..annual...": "inflation",
    "Net.migration": "net_migration",
}


@dataclass(frozen=True)
class SpillPanel:
    """A bundled case study, ready to feed :class:`scspill.SCSPILLConfig`.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel with a 0/1 ``treated`` indicator column (1 for the treated
        unit in post-treatment periods).
    spatial_w : pd.Series
        Treated-to-control exposure weights, indexed by donor unit label.
        Raw (unnormalized); the estimator scales it to sum to one.
    spatial_W : pd.DataFrame
        Control-to-control spatial weights, indexed and columned by donor unit
        label. Raw (unnormalized); the estimator row-normalizes it.
    outcome, unitid, time, treat : str
        Column names in ``df``.
    covariates : tuple of str
        Covariate column names in ``df``.
    treated_unit : str
        Label of the treated unit.
    treatment_time : int
        First treated period.
    description : str
        One-paragraph provenance note.
    column_map : dict
        Mapping from original CSV headers to the column names in ``df``
        (empty when no renaming was applied).
    """

    df: pd.DataFrame
    spatial_w: pd.Series
    spatial_W: pd.DataFrame
    outcome: str
    unitid: str
    time: str
    treat: str
    covariates: tuple[str, ...]
    treated_unit: str
    treatment_time: int
    description: str
    column_map: dict[str, str] = field(default_factory=dict)

    def config_kwargs(self) -> dict[str, Any]:
        """Keyword arguments ready to splat into :class:`scspill.SCSPILLConfig`.

        Returns
        -------
        dict
            ``df``, ``outcome``, ``treat``, ``unitid``, ``time``,
            ``spatial_w``, ``spatial_W``, and ``covariates``.

        Examples
        --------
        ```python
        from scspill import SCSPILL
        from scspill.data import load_california

        panel = load_california()
        result = SCSPILL({**panel.config_kwargs(), "seed": 1}).fit()
        ```
        """
        return {
            "df": self.df,
            "outcome": self.outcome,
            "treat": self.treat,
            "unitid": self.unitid,
            "time": self.time,
            "spatial_w": self.spatial_w,
            "spatial_W": self.spatial_W,
            "covariates": list(self.covariates),
        }


def _read_csv(name: str) -> pd.DataFrame:
    """Read a bundled CSV via ``importlib.resources`` (works from a wheel)."""
    resource = files("scspill.data").joinpath("files", name)
    with resource.open("r", encoding="utf-8") as fh:
        return pd.read_csv(fh)


def _load_weights(
    w_file: str, W_file: str, label_col: str, donors: list
) -> tuple[pd.Series, pd.DataFrame]:
    """Load and label-align the ``(w, W)`` pair for a case study."""
    w_df = _read_csv(w_file)
    w = pd.Series(
        w_df.iloc[:, 1].to_numpy(dtype=float),
        index=pd.Index(w_df[label_col], name=label_col),
        name=w_df.columns[1],
    )
    W_df = _read_csv(W_file).set_index(label_col)
    W_df.index.name = label_col
    W_df = W_df.astype(float)

    missing_w = sorted(set(donors) - set(w.index))
    missing_W = sorted((set(donors) - set(W_df.index)) | (set(donors) - set(W_df.columns)))
    if missing_w or missing_W:  # pragma: no cover - guards against packaging corruption
        raise ScspillDataError(
            f"Bundled spatial weights are misaligned with the panel donors: "
            f"missing in w: {missing_w[:5]}; missing in W: {missing_W[:5]}."
        )
    # Reindex to the donor order (sorted unit labels minus the treated unit),
    # the same order the estimator's setup uses.
    return w.loc[donors], W_df.loc[donors, donors]


def load_california() -> SpillPanel:
    """Load the California Proposition 99 tobacco panel with rook-contiguity weights.

    The panel covers 39 U.S. states over 1970-2000 with per-capita cigarette
    sales (``cigsale``) as the outcome and the retail cigarette price
    (``retprice``) as a covariate. California passed Proposition 99 in 1988,
    so ``treated`` is 1 for California from 1988 onward (18 pre-treatment
    years, 13 post-treatment years, 38 donors). ``spatial_w`` is California's
    rook-contiguity row over the donors (only Nevada shares a land border);
    ``spatial_W`` is the binary rook adjacency among the donors.

    Returns
    -------
    SpillPanel
        The panel, spatial weights, and column metadata. Use
        :meth:`SpillPanel.config_kwargs` to feed it to
        :class:`scspill.SCSPILLConfig` directly.

    Examples
    --------
    ```python
    from scspill.data import load_california

    panel = load_california()
    panel.df.head()
    panel.spatial_w["Nevada"]   # 1.0 -- the only contiguous donor
    ```
    """
    df = _read_csv("california_panel.csv")
    treated_unit = "California"
    treatment_time = 1988
    df["treated"] = ((df["state"] == treated_unit) & (df["year"] >= treatment_time)).astype(int)
    donors = sorted(set(df["state"]) - {treated_unit})
    w, W = _load_weights("california_w_vector.csv", "california_W_matrix.csv", "state", donors)
    return SpillPanel(
        df=df,
        spatial_w=w,
        spatial_W=W,
        outcome="cigsale",
        unitid="state",
        time="year",
        treat="treated",
        covariates=("retprice",),
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        description=(
            "California Proposition 99 tobacco panel (Abadie, Diamond & "
            "Hainmueller 2010): 39 states, 1970-2000, per-capita cigarette "
            "sales, treatment in 1988. Spatial weights are rook contiguity "
            "from the 2024 TIGER/Line state shapefile, stored unnormalized. "
            "Source: scspill replication package, nonproprietary export."
        ),
    )


def load_sudan(raw_names: bool = False) -> SpillPanel:
    """Load the 2011 Sudan secession panel with bilateral-trade weights.

    The panel covers 34 African countries over 2000-2015 with GDP per capita
    (constant 2015 US$) as the outcome and six World Development Indicators
    series as covariates. South Sudan seceded in July 2011; the treated unit
    "Sudan" aggregates Sudan and South Sudan after the split, and ``treated``
    is 1 for Sudan from 2011 onward (11 pre-treatment years, 5 post-treatment
    years, 33 donors). ``spatial_w`` holds each donor's average pre-period
    bilateral trade with Sudan (IMF Direction of Trade Statistics) and
    ``spatial_W`` the average bilateral trade among donors -- both raw US$
    values, normalized inside the estimator.

    Parameters
    ----------
    raw_names : bool, default False
        When False, the R-mangled WDI column headers are renamed to clean
        snake_case (see the returned ``column_map``); when True, the original
        CSV headers are kept (useful for byte-level comparison against the R
        replication package).

    Returns
    -------
    SpillPanel
        The panel, spatial weights, and column metadata.

    Examples
    --------
    ```python
    from scspill.data import load_sudan

    panel = load_sudan()
    panel.covariates
    panel.spatial_w.nlargest(2)   # Egypt and Kenya trade most with Sudan
    ```
    """
    df = _read_csv("sudan_panel.csv")
    column_map: dict[str, str] = {}
    if not raw_names:
        column_map = dict(SUDAN_COLUMN_MAP)
        df = df.rename(columns=column_map)
    outcome = column_map.get(
        "GDP.per.capita..constant.2015.US..", "GDP.per.capita..constant.2015.US.."
    )
    covariates = tuple(
        column_map.get(c, c)
        for c in (
            "Exports.of.goods.and.services....of.GDP.",
            "Merchandise.trade....of.GDP.",
            "Trade....of.GDP.",
            "Access.to.clean.fuels.and.technologies.for.cooking....of.population.",
            "Inflation..consumer.prices..annual...",
            "Net.migration",
        )
    )
    treated_unit = "Sudan"
    treatment_time = 2011
    df["treated"] = ((df["country"] == treated_unit) & (df["year"] >= treatment_time)).astype(int)
    donors = sorted(set(df["country"]) - {treated_unit})
    w, W = _load_weights("sudan_w_vector.csv", "sudan_W_matrix.csv", "country", donors)
    return SpillPanel(
        df=df,
        spatial_w=w,
        spatial_W=W,
        outcome=outcome,
        unitid="country",
        time="year",
        treat="treated",
        covariates=covariates,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        description=(
            "2011 Sudan secession panel: 34 African countries, 2000-2015, GDP "
            "per capita (constant 2015 US$) plus six WDI covariates "
            "(percentage series stored as proportions). Sudan's post-2011 "
            "outcome aggregates Sudan and South Sudan. Spatial weights are "
            "average pre-period bilateral trade (IMF DOTS), stored as raw US$ "
            "values. Source: scspill replication package, nonproprietary export."
        ),
        column_map=column_map,
    )
