"""Bundled case-study spillover panels for scspill.

Ships the two empirical applications of the ``sar`` model's source article
(Sakaguchi & Tagawa 2026) and the Proposition 99 panel of the ``cd`` model's
source article (Cao & Dowd, arXiv:1902.07343v3). Every loader returns a
:class:`SpillPanel` whose ``config_kwargs()`` feeds
:class:`scspill.SCSPILLConfig` directly. The Cao--Dowd panel has a different
sample and intervention date from the ``sar`` California example.

* :func:`load_california` -- the California Proposition 99 tobacco panel
  (Abadie, Diamond & Hainmueller 2010): 39 states, 1970-2000, treatment in
  1988; rook-contiguity spatial weights.
* :func:`load_sudan` -- the 2011 Sudan secession panel: 34 African countries,
  2000-2015, treatment in 2011; average bilateral-trade spatial weights.
* :func:`load_california_cd` -- the Cao--Dowd Proposition 99 panel: 50
  states plus DC (or 41 of them), 1970-2000, treatment from 1989, with the
  paper's set of potentially exposed states and no spatial weights.

The ``sar`` CSV files are verbatim copies of that replication package's
nonproprietary data exports, and the Cao--Dowd files of the authors'
repository (jcao0/synthetic-control-spillover at commit 86004020). Both
``sar`` weight objects are stored raw --
row-normalization of ``W`` and sum-normalization of ``w`` happen inside the
estimator, exactly as in the R package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import pandas as pd

from ..exceptions import ScspillDataError

__all__ = ["SpillPanel", "load_california", "load_california_cd", "load_sudan"]


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

    A panel carries what the models it was published with need: spatial
    weights for ``sar``, a set of potentially exposed units for ``cd``.

    Attributes
    ----------
    df : pd.DataFrame
        Long panel with a 0/1 ``treated`` indicator column (1 for the treated
        unit in post-treatment periods).
    spatial_w : pd.Series or None
        Treated-to-control exposure weights, indexed by donor unit label.
        Raw (unnormalized); the estimator scales it to sum to one. ``None``
        for a panel without spatial weights.
    spatial_W : pd.DataFrame or None
        Control-to-control spatial weights, indexed and columned by donor unit
        label. Raw (unnormalized); the estimator row-normalizes it. ``None``
        for a panel without spatial weights.
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
    affected_units : tuple
        Control units that may receive a spillover, as published for the
        ``cd`` model; empty when none were published.
    """

    df: pd.DataFrame
    spatial_w: pd.Series | None
    spatial_W: pd.DataFrame | None
    outcome: str
    unitid: str
    time: str
    treat: str
    covariates: tuple[str, ...]
    treated_unit: str
    treatment_time: int
    description: str
    column_map: dict[str, str] = field(default_factory=dict)
    affected_units: tuple[str, ...] = ()

    @property
    def has_spatial_weights(self) -> bool:
        """Whether the panel carries the spatial weights ``sar`` needs."""
        return self.spatial_w is not None and self.spatial_W is not None

    def config_kwargs(self, method: str | None = None) -> dict[str, Any]:
        """Keyword arguments ready to splat into :class:`scspill.SCSPILLConfig`.

        Parameters
        ----------
        method : {"sar", "cd"}, optional
            The model to configure. By default ``"sar"`` when the panel has
            spatial weights and ``"cd"`` otherwise.

        Returns
        -------
        dict
            The column names plus, for ``sar``, ``spatial_w``, ``spatial_W``
            and ``covariates`` (``method`` is left at its default); for
            ``cd``, ``method="cd"`` and ``affected_units``.

        Raises
        ------
        ScspillDataError
            When ``method="sar"`` is asked of a panel without spatial weights,
            or ``method`` is unknown.

        Examples
        --------
        ```python
        from scspill import SCSPILL
        from scspill.data import load_california, load_california_cd

        sar = SCSPILL({**load_california().config_kwargs(), "seed": 1}).fit()
        cd = SCSPILL(load_california_cd(51).config_kwargs()).fit()
        ```
        """
        if method is None:
            method = "sar" if self.has_spatial_weights else "cd"
        columns = {
            "df": self.df,
            "outcome": self.outcome,
            "treat": self.treat,
            "unitid": self.unitid,
            "time": self.time,
        }
        if method == "sar":
            if not self.has_spatial_weights:
                raise ScspillDataError(
                    "This panel has no spatial weights, which method='sar' requires."
                )
            return {
                **columns,
                "spatial_w": self.spatial_w,
                "spatial_W": self.spatial_W,
                "covariates": list(self.covariates),
            }
        if method == "cd":
            return {
                "method": "cd",
                **columns,
                "affected_units": list(self.affected_units),
            }
        raise ScspillDataError(f"Unknown method {method!r}; expected 'sar' or 'cd'.")


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


def load_california_cd(specification: int = 51) -> SpillPanel:
    """Load Cao and Dowd's Proposition 99 panel: 1970--2000, treated from 1989.

    The panel holds per-capita cigarette consumption (``cigs``) for the 50
    U.S. states plus the District of Columbia, as used in Cao & Dowd
    (arXiv:1902.07343v3, Section 6). California is treated from 1989, the
    first full year of Proposition 99 (19 pre-treatment years, 12
    post-treatment years). It is a different sample from
    :func:`load_california`, which follows Abadie et al. (2010).

    ``affected_units`` are the states the paper allows to receive a
    spillover:

    * ``51`` keeps every state and DC; the 13 exposed units are the twelve
      that passed their own tobacco-control policies (Arizona and Oregon
      among them) plus Nevada, so all three of California's neighbours are
      included.
    * ``41`` drops the ten policy states that do not border California,
      leaving Arizona, Nevada and Oregon as the exposed controls.

    The panel has no spatial weights, so ``config_kwargs()`` configures
    ``method="cd"``.

    Parameters
    ----------
    specification : {51, 41}, default 51
        Number of units in the sample.

    Returns
    -------
    SpillPanel

    Examples
    --------
    ```python
    from scspill import SCSPILL
    from scspill.data import load_california_cd

    panel = load_california_cd(51)
    result = SCSPILL(panel.config_kwargs()).fit()
    result.effects_panel.loc[1990, ["CA", "NV"]]
    ```
    """
    if specification not in (41, 51):
        raise ScspillDataError("Cao--Dowd California specification must be 41 or 51.")
    root = files("scspill.data").joinpath("files", "cao_dowd")
    with root.joinpath("cigs_consumption.csv").open("r", encoding="utf-8") as fh:
        df = pd.read_csv(fh)
    with root.joinpath("state_list.csv").open("r", encoding="utf-8-sig") as fh:
        flags = pd.read_csv(fh)
    policy = set(flags.loc[flags["missing12"] == 1, "state"])
    neighbors = set(flags.loc[flags["neighbor"] == 1, "state"])
    if specification == 41:
        df = df.loc[~df["state"].isin(policy - neighbors)].reset_index(drop=True)
        affected = tuple(sorted(neighbors))
    else:
        affected = tuple(sorted(policy | neighbors))
    treated_unit = "CA"
    treatment_time = 1989
    df["treated"] = ((df["state"] == treated_unit) & (df["year"] >= treatment_time)).astype(int)
    return SpillPanel(
        df=df,
        spatial_w=None,
        spatial_W=None,
        outcome="cigs",
        unitid="state",
        time="year",
        treat="treated",
        covariates=(),
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        description=(
            f"Cao & Dowd Proposition 99 panel ({specification} units): per-capita "
            "cigarette consumption, 1970-2000, California treated from 1989, "
            "with the paper's set of potentially exposed states. Source: "
            "jcao0/synthetic-control-spillover at commit 86004020."
        ),
        affected_units=affected,
    )
