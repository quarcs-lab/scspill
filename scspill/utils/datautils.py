"""Panel-data validation utilities.

The subset of ``mlsynth.utils.datautils`` that scspill needs: the strong-balance
check every ``fit()`` runs before touching the data. Estimator-specific panel
reshaping lives in ``scspill.utils.scspill_helpers.setup``.
"""

from __future__ import annotations

import pandas as pd

from ..exceptions import ScspillDataError


def balance(df: pd.DataFrame, unit_id_column_name: str, time_period_column_name: str) -> None:
    """Check that the panel is strongly balanced.

    A strongly balanced panel means every unit has an observation for every
    time period, and there are no duplicate unit-time observations.

    Parameters
    ----------
    df : pd.DataFrame
        The input panel data. Must contain the columns named by
        ``unit_id_column_name`` and ``time_period_column_name``.
    unit_id_column_name : str
        The name of the column in ``df`` that identifies the units.
    time_period_column_name : str
        The name of the column in ``df`` that identifies the time periods.

    Returns
    -------
    None
        This function does not return a value but raises an error if the
        panel is not strongly balanced or contains duplicates.

    Raises
    ------
    ScspillDataError
        If duplicate unit-time observations are found, or if the panel is not
        strongly balanced (i.e., not all units have observations for all time
        periods).
    """
    if df.duplicated([unit_id_column_name, time_period_column_name]).any():
        raise ScspillDataError(
            "Duplicate observations found. Ensure each combination of unit and time is unique."
        )

    total_unique_time_periods = df[time_period_column_name].nunique()
    observations_per_unit = df.groupby(unit_id_column_name)[time_period_column_name].nunique()
    if not (observations_per_unit == total_unique_time_periods).all():
        raise ScspillDataError(
            "The panel is not strongly balanced. Not all units have observations "
            "for all unique time periods in the dataset."
        )
