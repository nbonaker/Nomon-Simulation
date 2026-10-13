"""Offline data helpers preserved from the Nomon evaluation modules."""
from __future__ import annotations


from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

USER_DATA_ROOT = REPO_ROOT / "Nomon_User_Data" / "OSF Data"

TEXT_DATA_ROOT = USER_DATA_ROOT / "text_entry_task"

def normalize_phrase_order(phrase_df: pd.DataFrame) -> pd.DataFrame:
    return phrase_df.sort_values(["Session Num", "Phrase Num"]).reset_index(drop=True)


import numpy as np

import pandas as pd

CLOCK_PERIOD_COL = "Clock Period (s)"

CLICK_OFFSET_COL = "Click Time Relative (s)"

def assign_clock_regimes(
    click_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    absolute_tolerance_s: float = 0.02,
    relative_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Assign rows to their nearest compatible training-derived regime."""

    if regimes_df.empty:
        raise ValueError("At least one clock regime is required")
    result = click_df.copy()
    centers = regimes_df["period_center_s"].to_numpy(dtype=float)
    regime_ids = regimes_df["regime_id"].astype(str).to_numpy()
    assignments = []
    for period in pd.to_numeric(result[CLOCK_PERIOD_COL], errors="coerce"):
        if pd.isna(period):
            raise ValueError("Clock-period rows must be numeric before regime assignment")
        differences = np.abs(centers - float(period))
        nearest = int(np.argmin(differences))
        tolerance = max(
            absolute_tolerance_s,
            abs(float(centers[nearest])) * relative_tolerance,
        )
        if float(differences[nearest]) > tolerance:
            raise ValueError(
                f"Clock period {float(period):.6f}s has no compatible training regime; "
                f"nearest is {float(centers[nearest]):.6f}s"
            )
        assignments.append(regime_ids[nearest])
    result["Clock Regime"] = assignments
    return result


import pandas as pd

from User_Simulation.evaluation.synthetic_profiles import DEAD_TIME_COL

def zero_active_dead_time(click_df: pd.DataFrame) -> pd.DataFrame:
    """Set synthetic active click dead time to zero while preserving calibration NaNs."""

    result = click_df.copy()
    active_mask = result["Session Num"].notna()
    result.loc[active_mask, DEAD_TIME_COL] = 0.0
    return result
