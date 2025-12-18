"""
Implements short channel correction (short channel regression) to remove
superficial components from long-channel fNIRS signals.

References:
    - Gagnon et al., 2014
    - Brigadoi et al., 2014
"""

import pandas as pd
import numpy as np


def scr_regression(long_data: pd.DataFrame, short_data: pd.DataFrame) -> pd.DataFrame:
    """Expects columns in CH# HbO/HHb format"""
    corrected = long_data.copy()

    # Get HbO and HHb columns separately
    o2hb_cols = [col for col in long_data.columns if "HbO" in col]
    hhb_cols = [col for col in long_data.columns if "HHb" in col]

    # Process oxygenated and deoxygenated separately
    for hb_type, cols in [("HbO", o2hb_cols), ("HHb", hhb_cols)]:
        short_cols = [col for col in short_data.columns if hb_type in col]
        if not short_cols:
            continue

        short_mean = short_data[short_cols].mean(axis=1)

        for long_col in cols:
            Y = long_data[long_col]
            X = short_mean

            beta = np.dot(X, Y) / np.dot(X, X) if np.dot(X, X) != 0 else 0
            corrected[long_col] = Y - beta * X

    return corrected