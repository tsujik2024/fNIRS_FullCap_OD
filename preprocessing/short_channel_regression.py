"""Short-channel regression (SCR): strip the superficial (scalp blood flow)
component from long-channel concentration columns.

Shared by both pipelines. This only does the regression; picking WHICH short
channel(s) go with a long channel is the caller's job. Pass in `short_data`
already holding exactly the reference column(s) you want: one short, several
to be averaged here, or a pre-averaged column all behave the same.

Chromophore type is detected by substring so every naming convention works:
    "CH# HbO" / "CH# HbR", "CH#_oxy" / "CH#_deoxy", "CH# O2Hb" / "CH# HHb"

Refs: Scholkmann 2014; Gagnon 2014; Brigadoi 2014.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OXY_KEYS = ("HbO", "O2Hb", "_oxy")
_DEOXY_KEYS = ("HbR", "HHb", "_deoxy")


def scr_regression(long_data: pd.DataFrame, short_data: pd.DataFrame,
                   center: bool = False) -> pd.DataFrame:
    """Regress each long column on the mean of the matching short column(s) and subtract.

    Columns with no short reference of their type, or a reference with zero
    energy, come back unchanged (with a warning). Non-finite samples are
    ignored when fitting beta.

    center=False keeps the original no-intercept fit. If the inputs still carry
    offsets or slow drift (i.e. SCR runs before the bandpass), that biases beta;
    center=True fits on the mean-removed regressor, which is the safer choice.
    """
    out = long_data.copy()

    for label, keys in (("oxy", _OXY_KEYS), ("deoxy", _DEOXY_KEYS)):
        long_cols = [c for c in long_data.columns if any(k in str(c) for k in keys)]
        short_cols = [c for c in short_data.columns if any(k in str(c) for k in keys)]
        if not long_cols:
            continue
        if not short_cols:
            logger.warning(f"SCR: no {label} short reference in {list(short_data.columns)}; "
                           f"{long_cols} left uncorrected")
            continue

        x = short_data[short_cols].mean(axis=1).to_numpy(dtype=float)
        ok_x = np.isfinite(x)
        if center:
            x = x - x[ok_x].mean()

        for col in long_cols:
            y = long_data[col].to_numpy(dtype=float)
            ok = ok_x & np.isfinite(y)  # one NaN sample used to NaN the whole column via the dot products
            denom = x[ok] @ x[ok]
            if denom == 0:
                logger.warning(f"SCR: {label} reference {short_cols} is flat/empty; {col} left uncorrected")
                continue
            out[col] = y - (x[ok] @ y[ok] / denom) * x

    return out
