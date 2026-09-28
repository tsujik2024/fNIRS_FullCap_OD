"""Zero-phase Butterworth bandpass for fNIRS concentration data.

Replaced a windowed-sinc FIR that passed ~96% of DC at order=1000,
Wn=[0.01, 0.1], fs=50: its transition band (~3.3*fs/ntaps = 0.165 Hz) is wider
than the whole passband, so no FIR that fits in a 120 s record can form a clean
edge at 0.01 Hz. Don't bring it back. SOS + sosfiltfilt because the b/a form
goes unstable above order 2-4.

The 0.01 Hz corner is long relative to a ~2 min record, so the default padding
leaves visible edge distortion (worst at the END of the record). Pass
padlen=len(df) - 1 to cut most of it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

_SKIP = {"Event", "Sample number"}


def butterworth_bandpass(df: pd.DataFrame, order: int, Wn: list, fs: float,
                         padlen: int | None = None) -> pd.DataFrame:
    """Bandpass every numeric column except metadata.

    `order` is the pole count (2-4 is typical here), not a tap count.
    Columns containing NaN/Inf can't be filtered; they come back all-NaN so
    downstream means skip them instead of averaging in a fake flat zero line.
    """
    out = df.copy()
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in _SKIP]
    if not cols:
        return out

    nyq = fs / 2
    if any(w <= 0 or w >= nyq for w in Wn):
        raise ValueError(f"Wn={Wn} must be strictly between 0 and Nyquist ({nyq} Hz)")

    sos = butter(order, Wn, btype="bandpass", fs=fs, output="sos")

    min_len = 3 * (2 * len(sos) + 1)
    if len(df) <= min_len:
        logger.warning(f"Bandpass: {len(df)} samples is too short for order={order} "
                       f"(needs > {min_len}); returning data unfiltered")
        return out

    data = df[cols].to_numpy(dtype=float)
    bad = ~np.isfinite(data).all(axis=0)
    if bad.any():
        bad_cols = [c for c, b in zip(cols, bad) if b]
        logger.warning(f"Bandpass: non-finite values in {bad_cols}; setting those columns to NaN")
        out[bad_cols] = np.nan

    good_cols = [c for c, b in zip(cols, bad) if not b]
    if good_cols:
        out[good_cols] = sosfiltfilt(sos, data[:, ~bad], axis=0, padlen=padlen)
    return out
