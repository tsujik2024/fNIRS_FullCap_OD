"""Peak Spectral Power (PSP), QT-NIRS style (Pollonini et al. 2016, Biomed. Opt. Express 7:5104).

Same windowing as SCI (see sci.py): bandpass to the cardiac band, cut into windows. In
each window, cross-correlate the two z-scored traces and take the peak of a
Hamming-windowed periodogram of that cross-correlation. A channel's PSP is the median
across windows.

Scaling matters: the periodogram here is a POWER spectrum (scaling='spectrum'), so a
unit-amplitude sinusoid peaks at 0.5. That's the scale the paper's own numbers use (0.5
ideal, ~0.16 for a clean channel, 0.1 recommended threshold) and what QT-NIRS's MATLAB
code computes (periodogram(..., 'power')). scipy's other option, a power spectral
DENSITY ('density'), reads roughly 6-8x higher for the same data, so a 0.1 threshold on
that scale would be far more permissive than the published one -- don't switch scaling
without also rescaling the threshold. The score also depends on window length, so keep
WINDOW_S fixed when comparing against a threshold measured at a different window size.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

from channel_quality.sci import CARDIAC_BAND, WINDOW_S, _filtered_windows, zscore_or_zero

_SCALINGS = ("spectrum", "density")


def _window_peak_power(w1: np.ndarray, w2: np.ndarray, fs: float, scaling: str, floor1: float, floor2: float) -> float:
    n1, n2 = zscore_or_zero(w1, floor1), zscore_or_zero(w2, floor2)
    if not n1.any() or not n2.any():
        return 0.0
    xc = np.correlate(n1, n2, mode="full") / len(n1)
    _, power = periodogram(xc, fs=fs, window="hamming", nfft=len(xc), scaling=scaling)
    return float(power.max())


def peak_spectral_power(od_wl1, od_wl2, fs: float,
                        cardiac_band: tuple[float, float] = CARDIAC_BAND,
                        window_s: float = WINDOW_S,
                        filter_order: int = 4,
                        scaling: str = "spectrum") -> float:
    """Median per-window PSP. NaN if unscoreable."""
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {_SCALINGS}, got {scaling!r}")
    result = _filtered_windows(od_wl1, od_wl2, fs, cardiac_band, window_s, filter_order)
    if result is None:
        return float("nan")
    windows, floor1, floor2 = result
    return float(np.median([_window_peak_power(w1, w2, fs, scaling, floor1, floor2) for w1, w2 in windows]))
