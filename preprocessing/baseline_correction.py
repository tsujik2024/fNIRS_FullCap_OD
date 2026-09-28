"""Baseline subtraction for fNIRS data.

baseline_subtraction() picks the window from, in order: a BaselineStart /
BaselineEnd marker pair, the span between the first two markers, recording
start to a lone marker, or the first 20 s. A baseline_df overrides all of that.
The level is a 10% trimmed mean so motion spikes in the window don't skew it.

find_baseline_window() is the stricter S1 -> W1 resolver used by the full-cap
pipeline (OD referencing and the concentration-stage baseline both call it, so
they can't drift apart).
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

logger = logging.getLogger(__name__)

TRIM_PROPORTION = 0.10
DEFAULT_BASELINE_S = 20.0
BASELINE_START_SAMPLE = 4  # skip the first few samples

_IGNORE_COLS = ["Sample number", "Event", "Time (s)", "Condition", "Subject"]


def find_baseline_window(events, fs, n, max_s=DEFAULT_BASELINE_S):
    """Return (start, end) samples of the pre-task baseline, or None if `events`
    has nothing usable. Callers pick their own fallback.

    Priority: S1->W1, S1->S2, S1 + max_s, then Task*Start / Baseline*End.
    S1 marks the START of the standing baseline in this protocol; only the
    Task*Start / Baseline*End markers mark its end. Treating bare S1 as an end
    marker silently references to the pre-S1 device warm-up instead (this was a
    real bug: every file was using [4, S1)).
    """
    if events is None or events.empty or "Event" not in events.columns:
        return None

    max_end = int(max_s * fs)
    min_span = max(1, int(fs))  # need at least ~1 s

    ev = events.copy()
    ev["Event"] = ev["Event"].astype(str).str.strip()
    ev["Sample number"] = pd.to_numeric(ev["Sample number"], errors="coerce")
    ev = ev.dropna(subset=["Sample number"]).sort_values("Sample number")

    def first(pattern, after=None):
        m = ev["Event"].str.match(pattern, case=False, na=False)
        if after is not None:
            m &= ev["Sample number"] > after
        return ev.loc[m, "Sample number"].iloc[0] if m.any() else None

    def window(a, b):
        a, b = int(max(0, a)), int(min(b, n))
        return (a, b) if b - a >= min_span else None

    s1 = first(r"^S1$")
    if s1 is not None:
        for end in (first(r"^W1$", s1), first(r"^S2$", s1), s1 + max_end):
            if end is not None and (w := window(s1, end)):
                return w

    marker = first(r"^Task.*Start$|^Baseline.*End$")
    if marker is not None:
        return window(BASELINE_START_SAMPLE, min(int(marker) - 1, max_end))
    return None


def baseline_subtraction(df: pd.DataFrame, events_df: pd.DataFrame,
                         baseline_df: pd.DataFrame = None, fs: float = None) -> pd.DataFrame:
    """Subtract each channel's baseline level. Metadata columns are left alone.

    baseline_df: if given, levels come from it instead of from `events_df`
    (same channel column names as `df`).
    fs: only matters for the marker fallbacks; estimated from 'Time (s)' or
    50 Hz if omitted.
    """
    out = df.copy()
    cols = [c for c in out.columns if c not in _IGNORE_COLS]

    if baseline_df is not None:
        for c in cols:
            out[c] = out[c] - trim_mean(baseline_df[c].dropna(), TRIM_PROPORTION)
        return out

    if fs is None:
        fs = _estimate_fs(df)

    start, end = _window_from_markers(events_df, fs)
    n = len(out)
    if not 0 <= start < n:
        logger.warning(f"Baseline start {start} out of bounds; using 4")
        start = 4
    end = min(end, n)
    if start >= end:
        logger.warning(f"Invalid baseline interval ({start}, {end}); using first 20 s")
        start, end = 4, min(int(20 * fs), n)

    duration = (end - start) / fs
    logger.info(f"Baseline: samples {start}-{end} ({duration:.1f}s)")

    for c in cols:
        seg = out[c].iloc[start:end].dropna()
        out[c] = out[c] - (trim_mean(seg, TRIM_PROPORTION) if len(seg) else np.nan)

    out.attrs.update(baseline_start=start, baseline_end=end, baseline_duration_s=duration)
    return out


def _estimate_fs(df):
    if "Time (s)" in df.columns and len(df) > 1:
        dt = df["Time (s)"].iloc[1] - df["Time (s)"].iloc[0]
        if dt > 0:
            return 1 / dt
    return 50.0


def _window_from_markers(events_df, fs):
    ev = events_df.copy()
    ev["Event"] = (ev["Event"].astype(str).str.strip().str.upper()
                   .str.replace(r"\s+", "", regex=True))
    ev = ev[ev["Event"].str.contains(r"[A-Z0-9]", na=False)]
    ev = ev.sort_values("Sample number").reset_index(drop=True)
    names = ev["Event"].to_numpy()

    if "BASELINESTART" in names and "BASELINEEND" in names:
        start = ev.loc[ev["Event"] == "BASELINESTART", "Sample number"].iloc[0]
        end = ev.loc[ev["Event"] == "BASELINEEND", "Sample number"].iloc[0]

    elif len(ev) >= 2:  # e.g. S1 -> W1, S1 -> S2
        start, end = ev["Sample number"].iloc[0], ev["Sample number"].iloc[1]
        dur = (end - start) / fs
        if dur < 2:
            logger.warning(f"Baseline between first two markers is only {dur:.1f}s; check markers")
        elif dur > 60:
            logger.warning(f"Baseline between first two markers is {dur:.1f}s; capping at 30s")
            end = start + int(30 * fs)

    elif len(ev) == 1:  # recording start -> the one marker
        start, end = 4, ev["Sample number"].iloc[0]
        if (end - start) / fs < 5:
            logger.warning("Only one marker and baseline < 5s; using first 20 s")
            start, end = 4, int(20 * fs)

    else:
        logger.warning("No event markers; using first 20 s as baseline")
        start, end = 4, int(20 * fs)

    return int(start), int(end)
