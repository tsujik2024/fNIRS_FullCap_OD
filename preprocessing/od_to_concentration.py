"""Optical density -> HbO/HHb concentration change via the modified Beer-Lambert law.

Output columns: "CH{n} HbO", "CH{n} HHb" in uM.
Extinction coefficients: Prahl / OMLC, https://omlc.org/spectra/hemoglobin/summary.html

OxySoft exports ABSOLUTE optical density, but MBLL solves for a CHANGE in
concentration, so each channel's OD is referenced to a baseline first
(dOD = OD - baseline mean). Without that you get absolute values in the tens
of uM instead of task-evoked changes. Baseline window:
    1. from `events` via find_baseline_window (S1->W1, S1->S2, S1+20s, or a
       Task*Start / Baseline*End marker)
    2. otherwise the whole-record per-channel mean
reference_baseline=False skips referencing and returns absolute concentration.
"""

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from preprocessing.baseline_correction import DEFAULT_BASELINE_S, find_baseline_window

logger = logging.getLogger(__name__)

DEFAULT_DPF = 6.0

# Prahl table, cm^-1/M -> divided by 1000 below to get cm^-1/mM.
# lambda: (HbO2, Hb). 757/759/839 are interpolated from their neighbours.
_PRAHL_RAW = {
    750: (518.0, 1405.24),
    752: (533.2, 1515.32),
    754: (548.4, 1541.76),
    756: (562.0, 1560.48),
    757: (568.0, 1560.48),
    758: (574.0, 1560.48),
    759: (580.0, 1554.50),
    760: (586.0, 1548.52),
    762: (598.0, 1508.44),
    764: (610.0, 1459.56),
    836: (1001.2, 692.64),
    838: (1011.6, 692.48),
    839: (1016.8, 692.42),
    840: (1022.0, 692.36),
    842: (1032.4, 692.20),
    844: (1042.8, 691.96),
    846: (1050.0, 691.76),
    848: (1054.0, 691.52),
    850: (1058.0, 691.32),
    852: (1062.0, 691.08),
}

EXT_COEFFS = {wl: {"HbO": o / 1000.0, "HbR": r / 1000.0} for wl, (o, r) in _PRAHL_RAW.items()}


@lru_cache(maxsize=None)
def get_extinction_coefficient(wavelength: int) -> dict:
    """{"HbO", "HbR"} in cm^-1/mM; nearest tabulated wavelength if there's no exact match."""
    if wavelength in EXT_COEFFS:
        return EXT_COEFFS[wavelength]
    nearest = min(EXT_COEFFS, key=lambda w: abs(w - wavelength))
    logger.warning(f"{wavelength} nm not in extinction table, using {nearest} nm")
    return EXT_COEFFS[nearest]


def mbll_dual_wavelength(od1, od2, wl1, wl2, dpf, distance_cm):
    """Solve dOD_i = (eps_HbO_i * dHbO + eps_HbR_i * dHbR) * d * DPF for two
    wavelengths (Cramer's rule). od1/od2 must be baseline-referenced.
    Returns (HbO, HbR) changes in uM.
    """
    e1, e2 = get_extinction_coefficient(wl1), get_extinction_coefficient(wl2)
    det = e1["HbO"] * e2["HbR"] - e2["HbO"] * e1["HbR"]
    if abs(det) < 1e-12:
        raise ValueError(f"Singular extinction matrix for {wl1}/{wl2} nm")

    scale = 1000.0 / (det * dpf * distance_cm)  # mM -> uM
    hbo = (e2["HbR"] * od1 - e1["HbR"] * od2) * scale
    hbr = (e1["HbO"] * od2 - e2["HbO"] * od1) * scale
    return hbo, hbr


def convert_od_to_concentration(df_od, channel_map, metadata=None, events=None, fs=50.0,
                                baseline_duration=DEFAULT_BASELINE_S, reference_baseline=True):
    """OD -> delta-concentration for every channel in `channel_map`.

    channel_map: from loaders.py; per channel needs distance_mm, columns
    ({wavelength: [col, ...]}) and optionally dpf (default 6.0).
    events: 'Sample number'/'Event' in the SAME sample coordinates as df_od
    (i.e. after any initial crop).
    metadata: unused, kept so existing callers don't break.
    Channels with missing geometry/columns or != 2 wavelengths are skipped.
    """
    window = find_baseline_window(events, fs, len(df_od), baseline_duration) if reference_baseline else None

    def ref(sig):
        if not reference_baseline:
            return 0.0
        if window:
            r = np.nanmean(sig[window[0]:window[1]])
            if np.isfinite(r):
                return r
        return np.nanmean(sig)

    out, skipped = {}, []
    for ch, info in channel_map.items():
        dist_mm, wl_cols = info.get("distance_mm"), info.get("columns")
        if dist_mm is None or not wl_cols or len(wl_cols) != 2:
            skipped.append(ch)
            continue

        wl1, wl2 = sorted(wl_cols)
        cols1, cols2 = wl_cols[wl1], wl_cols[wl2]
        if not cols1 or not cols2 or cols1[0] not in df_od.columns or cols2[0] not in df_od.columns:
            skipped.append(ch)
            continue

        od1 = df_od[cols1[0]].to_numpy(dtype=float)
        od2 = df_od[cols2[0]].to_numpy(dtype=float)
        hbo, hbr = mbll_dual_wavelength(od1 - ref(od1), od2 - ref(od2), wl1, wl2,
                                        info.get("dpf") or DEFAULT_DPF, dist_mm / 10.0)
        out[f"CH{ch} HbO"] = hbo
        out[f"CH{ch} HHb"] = hbr

    if skipped:
        logger.warning(f"OD->conc skipped channels (missing distance/columns/2 wavelengths): {skipped}")
    if not out:
        logger.error("OD->conc: no channels converted")
    return pd.DataFrame(out, index=df_od.index)
