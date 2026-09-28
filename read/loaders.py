"""OxySoft full-cap OD .txt reader.

Returns OD data plus a per-channel map; all channel metadata (Rx/Tx, distance,
short/long, region) comes from channel_config.CHANNEL_INFO.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from read.channel_config import (
    CHANNEL_INFO,
    LONG_CHANNEL_LIST,
    SHORT_CHANNEL_LIST,
    is_long_channel,
    is_short_channel,
)

logger = logging.getLogger(__name__)

# aliases kept for older callers
SHORT_CHANNELS = SHORT_CHANNEL_LIST
LONG_CHANNELS = LONG_CHANNEL_LIST
CHANNEL_DISTANCES = {ch: i.distance_mm for ch, i in CHANNEL_INFO.items()}
CHANNEL_REGION = {ch: i.region for ch, i in CHANNEL_INFO.items()}

N_OD_COLS = 52
DPF = 6.0


def read_txt_file(file_path: str | Path) -> Dict:
    """-> {'metadata', 'data', 'events', 'channel_map'}"""
    file_path = Path(file_path)
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    meta = _read_header(lines)
    wavelengths = _read_wavelengths(lines)
    df_od, channel_map = _label_channels(_read_data(lines, file_path), wavelengths)

    meta.update({
        "file": str(file_path),
        "wavelength_map": wavelengths,
        "short_channels": list(SHORT_CHANNEL_LIST),
        "long_channels": list(LONG_CHANNEL_LIST),
        "channel_distances_mm": dict(CHANNEL_DISTANCES),
    })
    return {"metadata": meta, "data": df_od, "events": _extract_events(df_od), "channel_map": channel_map}


def _read_header(lines: List[str]) -> Dict:
    meta: Dict = {"sample_rate": None, "export_rate": None, "device_ids": [],
                  "num_receivers": None, "num_sources": None}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if "Datafile sample rate:" in line:
            meta["sample_rate"] = _safe_int(parts[-1])
        elif "Export sample rate" in line:
            meta["export_rate"] = _safe_int(parts[-1])
        elif "# Receivers:" in line:
            meta["num_receivers"] = _safe_int(parts[-1])
        elif "# Light sources:" in line:
            meta["num_sources"] = _safe_int(parts[-1])
        elif "Device ids:" in line:
            meta["device_ids"] = parts[2:]
    return meta


def _read_wavelengths(lines: List[str]) -> Dict[Tuple[int, int], int]:
    """'Light source wavelengths' block -> {(device, light_idx): nm}."""
    out: Dict[Tuple[int, int], int] = {}
    inside = False
    for line in lines:
        if "Light source wavelengths:" in line:
            inside = True
        elif not inside:
            continue
        elif "Export sample rate" in line or "Selected time span" in line:
            break
        else:
            p = line.split()
            if len(p) >= 3 and p[0].isdigit() and p[1].isdigit():
                try:
                    out[(int(p[0]), int(p[1]))] = int(p[2].replace("nm", ""))
                except ValueError:
                    pass
    return out


def _data_delimiter(lines: List[str], probe: int = 20) -> str:
    """"\t" if the block is tab-delimited (the OxySoft standard), else " "."""
    for line in lines[:probe]:
        if "\t" in line:
            return "\t"
    return " "


def _read_data(lines: List[str], file_path: Path) -> pd.DataFrame:
    """Sample number + OD columns + Event. Trailing ADC columns are dropped.

    OxySoft's export is TAB-delimited and the Event column is usually blank.
    Splitting on whitespace (`line.split()`) collapses the two tabs around a
    blank Event into nothing, so that row comes out one field shorter than a
    row with an event on it -- not because the file is malformed, but because
    split() can't see an empty field between two tabs. Split on "\t" so a
    blank Event still counts as a field, and pad/reconcile any row that's
    still a different width (e.g. the exporter drops the final trailing tab)
    instead of raising.
    """
    delim = _data_delimiter(lines)

    raw_rows: List[List[str]] = []
    started = False
    for line in lines:
        if not line.strip():
            continue
        p = line.split(delim) if delim == "\t" else line.strip().split()
        try:
            float(p[0])
        except (ValueError, IndexError):
            continue
        if not started:
            if len(p) <= 50:  # header-ish numeric line, not the data table yet
                continue
            started = True
        raw_rows.append(p)

    if not raw_rows:
        raise ValueError(f"No data table in {file_path}")

    # canonical width = the modal row length, not just the first row's --
    # a short first row (blank Event, dropped trailing tab) shouldn't make
    # every other row look wrong.
    width_counts: Dict[int, int] = {}
    for r in raw_rows:
        width_counts[len(r)] = width_counts.get(len(r), 0) + 1
    n = max(width_counts, key=width_counts.get)

    n_off = sum(1 for r in raw_rows if len(r) != n)
    if n_off:
        logger.debug(f"{file_path.name}: {n_off}/{len(raw_rows)} data rows padded/trimmed to "
                     f"the modal width ({n} fields) -- normal for a blank-Event export")

    rows = []
    for r in raw_rows:
        if len(r) < n:
            r = r + [""] * (n - len(r))
        elif len(r) > n:
            r = r[:n - 1] + [delim.join(r[n - 1:])]
        rows.append(r)

    n_od = min(N_OD_COLS, n - 2)
    df = pd.DataFrame(rows)[[0, *range(1, n_od + 1), n - 1]]
    df.columns = ["Sample number", *[f"ODcol_{i}" for i in range(1, n_od + 1)], "Event"]
    df["Sample number"] = pd.to_numeric(df["Sample number"], errors="coerce").fillna(0).astype(int)
    df["Event"] = df["Event"].astype(str).str.strip().replace({"nan": ""})
    od = [c for c in df.columns if c.startswith("ODcol_")]
    df[od] = df[od].apply(pd.to_numeric, errors="coerce")
    return df


def _label_channels(df_raw: pd.DataFrame, wavelengths: Dict[Tuple[int, int], int]) -> Tuple[pd.DataFrame, Dict]:
    """Rename ODcol_N -> D{d}_R{r}_T{t}_WL{nm} and build the channel map."""
    rename: Dict[str, str] = {}
    channel_map: Dict[int, Dict] = {}

    for ch, info in CHANNEL_INFO.items():
        dev = 1 if info.device == "OctaMon" else 2
        # each Tx emits two consecutive light indices
        wl1 = wavelengths.get((dev, 2 * info.tx_id - 1))
        wl2 = wavelengths.get((dev, 2 * info.tx_id))
        if wl1 is None or wl2 is None:
            logger.warning(f"CH{ch}: missing wavelength for D{dev}-T{info.tx_id}")
            continue

        src = [f"ODcol_{c - 1}" for c in info.file_columns]
        if not all(c in df_raw.columns for c in src):
            logger.warning(f"CH{ch}: source columns {src} not in data")
            continue

        new = [f"D{dev}_R{info.rx_id}_T{info.tx_id}_WL{wl}" for wl in (wl1, wl2)]
        rename.update(zip(src, new))
        channel_map[ch] = {
            "device": dev,
            "receiver": info.rx_id,
            "tx": info.tx_id,
            "distance_mm": info.distance_mm,
            "dpf": DPF,
            "short_channel": info.channel_type == "SHORT",
            "region": info.region,
            "hemisphere": info.hemisphere,
            "wavelength_pairs": [(wl1, wl2)],
            "columns": {wl1: [new[0]], wl2: [new[1]]},
        }
    return df_raw.rename(columns=rename), channel_map


def _extract_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per event onset. A label held over consecutive rows is one event;
    the same label after a blank row is a new one. Duration runs to the next onset."""
    if "Event" not in df.columns or "Sample number" not in df.columns:
        return pd.DataFrame(columns=["Sample number", "Event", "Duration"])

    onsets, prev = [], ""
    for samp, ev in zip(df["Sample number"], df["Event"]):
        if ev and ev != prev:
            onsets.append((int(samp), ev))
        prev = ev

    end = int(df["Sample number"].iloc[-1]) + 1 if onsets else 0
    nxt = [s for s, _ in onsets[1:]] + [end]
    return pd.DataFrame([{"Sample number": s, "Event": ev, "Duration": n - s}
                         for (s, ev), n in zip(onsets, nxt)])


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def get_short_channels() -> List[int]:
    return list(SHORT_CHANNEL_LIST)


def get_long_channels() -> List[int]:
    return list(LONG_CHANNEL_LIST)


def get_channel_distance(ch_idx: int) -> Optional[float]:
    info = CHANNEL_INFO.get(ch_idx)
    return info.distance_mm if info else None


__all__ = [
    "read_txt_file", "SHORT_CHANNELS", "LONG_CHANNELS", "CHANNEL_DISTANCES", "CHANNEL_REGION",
    "is_short_channel", "is_long_channel", "get_short_channels", "get_long_channels", "get_channel_distance",
]
