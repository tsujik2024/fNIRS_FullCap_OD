# loaders.py -------------------------------------------------------------
"""
Full-cap fNIRS TXT *OD-domain* reader for the fnirs_FullCap_2025 pipeline.

This loader:
    ✓ reads optical density (OD) data from OxySoft full-cap TXT exports
    ✓ assigns Device / Receiver / Source / Wavelength to each OD column
    ✓ creates human-readable column names (D1_R1_S1_WL850)
    ✓ drops ADC channels automatically
    ✓ extracts event markers
    ✓ classifies short (10mm) vs long channels based on XML template
    ✓ builds a channel_map for MBLL pairing downstream
    ✓ returns OD (not HbO/HbR)

CORRECTED VERSION:
    - Uses actual channel distances from XML templates:
      * OctaMon 2x3 channel + 2x1 SSC (ID 87)
      * Brite24 custom template Mancini lab (ID 155)
    - Short channels are CH 4, 8, 23, 24, 25, 26
    - OctaMon (PFC): 6 long + 2 short channels
    - Brite24 (Motor): 14 long + 4 short channels
    - Total: 20 long + 6 short = 26 channels

Output dictionary:
{
    "metadata": {...},
    "data": df_od,          # wide-format OD columns
    "events": events_df,    # onset/duration
    "channel_map": {...},   # for MBLL, distances, wavelengths
}
"""

from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

# -------------------------------------------------------------------------
# CHANNEL CONFIGURATION FROM XML TEMPLATE - CORRECTED
# -------------------------------------------------------------------------
# Based on XML template "Brite24 custom template Mancini lab" (ID 155)
# and OctaMon 2x3 channel + 2x1 SSC (ID 87)
#
# Channel numbering convention:
#   - OctaMon (Device 1): CH 1-8 (PFC)
#   - Brite24 (Device 2): CH 9-26 (Motor/Sensory/Visual cortex)
#
# From Brite24 Mancini XML template:
#   - SubTemplates 1-4: DistanceScale=30mm (LONG) → combinations 1-14 → CH 9-22
#   - SubTemplates 5-8: DistanceScale=10mm (SHORT) → combinations 15-18 → CH 23-26
#
# From OctaMon XML template:
#   - Group 1,3: DistanceScale=35mm (LONG) → combinations 1-3, 5-7 → CH 1-3, 5-7
#   - Group 2,4: DistanceScale=10mm (SHORT) → combinations 4, 8 → CH 4, 8

# Actual channel distances from XML template (in mm)
CHANNEL_DISTANCES = {
    # OctaMon (Device 1) - PFC
    1: 35, 2: 35, 3: 35,  # Rx1 long channels (PFC_R)
    4: 10,                 # Rx1 short channel (PFC_R)
    5: 35, 6: 35, 7: 35,  # Rx2 long channels (PFC_L)
    8: 10,                 # Rx2 short channel (PFC_L)

    # Brite24 (Device 2) - Motor cortex - ALL LONG CHANNELS FIRST
    9: 30,                 # Rx1 (V1_L) - combination 1
    10: 30, 11: 30,        # Rx2 (S1_L) - combinations 2-3
    12: 30, 13: 30,        # Rx3 (M1_L) - combinations 4-5
    14: 30, 15: 30,        # Rx4 (SMA_L) - combinations 6-7
    16: 30, 17: 30,        # Rx5 (SMA_R) - combinations 8-9
    18: 30, 19: 30,        # Rx6 (M1_R) - combinations 10-11
    20: 30, 21: 30,        # Rx7 (S1_R) - combinations 12-13
    22: 30,                # Rx8 (V1_R) - combination 14

    # Brite24 SHORT CHANNELS (combinations 15-18 from XML)
    23: 10,                # M1_L short (Rx3-Tx5) - combination 15
    24: 10,                # SMA_R short (Rx5-Tx6) - combination 16
    25: 10,                # SMA_L short (Rx4-Tx5) - combination 17
    26: 10,                # M1_R short (Rx6-Tx6) - combination 18
}

# Short channel list (from XML template)
# OctaMon: CH 4, 8 (PFC shorts)
# Brite24: CH 23-26 (Motor shorts - combinations 15-18)
SHORT_CHANNELS = [4, 8, 23, 24, 25, 26]  # 6 total short channels

# Long channel list (all channels except shorts)
LONG_CHANNELS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # 20 total


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def read_txt_file(file_path: str | Path) -> Dict:
    """
    Read a full-cap OxySoft TXT file containing OD data.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # read all lines (including blank lines to preserve structure)
    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    # Process the file
    metadata = _extract_header_metadata(lines)
    wavelength_map = _extract_wavelength_table(lines)
    distances_mm_from_header = _extract_optode_distances(lines)

    # Parse the data section
    df_raw = _read_od_data_space_separated(lines, metadata, file_path)

    # clean sample + event columns
    df_raw["Sample number"] = pd.to_numeric(df_raw["Sample number"],
                                            errors="coerce").astype("int64")
    df_raw["Event"] = df_raw["Event"].astype(str).replace({"nan": ""})

    events_df = _extract_events(df_raw)

    # Get the OD column names from the DataFrame
    # EXCLUDE Sample number and Event
    od_column_names = [col for col in df_raw.columns
                       if col not in ["Sample number", "Event"] and col.startswith("ODcol_")]

    # Debug: print what we found
    logger.debug(f"Found {len(od_column_names)} OD columns: {od_column_names[:5]}...")

    # build OD column mapping
    od_cols, column_mapping = _assign_column_names(od_column_names,
                                                   metadata,
                                                   wavelength_map)

    # Debug: print what we're renaming to
    logger.debug(f"Renaming to {len(od_cols)} columns: {od_cols[:5]}...")

    # Check if renaming worked
    if not od_cols:
        logger.error(f"No OD columns generated for {file_path}")
        channel_map = {}
        df_od = df_raw
    else:
        # Create rename dictionary
        rename_dict = {}
        for old_name, new_name in zip(od_column_names, od_cols):
            rename_dict[old_name] = new_name

        # Rename the columns
        df_od = df_raw.rename(columns=rename_dict)

        # Ensure we have Sample number and Event columns
        if "Sample number" not in df_od.columns:
            df_od["Sample number"] = df_raw["Sample number"]
        if "Event" not in df_od.columns:
            df_od["Event"] = df_raw["Event"]

        # build channel map for MBLL using CORRECTED distances from XML template
        channel_map = _build_channel_map(column_mapping, distances_mm_from_header)

        # Log channel info
        for ch_idx, info in channel_map.items():
            logger.debug(
                f"CH{ch_idx}: Receiver={info.get('receiver')}, "
                f"Distance={info.get('distance_mm')}mm, "
                f"Short={info.get('short_channel', False)}")

    # prepare metadata for return
    # Include both header distances (simplified) and actual distances (from XML)
    metadata_out = {
        "file": str(file_path),
        "sample_rate": metadata["sample_rate"],
        "wavelength_map": wavelength_map,
        "distances_mm_header": distances_mm_from_header,  # From TXT header (simplified)
        "distances_mm_actual": CHANNEL_DISTANCES,  # From XML template (accurate)
        "Optode distance (mm)": [CHANNEL_DISTANCES.get(ch, 30) for ch in range(1, 27)],
        "num_receivers": metadata["num_receivers"],
        "num_sources": metadata["num_sources"],
        "device_ids": metadata["device_ids"],
        "short_channels": SHORT_CHANNELS,
        "long_channels": LONG_CHANNELS,
    }

    return {
        "metadata": metadata_out,
        "data": df_od,
        "events": events_df,
        "channel_map": channel_map,
    }


# -------------------------------------------------------------------------
# Header extraction helpers (updated for space-separated)
# -------------------------------------------------------------------------

def _extract_header_metadata(lines: List[str]) -> Dict:
    """Extract sample rate, device ids, counts, etc."""
    meta = {
        "sample_rate": None,
        "device_ids": [],
        "num_receivers": None,
        "num_sources": None,
    }

    for line in lines:
        if "Datafile sample rate:" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    meta["sample_rate"] = int(float(parts[-1]))
                except (ValueError, IndexError):
                    pass
        if "# Receivers:" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    meta["num_receivers"] = int(parts[-1])
                except (ValueError, IndexError):
                    pass
        if "# Light sources:" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    meta["num_sources"] = int(parts[-1])
                except (ValueError, IndexError):
                    pass
        if "Device ids:" in line:
            parts = line.split()
            for i in range(1, len(parts)):
                if parts[i].strip():
                    meta["device_ids"].append(parts[i].strip())

    return meta


def _extract_wavelength_table(lines: List[str]) -> Dict[Tuple[int, int], int]:
    """
    Build mapping: (device, source_index) -> wavelength_nm.
    """
    wl_map = {}

    inside = False
    for line in lines:
        if "Light source wavelengths:" in line:
            inside = True
            continue
        if inside:
            parts = line.split()
            # Skip header lines or empty lines
            if len(parts) < 4:
                continue

            # Check if this is a wavelength table row
            # It should have: device, index, wavelength, "nm"
            if parts[0].isdigit() and parts[1].isdigit():
                try:
                    device = int(parts[0])
                    idx = int(parts[1])

                    # The wavelength might have "nm" attached or not
                    wl_str = parts[2].replace("nm", "").strip()
                    wl = int(wl_str)

                    wl_map[(device, idx)] = wl
                except (ValueError, IndexError) as e:
                    # Skip malformed rows
                    continue
            else:
                # Check if we've reached the end of the table
                if "Export sample rate" in line or "Selected time span" in line:
                    break
                if not (len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit()):
                    continue

    return wl_map


def _extract_optode_distances(lines: List[str]) -> Dict[int, float]:
    """
    Parse 'Optode distance (mm):' table → return mapping: receiver_index → mm.

    NOTE: These are simplified per-receiver distances from the TXT header.
    The actual per-channel distances come from the XML template and are
    stored in CHANNEL_DISTANCES constant.
    """
    distances = []
    for line in lines:
        if "Optode distance" in line:
            parts = line.split()
            for tok in parts[1:]:
                try:
                    distances.append(float(tok))
                except Exception:
                    pass
            break

    # Receivers numbered 1..len(distances)
    return {rx + 1: distances[rx] for rx in range(len(distances))}


# -------------------------------------------------------------------------
# OD data reader for SPACE-SEPARATED data
# -------------------------------------------------------------------------
def _read_od_data_space_separated(lines: List[str], metadata: Dict, file_path: Path) -> pd.DataFrame:
    """
    Extract OD domain values from the DATA region of the TXT file.
    Handles space-separated data (not tab-separated).
    """
    # Find where the actual data starts
    data_start_line = None
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        # Check if this is a data row (starts with 0 or 1)
        if parts[0] in ['0', '1'] and len(parts) > 50:
            data_start_line = i
            break

    if data_start_line is None:
        # Try to find any line that starts with a number and has many columns
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 50:
                continue

            try:
                float(parts[0])
                data_start_line = i
                break
            except ValueError:
                continue

    if data_start_line is None:
        raise ValueError(f"Cannot locate data table in {file_path}")

    # Now collect all data rows starting from data_start_line
    data_rows = []
    for i in range(data_start_line, len(lines)):
        line = lines[i].strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        # Check if this is a data row (first element is a number)
        try:
            float(parts[0])
            data_rows.append(parts)
        except ValueError:
            continue

    if not data_rows:
        raise ValueError(f"No data rows found in {file_path}")

    # Determine column count from data
    n_columns = len(data_rows[0])

    # Create column names dynamically based on actual column count
    col_names = ["Sample number"]

    # Data structure: 52 OD columns (26 channels × 2 wavelengths), then ADC columns, then Event
    # Expected: 1 (sample) + 52 (OD) + N (ADC) + 1 (Event) = 54+ columns
    if n_columns >= 59:
        od_count = 52
        adc_count = n_columns - od_count - 2  # -2 for Sample number and Event
    elif n_columns >= 54:
        od_count = 52
        adc_count = n_columns - od_count - 2
    else:
        # Fallback: assume most columns are OD
        od_count = n_columns - 2  # Leave room for Sample number and Event
        adc_count = 0
        logger.warning(f"Unexpected column count {n_columns} in {file_path}, assuming {od_count} OD columns")

    # Create OD column names (1 to od_count)
    for i in range(1, od_count + 1):
        col_names.append(f"ODcol_{i}")

    # Create ADC column names
    for i in range(1, adc_count + 1):
        col_names.append(f"ADC_{i}")

    # Event column
    col_names.append("Event")

    # Verify column count
    if len(col_names) != n_columns:
        logger.warning(f"Column mismatch in {file_path}: expected {len(col_names)}, got {n_columns}")
        # Adjust column names to match actual data
        if n_columns > len(col_names):
            # Add extra columns
            for i in range(len(col_names), n_columns):
                col_names.insert(-1, f"extra_{i}")  # Insert before Event
        else:
            # Trim column names
            col_names = col_names[:n_columns-1] + ["Event"]

    # Create DataFrame
    try:
        df = pd.DataFrame(data_rows, columns=col_names)
    except ValueError as e:
        logger.error(f"DataFrame creation failed for {file_path}: {e}")
        from collections import Counter
        length_counter = Counter(len(row) for row in data_rows)
        most_common_length = length_counter.most_common(1)[0][0]
        data_rows = [row for row in data_rows if len(row) == most_common_length]
        col_names = [f"col_{i}" for i in range(most_common_length)]
        col_names[0] = "Sample number"
        col_names[-1] = "Event"
        df = pd.DataFrame(data_rows, columns=col_names)

    # Convert Sample number to integer
    df["Sample number"] = pd.to_numeric(df["Sample number"], errors='coerce').fillna(0).astype(int)

    # Convert other columns to numeric
    for col in df.columns:
        if col not in ["Sample number", "Event"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop ADC columns - we only want OD columns
    adc_cols = [col for col in df.columns if col.startswith("ADC_") or col.startswith("extra_")]
    if adc_cols:
        df = df.drop(columns=adc_cols)

    return df


# -------------------------------------------------------------------------
# Assign readable OD column names
# -------------------------------------------------------------------------
def _assign_column_names(od_cols: List[str],
                         metadata: Dict,
                         wavelength_map: Dict[Tuple[int, int], int]):
    """
    Rename OD columns using legend mapping from the file header.
    Columns are paired sequentially: 1&2, 3&4, 5&6, etc.

    The legend_map matches the TXT file structure exactly:
    - 52 OD columns total (26 channels × 2 wavelengths)
    - Device 1 (OctaMon): Rx1 uses Tx1-8, Rx2 uses Tx9-16
    - Device 2 (Brite24): Rx1-8 use Tx1-20 (dual wavelength per physical Tx)
    """
    column_mapping = {}
    new_names = []

    # Complete legend map matching your file structure (52 OD columns)
    # Format: column_index -> (device, receiver, source_index)
    legend_map = {
        # Device 1, Receiver 1 (OctaMon Rx1 - PFC Right)
        1: (1, 1, 1), 2: (1, 1, 2), 3: (1, 1, 3), 4: (1, 1, 4),
        5: (1, 1, 5), 6: (1, 1, 6), 7: (1, 1, 7), 8: (1, 1, 8),
        # Device 1, Receiver 2 (OctaMon Rx2 - PFC Left)
        9: (1, 2, 9), 10: (1, 2, 10), 11: (1, 2, 11), 12: (1, 2, 12),
        13: (1, 2, 13), 14: (1, 2, 14), 15: (1, 2, 15), 16: (1, 2, 16),
        # Device 2, Receiver 1 (Brite24 Rx1 - V1_L)
        17: (2, 1, 1), 18: (2, 1, 2),
        # Device 2, Receiver 2 (Brite24 Rx2 - S1_L)
        19: (2, 2, 5), 20: (2, 2, 6), 21: (2, 2, 7), 22: (2, 2, 8),
        # Device 2, Receiver 3 (Brite24 Rx3 - M1_L)
        23: (2, 3, 3), 24: (2, 3, 4), 25: (2, 3, 5), 26: (2, 3, 6),
        27: (2, 3, 9), 28: (2, 3, 10),
        # Device 2, Receiver 4 (Brite24 Rx4 - SMA_L)
        29: (2, 4, 5), 30: (2, 4, 6), 31: (2, 4, 7), 32: (2, 4, 8),
        33: (2, 4, 9), 34: (2, 4, 10),
        # Device 2, Receiver 5 (Brite24 Rx5 - SMA_R)
        35: (2, 5, 11), 36: (2, 5, 12), 37: (2, 5, 13), 38: (2, 5, 14),
        39: (2, 5, 15), 40: (2, 5, 16),
        # Device 2, Receiver 6 (Brite24 Rx6 - M1_R)
        41: (2, 6, 11), 42: (2, 6, 12), 43: (2, 6, 15), 44: (2, 6, 16),
        45: (2, 6, 17), 46: (2, 6, 18),
        # Device 2, Receiver 7 (Brite24 Rx7 - S1_R)
        47: (2, 7, 15), 48: (2, 7, 16), 49: (2, 7, 17), 50: (2, 7, 18),
        # Device 2, Receiver 8 (Brite24 Rx8 - V1_R)
        51: (2, 8, 19), 52: (2, 8, 20),
    }

    for i, col in enumerate(od_cols, start=1):
        if i > 52:
            new_names.append(col)
            column_mapping[col] = {
                "original_col": col,
                "device": None,
                "receiver": None,
                "source": None,
                "wavelength": None,
                "pair_index": None,
            }
            continue

        if i not in legend_map:
            logger.warning(f"Column index {i} not in legend_map for {col}")
            new_names.append(col)
            continue

        device, rx, src_idx = legend_map[i]
        wavelength = wavelength_map.get((device, src_idx), None)

        if wavelength is not None:
            # Determine which channel pair this belongs to (1-26)
            # Columns are paired: 1&2→ch1, 3&4→ch2, etc.
            channel_pair = (i + 1) // 2

            newname = f"D{device}_R{rx}_S{src_idx}_WL{wavelength}"
            new_names.append(newname)
            column_mapping[newname] = {
                "original_col": col,
                "device": device,
                "receiver": rx,
                "source": src_idx,
                "wavelength": wavelength,
                "channel_pair": channel_pair,
            }
        else:
            logger.warning(f"No wavelength found for device {device}, source {src_idx}")
            newname = f"D{device}_R{rx}_S{src_idx}_UNKNOWN"
            new_names.append(newname)
            column_mapping[newname] = {
                "original_col": col,
                "device": device,
                "receiver": rx,
                "source": src_idx,
                "wavelength": None,
                "channel_pair": None,
            }

    return new_names, column_mapping


def _build_channel_map(column_mapping: Dict[str, Dict],
                       distances_mm_from_header: Dict[int, float]):
    """
    Build channel structures for MBLL.
    Columns are paired sequentially: every 2 consecutive columns form one channel.
    Creates 26 channels from 52 OD columns.

    IMPORTANT: Uses CHANNEL_DISTANCES from XML template for accurate short/long
    classification, NOT the simplified per-receiver distances from the TXT header.
    """
    # Group by channel_pair number
    channel_groups = {}
    for col_name, info in column_mapping.items():
        pair_idx = info.get("channel_pair")
        if pair_idx is None:
            continue

        if pair_idx not in channel_groups:
            channel_groups[pair_idx] = []

        channel_groups[pair_idx].append({
            "column": col_name,
            "device": info["device"],
            "receiver": info["receiver"],
            "source": info["source"],
            "wavelength": info["wavelength"],
        })

    logger.info(f"Found {len(channel_groups)} channel pairs")

    channel_map = {}

    for ch_idx in sorted(channel_groups.keys()):
        cols = channel_groups[ch_idx]

        if len(cols) != 2:
            logger.warning(f"Channel {ch_idx}: expected 2 columns, got {len(cols)}")
            continue

        # Get the two wavelengths
        wl1 = cols[0]["wavelength"]
        wl2 = cols[1]["wavelength"]

        if wl1 is None or wl2 is None:
            logger.warning(f"Channel {ch_idx}: missing wavelength info")
            continue

        # Use info from first column for device/receiver
        device = cols[0]["device"]
        rx = cols[0]["receiver"]

        # Map to global receiver index (for reference to header distances)
        if device == 1:
            global_rx = rx
        else:
            global_rx = 2 + rx

        # Get the ACTUAL distance from XML template (not simplified header)
        distance = CHANNEL_DISTANCES.get(ch_idx, None)

        # Fallback to header distance if channel not in our map
        if distance is None:
            distance = distances_mm_from_header.get(global_rx, 30)
            logger.warning(f"Channel {ch_idx} not in CHANNEL_DISTANCES, using header value: {distance}mm")

        # Determine if short channel based on XML template
        is_short = ch_idx in SHORT_CHANNELS

        # Create channel entry
        channel_map[ch_idx] = {
            "device": device,
            "receiver": rx,
            "global_receiver": global_rx,
            "sources": [cols[0]["source"], cols[1]["source"]],
            "distance_mm": distance,
            "dpf": 6.0,
            "wavelength_pairs": [(wl1, wl2)],
            "columns": {
                wl1: [cols[0]["column"]],
                wl2: [cols[1]["column"]]
            },
            "short_channel": is_short,
        }

    logger.info(f"Created {len(channel_map)} channels from column mapping")

    # Log short vs long channels
    short_chs = [idx for idx, info in channel_map.items() if info.get("short_channel")]
    long_chs = [idx for idx, info in channel_map.items() if not info.get("short_channel")]
    logger.info(f"Short channels (10mm): CH {short_chs}")
    logger.info(f"Long channels (30-35mm): CH {long_chs}")

    # Log device breakdown
    octa_short = [idx for idx in short_chs if channel_map[idx]["device"] == 1]
    octa_long = [idx for idx in long_chs if channel_map[idx]["device"] == 1]
    brite_short = [idx for idx in short_chs if channel_map[idx]["device"] == 2]
    brite_long = [idx for idx in long_chs if channel_map[idx]["device"] == 2]

    logger.info(
        f"OctaMon: {len(octa_long)} long + {len(octa_short)} short = {len(octa_long) + len(octa_short)} channels")
    logger.info(
        f"Brite24: {len(brite_long)} long + {len(brite_short)} short = {len(brite_long) + len(brite_short)} channels")

    return channel_map


# -------------------------------------------------------------------------
# Event extractor
# -------------------------------------------------------------------------

def _extract_events(df: pd.DataFrame) -> pd.DataFrame:
    """Translate Event column into onset/duration list."""
    events = []
    current = None
    onset = None

    for samp, ev in zip(df["Sample number"], df["Event"]):
        if not ev:
            continue
        if ev != current:
            if current is not None:
                events.append({
                    "Sample number": onset,
                    "Event": current,
                    "Duration": samp - onset,
                })
            current = ev
            onset = samp

    if current is not None:
        events.append({
            "Sample number": onset,
            "Event": current,
            "Duration": df["Sample number"].iloc[-1] - onset + 1,
        })

    return pd.DataFrame(events)


# -------------------------------------------------------------------------
# Utility functions for external use
# -------------------------------------------------------------------------

def get_short_channels():
    """Return list of short channel indices."""
    return SHORT_CHANNELS.copy()


def get_long_channels():
    """Return list of long channel indices."""
    return LONG_CHANNELS.copy()


def get_channel_distance(ch_idx: int) -> float:
    """Get the source-detector distance for a specific channel."""
    return CHANNEL_DISTANCES.get(ch_idx, None)


def is_short_channel(ch_idx: int) -> bool:
    """Check if a channel is a short separation channel."""
    return ch_idx in SHORT_CHANNELS


def is_long_channel(ch_idx: int) -> bool:
    """Check if a channel is a long (brain-measuring) channel."""
    return ch_idx in LONG_CHANNELS