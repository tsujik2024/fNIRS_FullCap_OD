#!/usr/bin/env python3
"""
Diagnostic script for fNIRS channel mapping verification.

This script analyzes your data file and loader to verify:
1. Channel assignments are correct
2. Short channels are properly detected
3. Wavelength pairing is accurate

Usage:
    python diagnose_channel_mapping.py <path_to_txt_file>

Or run with the sample data embedded below.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# ============================================================================
# CONFIGURATION - Based on your data file header
# ============================================================================

# From your file: "Optode distance (mm): 35.00 35.00 30.00 30.00 30.00 30.00 10.00 10.00 10.00 10.00"
# These correspond to GLOBAL receiver indices 1-10
DISTANCES_MM = {
    1: 35.0,  # Device 1, Rx 1
    2: 35.0,  # Device 1, Rx 2
    3: 30.0,  # Device 2, Rx 1
    4: 30.0,  # Device 2, Rx 2
    5: 30.0,  # Device 2, Rx 3
    6: 30.0,  # Device 2, Rx 4
    7: 10.0,  # Device 2, Rx 5 - SHORT
    8: 10.0,  # Device 2, Rx 6 - SHORT
    9: 10.0,  # Device 2, Rx 7 - SHORT
    10: 10.0,  # Device 2, Rx 8 - SHORT
}

# Wavelength map from your file header
# Format: (device, source_index) -> wavelength_nm
WAVELENGTH_MAP = {
    # Device 1 - OctaMon (alternating 850/760)
    (1, 1): 850, (1, 2): 760,
    (1, 3): 850, (1, 4): 760,
    (1, 5): 850, (1, 6): 760,
    (1, 7): 850, (1, 8): 760,
    (1, 9): 850, (1, 10): 760,
    (1, 11): 850, (1, 12): 760,
    (1, 13): 850, (1, 14): 760,
    (1, 15): 850, (1, 16): 760,
    # Device 2 - Brite24 (alternating ~758/~840)
    (2, 1): 758, (2, 2): 840,
    (2, 3): 757, (2, 4): 839,
    (2, 5): 758, (2, 6): 839,
    (2, 7): 759, (2, 8): 839,
    (2, 9): 758, (2, 10): 848,
    (2, 11): 758, (2, 12): 848,
    (2, 13): 757, (2, 14): 839,
    (2, 15): 759, (2, 16): 839,
    (2, 17): 758, (2, 18): 840,
    (2, 19): 752, (2, 20): 839,
}

# Legend from your file - maps OD column index to (Device, Receiver, Source)
# Column 1 = Sample number, Column 2-53 = OD data, Column 54-58 = ADC, Column 59 = Event
LEGEND_MAP = {
    # ODcol index -> (Device, Receiver, Source)
    # Device 1, Receiver 1: Sources 1-8 (columns 2-9 in file, ODcol 1-8)
    1: (1, 1, 1), 2: (1, 1, 2), 3: (1, 1, 3), 4: (1, 1, 4),
    5: (1, 1, 5), 6: (1, 1, 6), 7: (1, 1, 7), 8: (1, 1, 8),
    # Device 1, Receiver 2: Sources 9-16 (columns 10-17 in file, ODcol 9-16)
    9: (1, 2, 9), 10: (1, 2, 10), 11: (1, 2, 11), 12: (1, 2, 12),
    13: (1, 2, 13), 14: (1, 2, 14), 15: (1, 2, 15), 16: (1, 2, 16),
    # Device 2, Receiver 1: Sources 1-2 (columns 18-19, ODcol 17-18)
    17: (2, 1, 1), 18: (2, 1, 2),
    # Device 2, Receiver 2: Sources 5-8 (columns 20-23, ODcol 19-22)
    19: (2, 2, 5), 20: (2, 2, 6), 21: (2, 2, 7), 22: (2, 2, 8),
    # Device 2, Receiver 3: Sources 3-4, 5-6, 9-10 (columns 24-29, ODcol 23-28)
    23: (2, 3, 3), 24: (2, 3, 4), 25: (2, 3, 5), 26: (2, 3, 6),
    27: (2, 3, 9), 28: (2, 3, 10),
    # Device 2, Receiver 4: Sources 5-10 (columns 30-35, ODcol 29-34)
    29: (2, 4, 5), 30: (2, 4, 6), 31: (2, 4, 7), 32: (2, 4, 8),
    33: (2, 4, 9), 34: (2, 4, 10),
    # Device 2, Receiver 5: Sources 11-16 (columns 36-41, ODcol 35-40)
    35: (2, 5, 11), 36: (2, 5, 12), 37: (2, 5, 13), 38: (2, 5, 14),
    39: (2, 5, 15), 40: (2, 5, 16),
    # Device 2, Receiver 6: Sources 11-12, 15-18 (columns 42-47, ODcol 41-46)
    41: (2, 6, 11), 42: (2, 6, 12), 43: (2, 6, 15), 44: (2, 6, 16),
    45: (2, 6, 17), 46: (2, 6, 18),
    # Device 2, Receiver 7: Sources 15-18 (columns 48-51, ODcol 47-50)
    47: (2, 7, 15), 48: (2, 7, 16), 49: (2, 7, 17), 50: (2, 7, 18),
    # Device 2, Receiver 8: Sources 19-20 (columns 52-53, ODcol 51-52)
    51: (2, 8, 19), 52: (2, 8, 20),
}


def get_global_receiver(device: int, receiver: int) -> int:
    """
    Convert device-local receiver to global receiver index.
    Device 1 has receivers 1-2 (global 1-2)
    Device 2 has receivers 1-8 (global 3-10)
    """
    if device == 1:
        return receiver
    else:
        return 2 + receiver


def analyze_channel_mapping():
    """Analyze the channel mapping and identify issues."""

    print("=" * 80)
    print("fNIRS CHANNEL MAPPING DIAGNOSTIC")
    print("=" * 80)
    print()

    # =========================================================================
    # Step 1: Analyze column-to-channel mapping
    # =========================================================================
    print("STEP 1: COLUMN ANALYSIS")
    print("-" * 40)

    columns_info = []
    for od_idx in sorted(LEGEND_MAP.keys()):
        device, rx, src = LEGEND_MAP[od_idx]
        wavelength = WAVELENGTH_MAP.get((device, src), None)
        global_rx = get_global_receiver(device, rx)
        distance = DISTANCES_MM.get(global_rx, None)
        is_short = distance is not None and distance <= 10.1

        columns_info.append({
            'od_col': od_idx,
            'device': device,
            'receiver': rx,
            'global_rx': global_rx,
            'source': src,
            'wavelength': wavelength,
            'distance_mm': distance,
            'short_channel': is_short,
        })

    print(f"Total OD columns: {len(columns_info)}")
    print()

    # =========================================================================
    # Step 2: Pair columns into channels (every 2 consecutive columns)
    # =========================================================================
    print("STEP 2: CHANNEL PAIRING (consecutive columns)")
    print("-" * 40)

    channels = {}
    for i in range(0, len(columns_info), 2):
        ch_num = (i // 2) + 1
        col1 = columns_info[i]
        col2 = columns_info[i + 1] if i + 1 < len(columns_info) else None

        if col2 is None:
            print(f"WARNING: Channel {ch_num} has only one column!")
            continue

        # Verify both columns have same receiver
        if col1['global_rx'] != col2['global_rx']:
            print(f"WARNING: Channel {ch_num} - columns have different receivers!")
            print(
                f"  Col {col1['od_col']}: Device {col1['device']}, Rx {col1['receiver']} (global {col1['global_rx']})")
            print(
                f"  Col {col2['od_col']}: Device {col2['device']}, Rx {col2['receiver']} (global {col2['global_rx']})")

        channels[ch_num] = {
            'columns': [col1['od_col'], col2['od_col']],
            'device': col1['device'],
            'receiver': col1['receiver'],
            'global_rx': col1['global_rx'],
            'sources': [col1['source'], col2['source']],
            'wavelengths': [col1['wavelength'], col2['wavelength']],
            'distance_mm': col1['distance_mm'],
            'short_channel': col1['short_channel'],
        }

    print(f"Total channels: {len(channels)}")
    print()

    # =========================================================================
    # Step 3: Display all channels in a table
    # =========================================================================
    print("STEP 3: CHANNEL DETAILS")
    print("-" * 100)
    print(
        f"{'CH':<4} {'Cols':<10} {'Dev':<4} {'Rx':<4} {'GRx':<4} {'Sources':<12} {'Wavelengths':<16} {'Dist':<8} {'Short':<6}")
    print("-" * 100)

    for ch_num in sorted(channels.keys()):
        ch = channels[ch_num]
        cols_str = f"{ch['columns'][0]}-{ch['columns'][1]}"
        src_str = f"{ch['sources'][0]}, {ch['sources'][1]}"
        wl_str = f"{ch['wavelengths'][0]}, {ch['wavelengths'][1]}"
        dist_str = f"{ch['distance_mm']:.1f}mm" if ch['distance_mm'] else "N/A"
        short_str = "YES" if ch['short_channel'] else ""

        print(
            f"{ch_num:<4} {cols_str:<10} {ch['device']:<4} {ch['receiver']:<4} {ch['global_rx']:<4} {src_str:<12} {wl_str:<16} {dist_str:<8} {short_str:<6}")

    print()

    # =========================================================================
    # Step 4: Summary statistics
    # =========================================================================
    print("STEP 4: SUMMARY")
    print("-" * 40)

    short_channels = [ch_num for ch_num, ch in channels.items() if ch['short_channel']]
    long_channels = [ch_num for ch_num, ch in channels.items() if not ch['short_channel']]

    print(f"Long channels (>10mm):  {len(long_channels)} - CH {long_channels}")
    print(f"Short channels (≤10mm): {len(short_channels)} - CH {short_channels}")
    print()

    # Group by device
    device_channels = defaultdict(list)
    for ch_num, ch in channels.items():
        device_channels[ch['device']].append(ch_num)

    for dev in sorted(device_channels.keys()):
        print(f"Device {dev}: {len(device_channels[dev])} channels - CH {device_channels[dev]}")
    print()

    # Group by global receiver
    rx_channels = defaultdict(list)
    for ch_num, ch in channels.items():
        rx_channels[ch['global_rx']].append(ch_num)

    print("Channels by Global Receiver:")
    for rx in sorted(rx_channels.keys()):
        dist = DISTANCES_MM[rx]
        short_marker = " [SHORT]" if dist <= 10.1 else ""
        print(f"  Global Rx {rx} ({dist}mm){short_marker}: CH {rx_channels[rx]}")
    print()

    # =========================================================================
    # Step 5: Wavelength pairing validation
    # =========================================================================
    print("STEP 5: WAVELENGTH PAIRING VALIDATION")
    print("-" * 40)

    issues = []
    for ch_num, ch in channels.items():
        wl1, wl2 = ch['wavelengths']
        if wl1 is None or wl2 is None:
            issues.append(f"CH {ch_num}: Missing wavelength - got {wl1}, {wl2}")
        elif abs(wl1 - wl2) < 50:
            issues.append(f"CH {ch_num}: Wavelengths too similar - {wl1}nm, {wl2}nm")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("✓ All channel wavelength pairs look valid")
    print()

    # =========================================================================
    # Step 6: Compare with your loader's legend_map
    # =========================================================================
    print("STEP 6: LOADER VALIDATION")
    print("-" * 40)

    # Your loader's legend_map from loaders.py
    your_legend_map = {
        1: (1, 1, 1), 2: (1, 1, 2), 3: (1, 1, 3), 4: (1, 1, 4),
        5: (1, 1, 5), 6: (1, 1, 6), 7: (1, 1, 7), 8: (1, 1, 8),
        9: (1, 2, 9), 10: (1, 2, 10), 11: (1, 2, 11), 12: (1, 2, 12),
        13: (1, 2, 13), 14: (1, 2, 14), 15: (1, 2, 15), 16: (1, 2, 16),
        17: (2, 1, 1), 18: (2, 1, 2),
        19: (2, 2, 5), 20: (2, 2, 6), 21: (2, 2, 7), 22: (2, 2, 8),
        23: (2, 3, 3), 24: (2, 3, 4), 25: (2, 3, 5), 26: (2, 3, 6),
        27: (2, 3, 9), 28: (2, 3, 10),
        29: (2, 4, 5), 30: (2, 4, 6), 31: (2, 4, 7), 32: (2, 4, 8),
        33: (2, 4, 9), 34: (2, 4, 10),
        35: (2, 5, 11), 36: (2, 5, 12), 37: (2, 5, 13), 38: (2, 5, 14),
        39: (2, 5, 15), 40: (2, 5, 16),
        41: (2, 6, 11), 42: (2, 6, 12), 43: (2, 6, 15), 44: (2, 6, 16),
        45: (2, 6, 17), 46: (2, 6, 18),
        47: (2, 7, 15), 48: (2, 7, 16), 49: (2, 7, 17), 50: (2, 7, 18),
        51: (2, 8, 19), 52: (2, 8, 20),
    }

    mismatches = []
    for idx in LEGEND_MAP:
        if idx in your_legend_map:
            if LEGEND_MAP[idx] != your_legend_map[idx]:
                mismatches.append(f"Index {idx}: Expected {LEGEND_MAP[idx]}, Your loader has {your_legend_map[idx]}")
        else:
            mismatches.append(f"Index {idx}: Missing from your loader's legend_map")

    if mismatches:
        print("MISMATCHES WITH YOUR LOADER:")
        for m in mismatches:
            print(f"  ✗ {m}")
    else:
        print("✓ Your loader's legend_map matches the file structure perfectly!")
    print()

    # =========================================================================
    # Step 7: Expected short channels based on distances
    # =========================================================================
    print("STEP 7: SHORT CHANNEL VERIFICATION")
    print("-" * 40)
    print("Based on distance array [35, 35, 30, 30, 30, 30, 10, 10, 10, 10]:")
    print("  - Global Rx 1-2 (Device 1, Rx 1-2): 35mm - LONG")
    print("  - Global Rx 3-6 (Device 2, Rx 1-4): 30mm - LONG")
    print("  - Global Rx 7-10 (Device 2, Rx 5-8): 10mm - SHORT")
    print()
    print(f"Expected SHORT channels: Any channel using Device 2, Receivers 5-8")
    print(f"Your data should have SHORT channels: {short_channels}")
    print()

    # Verify the short channels are correct
    expected_short_rx = [7, 8, 9, 10]  # Global receivers with 10mm
    actual_short_rx = list(set(channels[ch]['global_rx'] for ch in short_channels))

    if set(actual_short_rx) == set(expected_short_rx):
        print("✓ Short channel detection is CORRECT!")
    else:
        print(f"✗ Short channel detection issue:")
        print(f"  Expected short channels use Global Rx: {expected_short_rx}")
        print(f"  Detected short channels use Global Rx: {actual_short_rx}")

    return channels


def generate_region_mapping_template():
    """
    Generate a template for regional brain mapping.
    The XML template file does NOT contain anatomical region information -
    you'll need to define this based on your optode placement!
    """

    print()
    print("=" * 80)
    print("REGIONAL BRAIN MAPPING")
    print("=" * 80)
    print()
    print("The XML optode template file contains GEOMETRIC positions (X, Y coordinates)")
    print("but does NOT contain anatomical region labels like 'prefrontal cortex' or")
    print("'motor cortex'. These must be defined based on your actual optode placement.")
    print()
    print("Based on your setup (OctaMon + Brite24), here's a template to fill in:")
    print()

    # Template for user to fill in
    region_template = """
# REGIONAL MAPPING TEMPLATE
# Fill this in based on your actual optode placement on the head

REGION_MAP = {
    # Device 1 (OctaMon) - typically placed on prefrontal cortex
    # Channels 1-8: Device 1, Receivers 1-2
    1: {"region": "left_prefrontal", "hemisphere": "L", "lobe": "frontal"},
    2: {"region": "left_prefrontal", "hemisphere": "L", "lobe": "frontal"},
    3: {"region": "left_prefrontal", "hemisphere": "L", "lobe": "frontal"},
    4: {"region": "left_prefrontal", "hemisphere": "L", "lobe": "frontal"},
    5: {"region": "right_prefrontal", "hemisphere": "R", "lobe": "frontal"},
    6: {"region": "right_prefrontal", "hemisphere": "R", "lobe": "frontal"},
    7: {"region": "right_prefrontal", "hemisphere": "R", "lobe": "frontal"},
    8: {"region": "right_prefrontal", "hemisphere": "R", "lobe": "frontal"},

    # Device 2 (Brite24) - placement depends on your study
    # Long channels (30mm) - Channels 9-17
    9: {"region": "TBD", "hemisphere": "?", "lobe": "?"},
    10: {"region": "TBD", "hemisphere": "?", "lobe": "?"},
    11: {"region": "TBD", "hemisphere": "?", "lobe": "?"},
    12: {"region": "TBD", "hemisphere": "?", "lobe": "?"},
    13: {"region": "TBD", "hemisphere": "?", "lobe": "?"},
    14: {"region": "TBD", "hemisphere": "?", "lobe": "?"},
    15: {"region": "TBD", "hemisphere": "?", "lobe": "?"},
    16: {"region": "TBD", "hemisphere": "?", "lobe": "?"},
    17: {"region": "TBD", "hemisphere": "?", "lobe": "?"},

    # Short separation channels (10mm) - for superficial signal regression
    # These measure scalp blood flow, not brain activity
    18: {"region": "SSC", "hemisphere": "L", "lobe": "reference", "short": True},
    19: {"region": "SSC", "hemisphere": "L", "lobe": "reference", "short": True},
    20: {"region": "SSC", "hemisphere": "L", "lobe": "reference", "short": True},
    21: {"region": "SSC", "hemisphere": "R", "lobe": "reference", "short": True},
    22: {"region": "SSC", "hemisphere": "R", "lobe": "reference", "short": True},
    23: {"region": "SSC", "hemisphere": "R", "lobe": "reference", "short": True},
    24: {"region": "SSC", "hemisphere": "?", "lobe": "reference", "short": True},
    25: {"region": "SSC", "hemisphere": "?", "lobe": "reference", "short": True},
    26: {"region": "SSC", "hemisphere": "?", "lobe": "reference", "short": True},
}
"""
    print(region_template)
    print()
    print("Common region labels for fNIRS studies:")
    print("  - left_DLPFC / right_DLPFC (dorsolateral prefrontal cortex)")
    print("  - left_M1 / right_M1 (primary motor cortex)")
    print("  - left_PMC / right_PMC (premotor cortex)")
    print("  - left_SMA / right_SMA (supplementary motor area)")
    print("  - left_parietal / right_parietal")
    print("  - SSC (short separation channel - for superficial signal)")
    print()
    print("To properly map regions, you need:")
    print("  1. Your optode placement diagram/photo")
    print("  2. The 10-20 EEG positions you used as landmarks")
    print("  3. Knowledge of which receiver/source pairs measure which brain areas")


if __name__ == "__main__":
    channels = analyze_channel_mapping()
    generate_region_mapping_template()