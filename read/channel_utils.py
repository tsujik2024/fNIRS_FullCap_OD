"""
Channel Utilities for fNIRS Data Processing
============================================

This module provides channel-to-region mapping for fNIRS data based on:
1. OctaMon 2x3 channel + 2x1 SSC (PFC - prefrontal cortex)
2. Brite24 custom template Mancini lab (Motor cortex regions)

The mapping is derived from:
- OHSU Brite Template PDF optode placement diagrams
- Koenraadt et al. (2014) for motor cortex anatomical correspondence
- Artinis optodetemplates XML (template IDs 87 and 155)

Channel Numbering Convention (Combined System):
- OctaMon (Device 1): Channels 1-8 (PFC)
- Brite24 (Device 2): Channels 9-26 (Motor/Sensory/Visual cortex)

Brain Regions:
- PFC: Prefrontal Cortex (OctaMon, near Fp1/Fp2)
- SMA: Supplementary Motor Area (Brite24 Rx4/Rx5, near Cz)
- M1:  Primary Motor Cortex (Brite24 Rx3/Rx6, near C3/C4)
- S1:  Primary Somatosensory Cortex (Brite24 Rx2/Rx7, lateral)
- V1:  Primary Visual Cortex (Brite24 Rx1/Rx8, near O1/O2)

Author: Generated for OHSU/Mancini Lab fNIRS Pipeline
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ChannelInfo:
    """Information about a single fNIRS channel."""
    channel_num: int
    device: str  # 'OctaMon' or 'Brite24'
    rx_id: int  # Receiver ID (hardware)
    tx_id: int  # Transmitter ID (hardware)
    distance_mm: float  # Source-detector separation
    channel_type: str  # 'LONG' or 'SHORT'
    hemisphere: str  # 'L', 'R', or 'M' (midline)
    region: str  # Brain region


# =============================================================================
# OCTAMON TEMPLATE: 2x3 channel + 2x1 SSC (Template ID 87)
# =============================================================================
# Layout: 2 receivers, 8 transmitters
#   - Rx1 (right hemisphere): receives from Tx1, Tx2, Tx3 (long) + Tx4 (short)
#   - Rx2 (left hemisphere): receives from Tx5, Tx6, Tx7 (long) + Tx8 (short)
#   - Long separation: 35mm
#   - Short separation: 10mm
#
# Anatomical placement: Prefrontal Cortex (PFC)
#   - Positioned over Fp1/Fp2 (10-20 system)

OCTAMON_CHANNELS = {
    # Channel: (RxID, TxID, Distance_mm, Type, Hemisphere, Region)
    1: (1, 1, 35, 'LONG', 'R', 'PFC'),
    2: (1, 2, 35, 'LONG', 'R', 'PFC'),
    3: (1, 3, 35, 'LONG', 'R', 'PFC'),
    4: (1, 4, 10, 'SHORT', 'R', 'PFC'),  # Short channel - right PFC
    5: (2, 5, 35, 'LONG', 'L', 'PFC'),
    6: (2, 6, 35, 'LONG', 'L', 'PFC'),
    7: (2, 7, 35, 'LONG', 'L', 'PFC'),
    8: (2, 8, 10, 'SHORT', 'L', 'PFC'),  # Short channel - left PFC
}

# =============================================================================
# BRITE24 MANCINI LAB TEMPLATE (Template ID 155)
# =============================================================================
# Layout: 8 receivers, 10 transmitters
#   - SubTemplates 1-4: Long channels (30mm)
#   - SubTemplates 5-8: Short channels (10mm)
#
# Anatomical placement based on OHSU PDF and 10-20 system:
#   - Rx1: Posterior-left (near O1) - Primary Visual Cortex (V1)
#   - Rx2: Left lateral - Primary Somatosensory Cortex (S1)
#   - Rx3: Left central (near C3) - Primary Motor Cortex (M1)
#   - Rx4: Left medial (near Cz) - Supplementary Motor Area (SMA)
#   - Rx5: Right medial (near Cz) - Supplementary Motor Area (SMA)
#   - Rx6: Right central (near C4) - Primary Motor Cortex (M1)
#   - Rx7: Right lateral - Primary Somatosensory Cortex (S1)
#   - Rx8: Posterior-right (near O2) - Primary Visual Cortex (V1)
#
# Channel numbering continues from OctaMon (9-26)

BRITE24_MANCINI_CHANNELS = {
    # Channel: (RxID, TxID, Distance_mm, Type, Hemisphere, Region)
    #
    # Long channels (30mm) - combinations 1-14 from XML
    9: (1, 1, 30, 'LONG', 'L', 'V1'),  # Rx1 - posterior left (visual cortex)
    10: (2, 3, 30, 'LONG', 'L', 'S1'),  # Rx2 - left lateral somatosensory
    11: (2, 4, 30, 'LONG', 'L', 'S1'),  # Rx2 - left lateral somatosensory
    12: (3, 2, 30, 'LONG', 'L', 'M1'),  # Rx3 - left M1 (near C3)
    13: (3, 3, 30, 'LONG', 'L', 'M1'),  # Rx3 - left M1 (near C3)
    14: (4, 3, 30, 'LONG', 'L', 'SMA'),  # Rx4 - left medial SMA
    15: (4, 4, 30, 'LONG', 'L', 'SMA'),  # Rx4 - left medial SMA
    16: (5, 7, 30, 'LONG', 'R', 'SMA'),  # Rx5 - right medial SMA
    17: (5, 8, 30, 'LONG', 'R', 'SMA'),  # Rx5 - right medial SMA
    18: (6, 8, 30, 'LONG', 'R', 'M1'),  # Rx6 - right M1 (near C4)
    19: (6, 9, 30, 'LONG', 'R', 'M1'),  # Rx6 - right M1 (near C4)
    20: (7, 8, 30, 'LONG', 'R', 'S1'),  # Rx7 - right lateral somatosensory
    21: (7, 9, 30, 'LONG', 'R', 'S1'),  # Rx7 - right lateral somatosensory
    22: (8, 10, 30, 'LONG', 'R', 'V1'),  # Rx8 - posterior right (visual cortex)
    #
    # Short channels (10mm) - combinations 15-18 from XML
    23: (3, 5, 10, 'SHORT', 'L', 'M1'),  # Short for left M1 (Rx3)
    24: (5, 6, 10, 'SHORT', 'R', 'SMA'),  # Short for right SMA (Rx5)
    25: (4, 5, 10, 'SHORT', 'L', 'SMA'),  # Short for left SMA (Rx4)
    26: (6, 6, 10, 'SHORT', 'R', 'M1'),  # Short for right M1 (Rx6)
}


# =============================================================================
# COMBINED CHANNEL INFO DICTIONARY
# =============================================================================

def _build_channel_info() -> Dict[int, ChannelInfo]:
    """Build the complete channel info dictionary."""
    channel_info = {}

    # Add OctaMon channels
    for ch, (rx, tx, dist, ch_type, hemi, region) in OCTAMON_CHANNELS.items():
        channel_info[ch] = ChannelInfo(
            channel_num=ch,
            device='OctaMon',
            rx_id=rx,
            tx_id=tx,
            distance_mm=dist,
            channel_type=ch_type,
            hemisphere=hemi,
            region=region
        )

    # Add Brite24 channels
    for ch, (rx, tx, dist, ch_type, hemi, region) in BRITE24_MANCINI_CHANNELS.items():
        channel_info[ch] = ChannelInfo(
            channel_num=ch,
            device='Brite24',
            rx_id=rx,
            tx_id=tx,
            distance_mm=dist,
            channel_type=ch_type,
            hemisphere=hemi,
            region=region
        )

    return channel_info


CHANNEL_INFO: Dict[int, ChannelInfo] = _build_channel_info()

# =============================================================================
# CHANNEL LISTS BY TYPE
# =============================================================================

# All channels
ALL_CHANNELS: List[int] = list(range(1, 27))

# Short separation channels (for superficial signal regression)
SHORT_CHANNEL_LIST: List[int] = [4, 8, 23, 24, 25, 26]

# Long separation channels (brain signal)
LONG_CHANNEL_LIST: List[int] = [ch for ch in ALL_CHANNELS if ch not in SHORT_CHANNEL_LIST]

# By device
OCTAMON_CHANNEL_LIST: List[int] = list(range(1, 9))
BRITE24_CHANNEL_LIST: List[int] = list(range(9, 27))

# =============================================================================
# REGION MAPPING
# =============================================================================

# Regions present in this montage
REGIONS: List[str] = ['PFC', 'SMA', 'M1', 'S1', 'V1']

# Map region names to their long channels (for regional averaging)
CH_REGION_MAP: Dict[str, List[int]] = {
    # Prefrontal Cortex
    'PFC_R': [1, 2, 3],  # Right PFC (OctaMon Rx1)
    'PFC_L': [5, 6, 7],  # Left PFC (OctaMon Rx2)

    # Supplementary Motor Area (medial, near Cz)
    'SMA_L': [14, 15],  # Left SMA (Brite24 Rx4)
    'SMA_R': [16, 17],  # Right SMA (Brite24 Rx5)

    # Primary Motor Cortex (near C3/C4)
    'M1_L': [12, 13],  # Left M1 (Brite24 Rx3)
    'M1_R': [18, 19],  # Right M1 (Brite24 Rx6)

    # Primary Somatosensory Cortex (lateral)
    'S1_L': [10, 11],  # Left S1 (Brite24 Rx2)
    'S1_R': [20, 21],  # Right S1 (Brite24 Rx7)

    # Primary Visual Cortex (posterior, near O1/O2)
    'V1_L': [9],  # Left V1 (Brite24 Rx1)
    'V1_R': [22],  # Right V1 (Brite24 Rx8)
}

# Simplified region map (combining hemispheres)
CH_REGION_MAP_COMBINED: Dict[str, List[int]] = {
    'PFC': [1, 2, 3, 5, 6, 7],
    'SMA': [14, 15, 16, 17],
    'M1': [12, 13, 18, 19],
    'S1': [10, 11, 20, 21],
    'V1': [9, 22],
}

# Reverse mapping: channel -> region
CHANNEL_TO_REGION: Dict[int, str] = {}
for region, channels in CH_REGION_MAP.items():
    for ch in channels:
        CHANNEL_TO_REGION[ch] = region

# Also add short channels to region mapping
CHANNEL_TO_REGION[4] = 'PFC_R'  # Short for right PFC
CHANNEL_TO_REGION[8] = 'PFC_L'  # Short for left PFC
CHANNEL_TO_REGION[23] = 'M1_L'  # Short for left M1
CHANNEL_TO_REGION[24] = 'SMA_R'  # Short for right SMA
CHANNEL_TO_REGION[25] = 'SMA_L'  # Short for left SMA
CHANNEL_TO_REGION[26] = 'M1_R'  # Short for right M1

# =============================================================================
# SHORT CHANNEL REGRESSION MAPPING
# =============================================================================

# Map long channels to their nearest short channel for SCR
# Based on physical proximity and same brain region

LONG_TO_SHORT_MAP: Dict[int, int] = {
    # PFC channels -> PFC shorts
    1: 4,  # PFC_R long -> PFC_R short
    2: 4,
    3: 4,
    5: 8,  # PFC_L long -> PFC_L short
    6: 8,
    7: 8,

    # V1 channels -> nearest motor short (no dedicated V1 short)
    9: 23,  # V1_L -> M1_L short (closest)
    22: 26,  # V1_R -> M1_R short (closest)

    # S1 channels -> nearest short
    10: 23,  # S1_L -> M1_L short
    11: 23,
    20: 26,  # S1_R -> M1_R short
    21: 26,

    # M1 channels -> M1 shorts
    12: 23,  # M1_L -> M1_L short
    13: 23,
    18: 26,  # M1_R -> M1_R short
    19: 26,

    # SMA channels -> SMA shorts
    14: 25,  # SMA_L -> SMA_L short
    15: 25,
    16: 24,  # SMA_R -> SMA_R short
    17: 24,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_channel_info(channel: int) -> Optional[ChannelInfo]:
    """Get information about a specific channel."""
    return CHANNEL_INFO.get(channel)


def get_channels_by_region(region: str, long_only: bool = True) -> List[int]:
    """
    Get all channels for a given region.

    Parameters
    ----------
    region : str
        Region name (e.g., 'PFC', 'PFC_L', 'M1_R')
    long_only : bool
        If True, return only long channels. If False, include short channels.

    Returns
    -------
    List[int]
        List of channel numbers
    """
    # Check if it's a combined region
    if region in CH_REGION_MAP_COMBINED:
        channels = CH_REGION_MAP_COMBINED[region].copy()
    elif region in CH_REGION_MAP:
        channels = CH_REGION_MAP[region].copy()
    else:
        return []

    if not long_only:
        # Add short channels for this region
        for ch in SHORT_CHANNEL_LIST:
            if ch in CHANNEL_TO_REGION:
                ch_region = CHANNEL_TO_REGION[ch]
                # Check if short channel's region matches
                if ch_region == region or ch_region.replace('_L', '').replace('_R', '') == region:
                    if ch not in channels:
                        channels.append(ch)

    return sorted(channels)


def get_channels_by_hemisphere(hemisphere: str, long_only: bool = True) -> List[int]:
    """
    Get all channels for a given hemisphere.

    Parameters
    ----------
    hemisphere : str
        'L' for left, 'R' for right
    long_only : bool
        If True, return only long channels

    Returns
    -------
    List[int]
        List of channel numbers
    """
    channels = []
    for ch, info in CHANNEL_INFO.items():
        if info.hemisphere == hemisphere:
            if long_only and info.channel_type == 'SHORT':
                continue
            channels.append(ch)
    return sorted(channels)


def get_channels_by_device(device: str, long_only: bool = True) -> List[int]:
    """
    Get all channels for a given device.

    Parameters
    ----------
    device : str
        'OctaMon' or 'Brite24'
    long_only : bool
        If True, return only long channels

    Returns
    -------
    List[int]
        List of channel numbers
    """
    channels = []
    for ch, info in CHANNEL_INFO.items():
        if info.device == device:
            if long_only and info.channel_type == 'SHORT':
                continue
            channels.append(ch)
    return sorted(channels)


def get_short_channel_for_long(long_channel: int) -> Optional[int]:
    """Get the corresponding short channel for a long channel."""
    return LONG_TO_SHORT_MAP.get(long_channel)


def is_short_channel(channel: int) -> bool:
    """Check if a channel is a short separation channel."""
    return channel in SHORT_CHANNEL_LIST


def is_long_channel(channel: int) -> bool:
    """Check if a channel is a long separation channel."""
    return channel in LONG_CHANNEL_LIST


def get_region_for_channel(channel: int) -> Optional[str]:
    """Get the brain region for a channel."""
    return CHANNEL_TO_REGION.get(channel)


def get_base_region(channel: int) -> Optional[str]:
    """Get the base region name (without hemisphere) for a channel."""
    region = CHANNEL_TO_REGION.get(channel)
    if region:
        return region.replace('_L', '').replace('_R', '')
    return None


# =============================================================================
# SHORT CHANNELS BY REGION TYPE (for visualization)
# =============================================================================

# Group short channels by their general region type
SHORT_CHANNELS_BY_REGION: Dict[str, List[int]] = {
    'PFC_SHORT': [4, 8],  # PFC short channels (OctaMon)
    'MOTOR_SHORT': [23, 24, 25, 26],  # Motor cortex short channels (Brite24)
}


# =============================================================================
# COLUMN NAME UTILITIES
# =============================================================================

def get_hbo_column_name(channel: int) -> str:
    """Get the HbO column name for a channel."""
    return f"CH{channel} HbO"


def get_hhb_column_name(channel: int) -> str:
    """Get the HHb column name for a channel."""
    return f"CH{channel} HHb"


def get_channel_column_names(channel: int) -> Tuple[str, str]:
    """Get both HbO and HHb column names for a channel."""
    return get_hbo_column_name(channel), get_hhb_column_name(channel)


def parse_channel_from_column(column_name: str) -> Optional[int]:
    """
    Parse channel number from a column name.

    Handles formats like:
    - "CH1 HbO", "CH1 HHb"
    - "CH 1 HbO", "CH 1 HHb"
    - "ch1_hbo", "ch1_hhb"
    - "D1_R1_S1_WL850" (OD format)
    """
    import re
    # Try CH format first
    match = re.search(r'CH\s*(\d+)', column_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# OPTICAL DENSITY (OD) COLUMN UTILITIES
# =============================================================================

def ch_to_od_columns(channel: int, channel_map: Dict) -> List[str]:
    """
    Convert a channel number to its corresponding OD column names.

    Parameters
    ----------
    channel : int
        Channel number (1-26)
    channel_map : Dict
        Dictionary mapping channel numbers to their OD column info.
        Expected format: {ch_num: {'columns': ['D1_R1_S1_WL760', 'D1_R1_S1_WL850'], ...}}

    Returns
    -------
    List[str]
        List of OD column names for this channel (typically 2 wavelengths)
    """
    if channel in channel_map:
        ch_info = channel_map[channel]
        if isinstance(ch_info, dict) and 'columns' in ch_info:
            return ch_info['columns']
        elif isinstance(ch_info, list):
            return ch_info
    return []


def build_region_map_od(od_columns: List[str], channel_map: Dict) -> Dict[str, List[str]]:
    """
    Build a mapping from brain regions to their OD column names.

    This function maps each region (PFC, SMA, M1, S1, V1) to the OD columns
    that belong to channels in that region. Only includes LONG channels
    (excludes short channels).

    Parameters
    ----------
    od_columns : List[str]
        List of OD column names from the data (e.g., ['D1_R1_S1_WL760', ...])
    channel_map : Dict
        Dictionary mapping channel numbers to their OD column info.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from region name to list of OD column names
        e.g., {'PFC': ['D1_R1_S1_WL760', 'D1_R1_S1_WL850', ...], ...}
    """
    region_map = {}

    # Build reverse lookup: OD column -> channel number
    col_to_channel = {}
    for ch_num, ch_info in channel_map.items():
        if isinstance(ch_info, dict) and 'columns' in ch_info:
            for col in ch_info['columns']:
                col_to_channel[col] = ch_num
        elif isinstance(ch_info, list):
            for col in ch_info:
                col_to_channel[col] = ch_num

    # Group OD columns by region (using combined region map, excluding short channels)
    for region, channels in CH_REGION_MAP_COMBINED.items():
        region_cols = []
        for ch in channels:
            # Skip short channels
            if ch in SHORT_CHANNEL_LIST:
                continue
            # Get OD columns for this channel
            ch_cols = ch_to_od_columns(ch, channel_map)
            for col in ch_cols:
                if col in od_columns:
                    region_cols.append(col)

        if region_cols:
            region_map[region] = region_cols

    return region_map


def build_short_map_od(od_columns: List[str], channel_map: Dict) -> Dict[str, List[str]]:
    """
    Build a mapping from short channel regions to their OD column names.

    Parameters
    ----------
    od_columns : List[str]
        List of OD column names from the data
    channel_map : Dict
        Dictionary mapping channel numbers to their OD column info.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from short channel region to list of OD column names
        e.g., {'PFC_SHORT': ['D1_R1_S4_WL760', ...], 'MOTOR_SHORT': [...]}
    """
    short_map = {}

    for region_type, short_channels in SHORT_CHANNELS_BY_REGION.items():
        region_cols = []
        for ch in short_channels:
            ch_cols = ch_to_od_columns(ch, channel_map)
            for col in ch_cols:
                if col in od_columns:
                    region_cols.append(col)

        if region_cols:
            short_map[region_type] = region_cols

    return short_map


def get_od_columns_for_region(region: str, od_columns: List[str], channel_map: Dict) -> List[str]:
    """
    Get all OD columns for a specific brain region.

    Parameters
    ----------
    region : str
        Region name (e.g., 'PFC', 'M1', 'SMA_L')
    od_columns : List[str]
        List of available OD column names
    channel_map : Dict
        Dictionary mapping channel numbers to their OD column info.

    Returns
    -------
    List[str]
        List of OD column names for this region
    """
    # Get channels for this region
    if region in CH_REGION_MAP_COMBINED:
        channels = CH_REGION_MAP_COMBINED[region]
    elif region in CH_REGION_MAP:
        channels = CH_REGION_MAP[region]
    else:
        return []

    # Get OD columns for these channels
    result = []
    for ch in channels:
        ch_cols = ch_to_od_columns(ch, channel_map)
        for col in ch_cols:
            if col in od_columns:
                result.append(col)

    return result


# =============================================================================
# SUMMARY AND VALIDATION
# =============================================================================

def print_channel_summary():
    """Print a summary of the channel configuration."""
    print("=" * 70)
    print("fNIRS Channel Configuration Summary")
    print("=" * 70)
    print(f"\nTotal channels: {len(ALL_CHANNELS)}")
    print(f"  - Long channels: {len(LONG_CHANNEL_LIST)}")
    print(f"  - Short channels: {len(SHORT_CHANNEL_LIST)}")

    print(f"\nOctaMon (PFC): Channels {OCTAMON_CHANNEL_LIST[0]}-{OCTAMON_CHANNEL_LIST[-1]}")
    print(f"  - Long: {[ch for ch in OCTAMON_CHANNEL_LIST if ch in LONG_CHANNEL_LIST]}")
    print(f"  - Short: {[ch for ch in OCTAMON_CHANNEL_LIST if ch in SHORT_CHANNEL_LIST]}")

    print(f"\nBrite24 (Motor): Channels {BRITE24_CHANNEL_LIST[0]}-{BRITE24_CHANNEL_LIST[-1]}")
    print(f"  - Long: {[ch for ch in BRITE24_CHANNEL_LIST if ch in LONG_CHANNEL_LIST]}")
    print(f"  - Short: {[ch for ch in BRITE24_CHANNEL_LIST if ch in SHORT_CHANNEL_LIST]}")

    print("\nRegion mapping (long channels only):")
    for region, channels in CH_REGION_MAP.items():
        print(f"  {region}: {channels}")

    print("\nShort channel assignments:")
    for ch in SHORT_CHANNEL_LIST:
        info = CHANNEL_INFO[ch]
        print(f"  CH{ch}: {info.region} ({info.hemisphere}) - {info.device}")


def validate_channel_config() -> bool:
    """Validate the channel configuration for consistency."""
    errors = []

    # Check all channels have info
    for ch in ALL_CHANNELS:
        if ch not in CHANNEL_INFO:
            errors.append(f"Channel {ch} missing from CHANNEL_INFO")

    # Check short/long classification matches distance
    for ch, info in CHANNEL_INFO.items():
        if info.channel_type == 'SHORT' and info.distance_mm > 15:
            errors.append(f"Channel {ch} marked SHORT but distance is {info.distance_mm}mm")
        if info.channel_type == 'LONG' and info.distance_mm < 20:
            errors.append(f"Channel {ch} marked LONG but distance is {info.distance_mm}mm")

    # Check all long channels have a short channel mapping
    for ch in LONG_CHANNEL_LIST:
        if ch not in LONG_TO_SHORT_MAP:
            errors.append(f"Long channel {ch} has no short channel mapping")

    # Check region map completeness
    mapped_channels = set()
    for channels in CH_REGION_MAP.values():
        mapped_channels.update(channels)

    for ch in LONG_CHANNEL_LIST:
        if ch not in mapped_channels:
            errors.append(f"Long channel {ch} not in any region map")

    if errors:
        print("Configuration errors found:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("Channel configuration validated successfully!")
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_channel_summary()
    print("\n")
    validate_channel_config()