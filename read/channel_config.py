"""Channel layout for the OHSU full cap (OctaMon + Brite24)."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ChannelInfo:
    channel_num: int
    device: str                    # 'OctaMon' | 'Brite24'
    rx_id: int
    tx_id: int
    distance_mm: float
    channel_type: str              # 'LONG' | 'SHORT'
    hemisphere: str                # 'L' | 'R' | 'M'
    region: str
    file_columns: Tuple[int, int]  # 1-indexed TXT columns (wl1, wl2)


# rx, tx, dist, type, hemi, region, file cols. Rx2's short is Tx6, not the last Tx.
OCTAMON_CHANNELS = {
    1: (1, 1, 35, 'LONG',  'R', 'PFC', (2,  3)),
    2: (1, 2, 35, 'LONG',  'R', 'PFC', (4,  5)),
    3: (1, 3, 35, 'LONG',  'R', 'PFC', (6,  7)),
    4: (1, 4, 10, 'SHORT', 'R', 'PFC', (8,  9)),
    5: (2, 5, 35, 'LONG',  'L', 'PFC', (10, 11)),
    6: (2, 6, 10, 'SHORT', 'L', 'PFC', (12, 13)),
    7: (2, 7, 35, 'LONG',  'L', 'PFC', (14, 15)),
    8: (2, 8, 35, 'LONG',  'L', 'PFC', (16, 17)),
}

# 'Mancini lab' template. T5 is the shared short for Rx3/Rx4, T6 for Rx5/Rx6.
BRITE24_MANCINI_CHANNELS = {
    9:  (1, 1,  30, 'LONG',  'L', 'V1',  (18, 19)),
    10: (2, 3,  30, 'LONG',  'L', 'M1',  (20, 21)),
    11: (2, 4,  30, 'LONG',  'L', 'S1',  (22, 23)),
    12: (3, 2,  30, 'LONG',  'L', 'SMA', (24, 25)),
    13: (3, 3,  30, 'LONG',  'L', 'SMA', (26, 27)),
    14: (3, 5,  10, 'SHORT', 'L', 'SMA', (28, 29)),
    15: (4, 3,  30, 'LONG',  'L', 'M1',  (30, 31)),
    16: (4, 4,  30, 'LONG',  'L', 'S1',  (32, 33)),
    17: (4, 5,  10, 'SHORT', 'L', 'S1',  (34, 35)),
    18: (5, 6,  10, 'SHORT', 'R', 'SMA', (36, 37)),
    19: (5, 7,  30, 'LONG',  'R', 'SMA', (38, 39)),
    20: (5, 8,  30, 'LONG',  'R', 'SMA', (40, 41)),
    21: (6, 6,  10, 'SHORT', 'R', 'S1',  (42, 43)),
    22: (6, 8,  30, 'LONG',  'R', 'M1',  (44, 45)),
    23: (6, 9,  30, 'LONG',  'R', 'S1',  (46, 47)),
    24: (7, 8,  30, 'LONG',  'R', 'M1',  (48, 49)),
    25: (7, 9,  30, 'LONG',  'R', 'S1',  (50, 51)),
    26: (8, 10, 30, 'LONG',  'R', 'V1',  (52, 53)),
}

CHANNEL_INFO: Dict[int, ChannelInfo] = {
    **{ch: ChannelInfo(ch, 'OctaMon', *row) for ch, row in OCTAMON_CHANNELS.items()},
    **{ch: ChannelInfo(ch, 'Brite24', *row) for ch, row in BRITE24_MANCINI_CHANNELS.items()},
}

ALL_CHANNELS: List[int] = list(range(1, 27))
SHORT_CHANNEL_LIST: List[int] = [4, 6, 14, 17, 18, 21]
LONG_CHANNEL_LIST: List[int] = [c for c in ALL_CHANNELS if c not in SHORT_CHANNEL_LIST]
OCTAMON_CHANNEL_LIST: List[int] = list(range(1, 9))
BRITE24_CHANNEL_LIST: List[int] = list(range(9, 27))
REGIONS: List[str] = ['PFC', 'SMA', 'M1', 'S1', 'V1']

CH_REGION_MAP: Dict[str, List[int]] = {
    'PFC_R': [1, 2, 3],   'PFC_L': [5, 7, 8],
    'M1_L':  [10, 15],    'M1_R':  [22, 24],
    'SMA_L': [12, 13],    'SMA_R': [19, 20],
    'S1_L':  [11, 16],    'S1_R':  [23, 25],
    'V1_L':  [9],         'V1_R':  [26],
}

CH_REGION_MAP_COMBINED: Dict[str, List[int]] = {
    'PFC': [1, 2, 3, 5, 7, 8],
    'M1':  [10, 15, 22, 24],
    'SMA': [12, 13, 19, 20],
    'S1':  [11, 16, 23, 25],
    'V1':  [9, 26],
}

CHANNEL_TO_REGION: Dict[int, str] = {ch: r for r, chs in CH_REGION_MAP.items() for ch in chs}
CHANNEL_TO_REGION.update({4: 'PFC_R', 6: 'PFC_L', 14: 'SMA_L', 17: 'S1_L', 18: 'SMA_R', 21: 'S1_R'})

# SCR reference(s) per long channel; the SCR step averages the list into one regressor.
# PFC longs use their one ipsilateral OctaMon short. SMA/M1/S1 longs share all four
# Brite24 shorts (M1 has no dedicated short, and pooling is steadier than picking the
# nearest). V1 (CH9, CH26) is deliberately absent: no short is anywhere near it.
_BRITE24_SHORTS = tuple(sorted(ch for ch, row in BRITE24_MANCINI_CHANNELS.items() if row[3] == 'SHORT'))

SCR_EXEMPT_LONGS: List[int] = sorted(
    ch for ch, i in CHANNEL_INFO.items() if i.region == 'V1' and i.channel_type == 'LONG')

LONG_TO_SHORT_MAP: Dict[int, List[int]] = {
    # PFC: ipsilateral short first, contralateral OctaMon short second as a
    # fallback -- used only if the ipsilateral one fails the active quality
    # criterion. See SCR_PRIORITY_LONGS: these two entries are never pooled
    # together the way the Brite24 motor shorts are.
    1: [4, 6], 2: [4, 6], 3: [4, 6],
    5: [6, 4], 7: [6, 4], 8: [6, 4],
    **{ch: _BRITE24_SHORTS for ch in (10, 11, 12, 13, 15, 16, 19, 20, 22, 23, 24, 25)},
}

# PFC longs: SCR uses ONE short at a time, in priority order (ipsilateral
# first) -- never averaged together, unlike the pooled Brite24 motor shorts.
# Falls back to the contralateral OctaMon short only when the ipsilateral
# one fails the active channel-quality criterion (see process_file._apply_scr).
SCR_PRIORITY_LONGS: List[int] = [1, 2, 3, 5, 7, 8]

SHORT_CHANNELS_BY_REGION: Dict[str, List[int]] = {
    'PFC_SHORT': [4, 6],
    'MOTOR_SHORT': [14, 17, 18, 21],
}

SHORT_CHANNELS_BY_SUBREGION: Dict[str, List[int]] = {
    'PFC_L_SHORT': [6],  'PFC_R_SHORT': [4],
    'SMA_L_SHORT': [14], 'SMA_R_SHORT': [18],
    'S1_L_SHORT':  [17], 'S1_R_SHORT':  [21],
}


def get_channel_info(channel: int) -> Optional[ChannelInfo]:
    return CHANNEL_INFO.get(channel)


def is_short_channel(channel: int) -> bool:
    return channel in SHORT_CHANNEL_LIST


def is_long_channel(channel: int) -> bool:
    return channel in LONG_CHANNEL_LIST


def get_region_for_channel(channel: int) -> Optional[str]:
    return CHANNEL_TO_REGION.get(channel)


def get_base_region(channel: int) -> Optional[str]:
    region = CHANNEL_TO_REGION.get(channel)
    return region.split('_')[0] if region else None


def get_shorts_for_long(long_channel: int) -> Optional[List[int]]:
    """Short(s) usable as the SCR reference, in priority order; None for
    SCR-exempt longs (V1). PFC longs (see SCR_PRIORITY_LONGS) use only the
    first entry that passes quality -- never pooled. Motor longs pool every
    entry that passes quality."""
    refs = LONG_TO_SHORT_MAP.get(long_channel)
    return list(refs) if refs is not None else None


def get_channels_by_region(region: str, long_only: bool = True) -> List[int]:
    chs = CH_REGION_MAP_COMBINED.get(region) or CH_REGION_MAP.get(region)
    if chs is None:
        return []
    chs = list(chs)
    if not long_only:
        base = region.split('_')[0]
        for ch in SHORT_CHANNEL_LIST:
            r = CHANNEL_TO_REGION.get(ch, '')
            # a hemisphere-specific region only gets its own hemisphere's short
            if r == region or (region == base and r.split('_')[0] == base):
                chs.append(ch)
    return sorted(set(chs))


def get_channels_by_hemisphere(hemisphere: str, long_only: bool = True) -> List[int]:
    return sorted(ch for ch, i in CHANNEL_INFO.items()
                  if i.hemisphere == hemisphere and not (long_only and i.channel_type == 'SHORT'))


def get_channels_by_device(device: str, long_only: bool = True) -> List[int]:
    return sorted(ch for ch, i in CHANNEL_INFO.items()
                  if i.device == device and not (long_only and i.channel_type == 'SHORT'))


def get_file_columns(channel: int) -> Optional[Tuple[int, int]]:
    info = CHANNEL_INFO.get(channel)
    return info.file_columns if info else None


def build_region_map_od(columns: List[str], channel_map: Dict[int, Dict]) -> Dict[str, List[str]]:
    """Group OD column names by region, using the loader's channel_map."""
    col_to_ch = {c: ch for ch, info in channel_map.items()
                 for cols in info.get('columns', {}).values() for c in cols}
    out: Dict[str, List[str]] = {}
    for col in columns:
        region = CHANNEL_TO_REGION.get(col_to_ch.get(col))
        if region:
            out.setdefault(region, []).append(col)
    return out


def channel_from_file_column(column_idx: int) -> Optional[int]:
    for ch, info in CHANNEL_INFO.items():
        if column_idx in info.file_columns:
            return ch
    return None


def parse_channel_from_column(column_name: str) -> Optional[int]:
    m = re.search(r'CH\s*(\d+)', column_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def validate_channel_config() -> bool:
    errors: List[str] = []
    errors += [f"CH{ch} missing from CHANNEL_INFO" for ch in ALL_CHANNELS if ch not in CHANNEL_INFO]

    for ch, i in CHANNEL_INFO.items():
        if i.channel_type == 'SHORT' and i.distance_mm > 15:
            errors.append(f"CH{ch} marked SHORT but distance is {i.distance_mm}mm")
        if i.channel_type == 'LONG' and i.distance_mm < 20:
            errors.append(f"CH{ch} marked LONG but distance is {i.distance_mm}mm")
        a, b = i.file_columns
        if b != a + 1:
            errors.append(f"CH{ch} file_columns {i.file_columns} are not adjacent")

    derived = sorted(c for c, i in CHANNEL_INFO.items() if i.channel_type == 'SHORT')
    if derived != sorted(SHORT_CHANNEL_LIST):
        errors.append(f"SHORT_CHANNEL_LIST {sorted(SHORT_CHANNEL_LIST)} != derived {derived}")

    for ch in LONG_CHANNEL_LIST:
        if ch not in LONG_TO_SHORT_MAP and ch not in SCR_EXEMPT_LONGS:
            errors.append(f"Long CH{ch} has no LONG_TO_SHORT_MAP entry and is not in SCR_EXEMPT_LONGS")

    for long_ch, refs in LONG_TO_SHORT_MAP.items():
        if not isinstance(refs, (list, tuple)) or not refs:
            errors.append(f"LONG_TO_SHORT_MAP[{long_ch}] = {refs!r}: expected a non-empty list of short channel IDs")
            continue
        errors += [f"LONG_TO_SHORT_MAP[{long_ch}] references CH{s}, which is not a short channel"
                   for s in refs if s not in SHORT_CHANNEL_LIST]

    mapped = {ch for chs in CH_REGION_MAP.values() for ch in chs}
    errors += [f"Long CH{ch} not in any CH_REGION_MAP entry" for ch in LONG_CHANNEL_LIST if ch not in mapped]

    seen: Dict[int, int] = {}
    for ch, i in CHANNEL_INFO.items():
        for col in i.file_columns:
            if col in seen:
                errors.append(f"file_column {col} reused by CH{seen[col]} and CH{ch}")
            seen[col] = ch

    for e in errors:
        print(f"  - {e}")
    return not errors


if __name__ == "__main__":
    ok = validate_channel_config()
    print(f"{len(ALL_CHANNELS)} channels ({len(LONG_CHANNEL_LIST)} long, {len(SHORT_CHANNEL_LIST)} short)")
    print("SCR-exempt longs:", SCR_EXEMPT_LONGS)
    print("Brite24 shorts:", list(_BRITE24_SHORTS))
    print("OK" if ok else "FAILED")
