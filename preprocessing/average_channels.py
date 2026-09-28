"""Per-region channel means for the full cap. Region definitions come from
read.channel_config """

import numpy as np
import pandas as pd

from read.channel_config import CH_REGION_MAP, CH_REGION_MAP_COMBINED

_PASSTHROUGH = ("Sample number", "Event", "grand_oxy", "grand_deoxy")

# convention -> (oxy suffix, deoxy suffix); anything unrecognised is treated as HbO/HHb
_SUFFIXES = {"_oxy": ("_oxy", "_deoxy"), "O2Hb": (" O2Hb", " HHb")}


class FullCapChannelAverager:
    def __init__(self):
        self.regions = {
            name: {"channels": chs,
                   "hemisphere": "right" if name.endswith("_R") else "left" if name.endswith("_L") else "midline"}
            for name, chs in CH_REGION_MAP.items()
        }
        self.combined_regions = CH_REGION_MAP_COMBINED

    @staticmethod
    def detect_naming_convention(column_names):
        """'HbO', 'O2Hb', '_oxy', '_deoxy', or None."""
        for key in ("HbO", "O2Hb", "_oxy", "_deoxy"):
            if any(key in col for col in column_names):
                return key
        return None

    def average_regions(self, df: pd.DataFrame, channels_to_exclude=None) -> pd.DataFrame:
        """Mean per region (and combined region) for oxy and deoxy.

        Per-hemisphere regions with no usable channels come back as NaN columns;
        combined regions are just omitted.
        """
        exclude = set(channels_to_exclude or ())
        oxy_sfx, deoxy_sfx = _SUFFIXES.get(self.detect_naming_convention(list(df.columns)), (" HbO", " HHb"))
        out = {c: df[c] for c in _PASSTHROUGH if c in df.columns}

        def add(name, channels, keep_empty):
            for out_sfx, sfx in (("_oxy", oxy_sfx), ("_deoxy", deoxy_sfx)):
                cols = [f"CH{ch}{sfx}" for ch in channels
                        if ch not in exclude and f"CH{ch}{sfx}" in df.columns]
                if cols:
                    out[f"{name}{out_sfx}"] = df[cols].mean(axis=1)
                elif keep_empty:
                    out[f"{name}{out_sfx}"] = np.nan

        for name, info in self.regions.items():
            add(name, info["channels"], keep_empty=True)
        for name, channels in self.combined_regions.items():
            add(name, channels, keep_empty=False)

        return pd.DataFrame(out, index=df.index)
