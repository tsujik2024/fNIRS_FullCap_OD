"""Plots for the fNIRS pipeline.

Concentration columns: CH{n}_oxy / CH{n}_deoxy for channels, {REGION}_oxy / _deoxy for
regions. OD columns: D{d}_R{r}_T{t}_WL{nm}, as produced by loaders.read_txt_file.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from read.channel_config import (
    CHANNEL_INFO,
    CH_REGION_MAP,
    CH_REGION_MAP_COMBINED,
    SHORT_CHANNEL_LIST,
    build_region_map_od,
)

logger = logging.getLogger(__name__)

OXY, DEOXY = "_oxy", "_deoxy"

CHANNEL_TYPE_COLORS = {
    "PFC_LONG":    "#1f77b4",
    "PFC_SHORT":   "#aec7e8",
    "MOTOR_LONG":  "#ff7f0e",
    "MOTOR_SHORT": "#ffbb78",
    "SHORT":       "#2ca02c",
    "LONG":        "#d62728",
}
RAW_COLORS       = {"oxy": "#E41A1C", "deoxy": "#377EB8", "wl1": "#E41A1C", "wl2": "#377EB8"}
PROCESSED_COLORS = {"oxy": "#FF6B6B", "deoxy": "#4ECDC4", "wl1": "#FF6B6B", "wl2": "#4ECDC4"}
PROCESSED_DEOXY_LS = "--"

_MOTOR_REGIONS = {"M1", "SMA", "S1", "V1"}
_SKIP_COLS = {"Sample number", "Event"}


def _od_cols(data: pd.DataFrame) -> List[str]:
    return [c for c in data.columns if "WL" in c and c not in _SKIP_COLS]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


class FNIRSVisualizer:
    def __init__(self, fs: float = 50.0, data_type: str = "concentration"):
        if data_type not in ("concentration", "od"):
            raise ValueError(f"data_type must be 'concentration' or 'od', got {data_type!r}")
        self.fs = fs
        self.data_type = data_type
        self.y_label = "μM" if data_type == "concentration" else "OD"
        plt.ioff()

    # ---- public API --------------------------------------------------------

    def plot_raw_od(self, data, output_path, y_limits=None, channel_map=None) -> None:
        if self.data_type != "od":
            logger.warning("plot_raw_od called on non-OD visualizer; ignoring")
            return
        cols = _od_cols(data)
        if not cols:
            logger.warning(f"No OD columns for {output_path}")
            return

        time = self._time_axis(len(data))
        fig, axes = self._make_subplots(len(cols))
        for ax, col in zip(axes, cols):
            ch = self._channel_for_od_col(col, channel_map)
            is_short, region = self._channel_attrs(ch)
            ax.plot(time, _num(data[col]), color=self._color_for(is_short, region), linewidth=1, label=col)
            self._format_axis(ax, self._od_title(ch, col, is_short, region))

        self._apply_ylimits(axes, y_limits)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        self._save_close(fig, output_path)

    def plot_raw_all_channels(self, data, output_path, y_limits=None, channel_map=None) -> None:
        if self.data_type == "concentration":
            self._plot_all_channels_conc(data, output_path, y_limits)
        else:
            self._plot_all_channels_od(data, output_path, y_limits, channel_map)

    def plot_raw_regions(self, regional_data, output_path, y_limits=None, channel_map=None) -> None:
        if self.data_type == "concentration":
            self._plot_concentration_regions(regional_data, output_path, processed=False, y_limits=y_limits)
        else:
            self._plot_od_regions(regional_data, output_path, channel_map, y_limits)

    def plot_processed_regions(self, data, output_path, y_limits=None) -> None:
        if self.data_type == "concentration":
            self._plot_concentration_regions(data, output_path, processed=True, y_limits=y_limits)
        else:
            logger.warning("plot_processed_regions for OD is not supported")

    def plot_raw_overall(self, data, output_path, y_limits=None, events=None) -> None:
        self._plot_overall(data, output_path, processed=False, events=events, y_limits=y_limits)

    def plot_processed_overall(self, data, output_path, y_limits=None, events=None) -> None:
        # needs grand_oxy / grand_deoxy columns; without them the axes come out empty
        self._plot_overall(data, output_path, processed=True, events=events, y_limits=y_limits)

    def plot_short_channels(self, data, output_path, y_limits=None, channel_map=None) -> None:
        if self.data_type == "concentration":
            triples = [t for t in self._channel_columns(data) if self._is_short(t[0])]
        else:
            triples = []
            for col in _od_cols(data):
                ch = self._channel_for_od_col(col, channel_map)
                if ch is not None and self._is_short(ch):
                    triples.append((ch, col, None))
        if not triples:
            logger.warning(f"No short channels found for {output_path}")
            return

        time = self._time_axis(len(data))
        fig, axes = self._make_subplots(len(triples))
        for ax, (ch, primary, secondary) in zip(axes, triples):
            _, region = self._channel_attrs(ch)
            if self.data_type == "concentration":
                ax.plot(time, _num(data[primary]), color=RAW_COLORS["oxy"], linewidth=1, label="HbO")
                if secondary and secondary in data.columns:
                    ax.plot(time, _num(data[secondary]), color=RAW_COLORS["deoxy"], linewidth=1, label="HHb")
            else:
                ax.plot(time, _num(data[primary]), color=self._color_for(True, region), linewidth=1, label=primary)
            self._format_axis(ax, f"CH{ch}" + (f" ({region})" if region else ""))

        self._apply_ylimits(axes, y_limits)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Short Channels", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_close(fig, output_path)

    # ---- per-channel -------------------------------------------------------

    def _plot_all_channels_conc(self, data, output_path, y_limits) -> None:
        channels = self._channel_columns(data)
        if not channels:
            logger.warning(f"No channel columns in concentration data for {output_path}")
            return

        time = self._time_axis(len(data))
        fig, axes = self._make_subplots(len(channels))
        for ax, (ch, oxy_col, deoxy_col) in zip(axes, channels):
            ax.plot(time, _num(data[oxy_col]), color=RAW_COLORS["oxy"], linewidth=1, label="HbO")
            if deoxy_col:
                ax.plot(time, _num(data[deoxy_col]), color=RAW_COLORS["deoxy"], linewidth=1, label="HHb")
            self._format_axis(ax, f"CH{ch}")

        self._apply_ylimits(axes, y_limits)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Raw All Channels", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        self._save_close(fig, output_path)

    def _plot_all_channels_od(self, data, output_path, y_limits, channel_map) -> None:
        od_cols = _od_cols(data)
        if not od_cols:
            logger.warning(f"No OD columns for {output_path}")
            return

        if channel_map:
            avail = set(od_cols)
            groups: List[Tuple[Optional[int], List[str]]] = []
            for ch in sorted(channel_map):
                # first available column per wavelength
                cols = []
                for wl in sorted(channel_map[ch].get("columns", {})):
                    hit = next((c for c in channel_map[ch]["columns"][wl] if c in avail), None)
                    if hit:
                        cols.append(hit)
                if cols:
                    groups.append((ch, cols))
        else:
            groups = [(None, [c]) for c in sorted(od_cols)]

        time = self._time_axis(len(data))
        fig, axes = self._make_subplots(len(groups))
        for ax, (ch, cols) in zip(axes, groups):
            is_short, region = self._channel_attrs(ch)
            color = self._color_for(is_short, region)
            for i, col in enumerate(cols):
                ax.plot(time, _num(data[col]), color=color, linewidth=1.0 if i == 0 else 0.7,
                        alpha=1.0 if i == 0 else 0.8, label=col)
            self._format_axis(ax, self._od_title(ch, cols[0], is_short, region))

        self._apply_ylimits(axes, y_limits)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Raw All Channels (OD)", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        self._save_close(fig, output_path)

    # ---- regions -----------------------------------------------------------

    def _plot_concentration_regions(self, data, output_path, processed, y_limits) -> None:
        pairs = self._region_oxy_deoxy_pairs(data)
        if not pairs:
            logger.warning(f"No region columns in data for {output_path}")
            return

        regions = sorted(pairs)
        time = self._time_axis(len(data))
        fig, axes = self._make_subplots(len(regions), height_per=3.0)
        colors = PROCESSED_COLORS if processed else RAW_COLORS
        deoxy_ls = PROCESSED_DEOXY_LS if processed else "-"

        for ax, region in zip(axes, regions):
            oxy_col, deoxy_col = pairs[region]
            ax.plot(time, data[oxy_col], color=colors["oxy"], linewidth=1.5, label=f"{region} HbO")
            ax.plot(time, data[deoxy_col], color=colors["deoxy"], linewidth=1.5,
                    linestyle=deoxy_ls, label=f"{region} HHb")
            self._format_axis(ax, self._region_title(region), fontsize=10, legend_size=9)

        self._apply_ylimits(axes, y_limits)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Processed Regional Averages" if processed else "Raw Regional Averages",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_close(fig, output_path)

    def _plot_od_regions(self, regional_data, output_path, channel_map, y_limits) -> None:
        if channel_map is None:
            raise ValueError("OD region plot requires a channel_map")
        if not isinstance(regional_data, pd.DataFrame):
            raise TypeError("OD region plot requires a DataFrame")

        region_map = build_region_map_od(regional_data.columns.tolist(), channel_map)
        if not region_map:
            logger.warning(f"No regions resolved for OD data at {output_path}")
            return

        regions = sorted(region_map)
        time = self._time_axis(len(regional_data))
        fig, axes = self._make_subplots(len(regions), height_per=3.0)
        wl_colors = [RAW_COLORS["wl1"], RAW_COLORS["wl2"]]

        for ax, region in zip(axes, regions):
            groups = self._group_by_wavelength(region_map[region], regional_data.columns)
            for color, (wl, cols) in zip(wl_colors, sorted(groups.items())):
                ax.plot(time, regional_data[cols].mean(axis=1), color=color, linewidth=1.5, label=f"{region} WL{wl}")
            self._format_axis(ax, self._region_title(region), fontsize=10, legend_size=9)

        self._apply_ylimits(axes, y_limits)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Raw OD Regional Averages", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_close(fig, output_path)

    # ---- overall -----------------------------------------------------------

    def _plot_overall(self, data, output_path, processed, events, y_limits) -> None:
        time = self._time_axis(len(data))
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = PROCESSED_COLORS if processed else RAW_COLORS
        deoxy_ls = PROCESSED_DEOXY_LS if processed else "-"
        prefix = "Processed" if processed else "Raw"

        if self.data_type == "concentration":
            if "grand_oxy" in data.columns:
                ax.plot(time, data["grand_oxy"], color=colors["oxy"], linewidth=1.5, label=f"{prefix} Mean HbO")
            if "grand_deoxy" in data.columns:
                ax.plot(time, data["grand_deoxy"], color=colors["deoxy"], linewidth=1.5,
                        linestyle=deoxy_ls, label=f"{prefix} Mean HHb")
        else:
            wl_cols = [c for c in data.columns if c.startswith("grand_")]
            for i, col in enumerate(wl_cols[:2]):
                ls = deoxy_ls if i == 1 and processed else "-"
                ax.plot(time, data[col], color=colors["wl1" if i == 0 else "wl2"], linewidth=1.5,
                        linestyle=ls, label=f"{prefix} Mean WL{i + 1}")

        if events is not None and not events.empty and "Sample number" in events.columns:
            self._draw_events(ax, events)
        if y_limits:
            ax.set_ylim(y_limits)
        ax.set_title(f"{prefix} Overall Mean")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self.y_label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save_close(fig, output_path)

    # ---- column / channel helpers -----------------------------------------

    @staticmethod
    def _channel_columns(data: pd.DataFrame) -> List[Tuple[int, str, Optional[str]]]:
        """(channel, oxy_col, deoxy_col or None), sorted by channel."""
        out = []
        for col in data.columns:
            m = col.endswith(OXY) and re.match(r"CH(\d+)$", col[: -len(OXY)])
            if m:
                deoxy = col.replace(OXY, DEOXY)
                out.append((int(m.group(1)), col, deoxy if deoxy in data.columns else None))
        return sorted(out, key=lambda x: x[0])

    @staticmethod
    def _region_oxy_deoxy_pairs(data: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
        out = {}
        for col in data.columns:
            if not col.endswith(OXY):
                continue
            name = col[: -len(OXY)]
            if name.startswith("CH") or name == "grand":
                continue
            if f"{name}{DEOXY}" in data.columns:
                out[name] = (col, f"{name}{DEOXY}")
        return out

    @staticmethod
    def _channel_for_od_col(col: str, channel_map: Optional[Dict]) -> Optional[int]:
        for ch, info in (channel_map or {}).items():
            if any(col in cols for cols in info.get("columns", {}).values()):
                return ch
        return None

    @staticmethod
    def _channel_attrs(ch: Optional[int]) -> Tuple[bool, Optional[str]]:
        if ch is None:
            return False, None
        info = CHANNEL_INFO.get(ch)
        if info is None:
            return ch in SHORT_CHANNEL_LIST, None
        return info.channel_type == "SHORT", info.region

    @staticmethod
    def _is_short(ch: Optional[int]) -> bool:
        if ch is None:
            return False
        info = CHANNEL_INFO.get(ch)
        return info.channel_type == "SHORT" if info else ch in SHORT_CHANNEL_LIST

    @staticmethod
    def _color_for(is_short: bool, region: Optional[str]) -> str:
        kind = "PFC" if region == "PFC" else "MOTOR" if region in _MOTOR_REGIONS else None
        if kind is None:
            return CHANNEL_TYPE_COLORS["SHORT" if is_short else "LONG"]
        return CHANNEL_TYPE_COLORS[f"{kind}_{'SHORT' if is_short else 'LONG'}"]

    @staticmethod
    def _od_title(ch, col, is_short, region) -> str:
        if ch is None:
            return col
        return f"CH{ch} - {col}" + (" [SHORT]" if is_short else "") + (f" ({region})" if region else "")

    @staticmethod
    def _region_title(region: str) -> str:
        chs = CH_REGION_MAP.get(region) or CH_REGION_MAP_COMBINED.get(region.split("_")[0], [])
        chs = sorted(set(chs))
        return f"{region}  (CH{', CH'.join(map(str, chs))})" if chs else region

    @staticmethod
    def _group_by_wavelength(cols: List[str], available) -> Dict[str, List[str]]:
        avail = set(available)
        groups: Dict[str, List[str]] = {}
        for col in cols:
            m = re.search(r"WL(\d+)", col)
            if col in avail and m:
                groups.setdefault(m.group(1), []).append(col)
        return groups

    def _draw_events(self, ax, events: pd.DataFrame) -> None:
        ymax = ax.get_ylim()[1]
        for _, row in events.iterrows():
            t = float(row["Sample number"]) / self.fs
            ax.axvline(x=t, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(t, ymax, f' {row.get("Event", "")}', rotation=90, verticalalignment="top",
                    fontsize=8, alpha=0.7)

    # ---- figure helpers ----------------------------------------------------

    def _time_axis(self, length: int) -> np.ndarray:
        return np.arange(length, dtype="float64") / self.fs

    @staticmethod
    def _make_subplots(n: int, height_per: float = 2.0, width: float = 12.0):
        fig, axes = plt.subplots(n, 1, figsize=(width, height_per * n), sharex=True)
        return fig, np.atleast_1d(axes)

    def _format_axis(self, ax, title: str, fontsize: int = 9, legend_size: int = 8) -> None:
        ax.set_title(title, loc="left", fontsize=fontsize, fontweight="bold")
        ax.set_ylabel(self.y_label)
        ax.legend(loc="upper right", fontsize=legend_size)
        ax.grid(True, alpha=0.2)

    @staticmethod
    def _apply_ylimits(axes, y_limits) -> None:
        if y_limits:
            for ax in axes:
                ax.set_ylim(y_limits)

    @staticmethod
    def _save_close(fig, output_path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
