"""fNIRS full-cap processing pipeline.

Order: quality metrics -> TDDR -> OD-to-concentration -> SCR -> bandpass -> baseline.

Quality metrics run on the pre-TDDR signal. TDDR is a motion-correction step
and SQI/SCI/PSP are designed to detect exactly the artifacts it removes, so
scoring them after TDDR inflates every metric.

Channel exclusion is a chain gated behind a precondition, not a flat OR of
independent tests. None of the SD-based steps below even run on a channel
unless it already passes SCI >= 0.75 and PSP >= 0.1 outright -- a channel
that fails that precondition is left entirely to the active criterion.
Among channels that pass it:
    1. flag  if weak vs. the dataset median or that channel's own history
    2/3. discard a flagged channel if it's dead (whole-recording or 5s
         windows) within THIS file
    4. discard a flagged channel that survived 2/3 if SQI <= 1
The active criterion always runs independently on every channel regardless
of the above. See _compute_quality_metrics for the exact order.

Two entry points:
    process_file               full pipeline; active criterion gates channels
                                unless apply_filter=False.
    process_file_quality_only  quality metrics only, no TDDR/SCR/bandpass/baseline.

Per-file output:
    *_processed.csv        long-channel concentration series, three views per
                            region (bilateral / left / right). No whole-cap or
                            whole-hemisphere aggregate (mixes PFC and motor).
    *_quality_report.csv   per-channel SQI/SCI/PSP + the exclusion decision.
    *_channel_quality.pdf  optional per-channel plots (plot_channel_quality=True).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from read.loaders import read_txt_file
from read.channel_config import (
    LONG_TO_SHORT_MAP,
    SCR_PRIORITY_LONGS,
    CH_REGION_MAP,
    CH_REGION_MAP_COMBINED,
    SHORT_CHANNEL_LIST,
)

from preprocessing.od_to_concentration import convert_od_to_concentration
from preprocessing.butterworth_filter import butterworth_bandpass
from preprocessing.short_channel_regression import scr_regression
from preprocessing.tddr import tddr
from preprocessing.baseline_correction import baseline_subtraction
from preprocessing.average_channels import FullCapChannelAverager

from channel_quality.signalqualityindex import SQI
from channel_quality.sci import scalp_coupling_index, cardiac_band_sd
from channel_quality.psp import peak_spectral_power
from channel_quality.exclusion import ExclusionCriterion, make_criterion

from viz.visualizer import FNIRSVisualizer
from viz.channel_quality_plots import plot_channel_quality_pdf

logger = logging.getLogger(__name__)

OXY = "_oxy"
DEOXY = "_deoxy"

# Zero-phase Butterworth bandpass (butterworth_filter.py). A windowed-sinc
# FIR previously lived here but couldn't form a clean edge at 0.01 Hz within
# these sample counts -- see butterworth_filter.py's docstring for the
# measured frequency response that showed this. _BUTTERWORTH_ORDER is a
# pole count (2-4 typical), NOT a tap count -- do not set this to FIR-era
# values in the hundreds/thousands, that will not stay numerically stable.
_BANDPASS_BAND = (0.01, 0.1)
_BUTTERWORTH_ORDER = 4

_DEFAULT_BASELINE_S = 20.0
_BASELINE_START_SAMPLE = 4
_DROP_INITIAL_S = 1.0

# Report columns, in report order. CardiacSD_* are OD-wavelength diagnostics
# that feed step 1's flagging decision below; they don't gate on their own.
_METRIC_COLS: Tuple[str, ...] = ("SQI", "SCI", "PSP", "CardiacSD_WL1", "CardiacSD_WL2", "CardiacSD")

# Precondition (gate 0): none of steps 1-4 below run on a channel unless it
# already clears SCI/PSP outright. A channel that fails this is left entirely
# to the active criterion -- the SD-based chain never touches it.
_SCI_MIN: float = 0.75
_PSP_MIN: float = 0.1

# Step 1: flag (not exclude) a channel that clears the precondition but has
# suspiciously weak cardiac-band amplitude. Either condition is sufficient:
#   (a) below _FLAG_DATASET_RATIO x the whole dataset's median CardiacSD
#   (b) below _FLAG_CHANNEL_RATIO x that channel's own median CardiacSD
#       across the whole dataset (catches a channel that's chronically weak
#       at that specific position, even if the dataset overall is fine)
# Both reference numbers are corpus-wide and this single-file pipeline can't
# compute them on its own -- they're supplied via dataset_median_cardiac_sd /
# channel_median_cardiac_sd on FullCapProcessor (e.g. produced by a separate
# pass over already-processed quality reports). Step 1 simply never flags
# anything if they're not provided.
_FLAG_DATASET_RATIO: float = 0.05
_FLAG_CHANNEL_RATIO: float = 0.20

# Steps 2 & 3: ONLY evaluated on a channel already flagged by step 1. Both
# ask the same question -- is the livelier chromophore (max(SD_oxy, SD_deoxy))
# dead relative to this recording's median liveliness -- just over different
# windows: step 2 the whole recording, step 3 each 5s window (discarding if
# >= half the windows are dead). The ratio is deliberately much stricter than
# a standalone flatline check would use (0.01%, not the more common ~0.1%)
# precisely because it's gated behind flagging already -- run unconditionally
# across every channel, this ratio would be too aggressive.
_DEAD_RATIO: float = 1e-4
_DEAD_WINDOW_S: float = 5.0
_DEAD_WINDOW_FRACTION: float = 0.5

# Step 4: the last step in the chain, and the only channels that ever reach
# it are ones flagged by step 1 that survived steps 2 and 3. SQI <= this
# value discards them; otherwise they're kept despite the flag.
_HARD_SQI_DISCARD: Optional[float] = 1.0
# Public alias so callers (batch.py, main.py) can default to the same value
# without hardcoding a second copy of it that could silently drift out of
# sync with this one.
DEFAULT_HARD_SQI_DISCARD: Optional[float] = _HARD_SQI_DISCARD

# Methods always shown on the per-channel quality plot; also the badge/column
# order on the summary heatmap.
_COMPARISON_METHODS: Tuple[str, ...] = ("sqi", "sci", "psp", "sci_psp")
# Public alias -- batch.py imports this name directly (same reason as
# DEFAULT_HARD_SQI_DISCARD above).
COMPARISON_METHODS: Tuple[str, ...] = _COMPARISON_METHODS


def _to_oxy_deoxy(df: pd.DataFrame) -> pd.DataFrame:
    """'CH{n} HbO' -> 'CH{n}_oxy', 'CH{n} HHb' -> 'CH{n}_deoxy'."""
    return df.rename(columns=lambda c: c.replace(" HbO", OXY).replace(" HHb", DEOXY))


def _windowed_liveliness(oxy: np.ndarray, deoxy: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Per-window max(std_oxy, std_deoxy) over non-overlapping windows.

    Drops a short trailing remainder rather than padding it, so a partial
    final window doesn't get scored as artificially dead.
    """
    win = max(1, int(window_s * fs))
    n = len(oxy)
    if n < win:
        return np.array([])
    n_windows = n // win
    return np.array([
        max(np.std(oxy[i * win:(i + 1) * win]), np.std(deoxy[i * win:(i + 1) * win]))
        for i in range(n_windows)
    ])


class FullCapProcessor:
    def __init__(
        self,
        fs: float = 50.0,
        criterion: Optional[ExclusionCriterion] = None,
        plot_channel_quality: bool = False,
        comparison_criteria: Optional[Sequence[ExclusionCriterion]] = None,
        skip_plots: bool = False,
        hard_sqi_discard: Optional[float] = _HARD_SQI_DISCARD,
        dataset_median_cardiac_sd: Optional[float] = None,
        channel_median_cardiac_sd: Optional[Dict[int, float]] = None,
    ):
        self.fs = fs
        self.criterion = criterion or make_criterion("sqi")
        self.plot_channel_quality = plot_channel_quality
        self.comparison_criteria: List[ExclusionCriterion] = (
            list(comparison_criteria) if comparison_criteria is not None
            else [make_criterion(m) for m in _COMPARISON_METHODS]
        )
        self.skip_plots = skip_plots
        self.hard_sqi_discard = hard_sqi_discard
        # Gate A's two corpus-wide reference numbers. Both come from outside
        # this file -- there's no way to compute a "whole dataset" statistic
        # from inside a per-file pipeline. Gate A is inert (never flags
        # anything) if these aren't supplied.
        self.dataset_median_cardiac_sd = dataset_median_cardiac_sd
        self.channel_median_cardiac_sd: Dict[int, float] = channel_median_cardiac_sd or {}
        self.viz_od = FNIRSVisualizer(fs=fs, data_type="od")
        self.viz_conc = FNIRSVisualizer(fs=fs, data_type="concentration")
        self.averager = FullCapChannelAverager()
        self.warning_files: List[Tuple[str, str]] = []

    # ---- entry points -------------------------------------------------

    def process_file(
        self,
        file_path: str,
        output_base_dir: str,
        input_base_dir: str,
        apply_filter: bool = True,
        plot_raw: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Run the full pipeline on one file.

        With apply_filter=True the active criterion (plus the hard gates
        below it) drops failing channels before SCR/bandpass/baseline, and output
        goes to with_<method>_filtering/. With apply_filter=False nothing is
        dropped and output goes to no_filtering/.
        """
        try:
            fp = Path(file_path)
            out_dir = self._make_output_dir(output_base_dir, input_base_dir, fp, apply_filter)
            basename = f"{fp.stem}_{'filtered' if apply_filter else 'unfiltered'}"

            loaded = read_txt_file(fp)
            df_od, events, metadata, channel_map = (
                loaded["data"], loaded["events"], loaded["metadata"], loaded["channel_map"]
            )
            df_od, events = self._drop_initial_seconds(df_od, events)

            if plot_raw and not self.skip_plots:
                self._plot_raw(df_od, channel_map, metadata, out_dir, basename, events)

            df_conc, quality = self._run_pipeline(df_od, events, metadata, channel_map, apply_filter)
            if df_conc is None:
                return None

            df_final = self._add_aggregates(df_conc)
            if not self.skip_plots:
                self._plot_processed(df_final, out_dir, basename, events)
            df_final = self._inject_events(df_final, events)

            df_final.to_csv(out_dir / f"{basename}_processed.csv", index=False)
            self._save_quality_report(quality, out_dir, basename, applied=apply_filter)

            if self.plot_channel_quality and quality:
                conc_pre = self._pre_tddr_concentration(df_od, channel_map, metadata, events)
                if conc_pre is not None:
                    self._plot_channel_quality(df_od, conc_pre, channel_map, quality, out_dir, basename, events)
            return df_final

        except Exception as exc:
            logger.error(f"Failed to process {file_path}: {exc}", exc_info=True)
            self.warning_files.append((str(file_path), str(exc)))
            return None

    def process_file_quality_only(
        self, file_path: str, output_base_dir: str, input_base_dir: str,
    ) -> Optional[Dict]:
        """Quality metrics only, pre-TDDR, no motion correction / SCR / bandpass / baseline."""
        try:
            fp = Path(file_path)
            out_dir = self._make_quality_only_output_dir(output_base_dir, input_base_dir, fp)
            basename = fp.stem

            loaded = read_txt_file(fp)
            df_od, events, metadata, channel_map = (
                loaded["data"], loaded["events"], loaded["metadata"], loaded["channel_map"]
            )
            df_od, events = self._drop_initial_seconds(df_od, events)

            od_only = df_od.filter(regex="WL")
            if od_only.empty:
                logger.error("No wavelength columns in input; aborting quality assessment")
                return None

            conc = _to_oxy_deoxy(convert_od_to_concentration(
                od_only, channel_map=channel_map, metadata=metadata, events=events, fs=self.fs,
            ))
            quality = self._compute_quality_metrics(df_od, conc, channel_map)
            self._save_quality_report(quality, out_dir, basename, applied=False)
            if self.plot_channel_quality:
                self._plot_channel_quality(df_od, conc, channel_map, quality, out_dir, basename, events)
            return quality

        except Exception as exc:
            logger.error(f"Failed quality assessment for {file_path}: {exc}", exc_info=True)
            self.warning_files.append((str(file_path), str(exc)))
            return None

    # ---- pipeline -------------------------------------------------------

    def _run_pipeline(
        self, df_od: pd.DataFrame, events: pd.DataFrame, metadata: Dict, channel_map: Dict, apply_filter: bool,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        od_pre = df_od.filter(regex="WL")
        if od_pre.empty:
            logger.error("No wavelength columns in input; aborting pipeline")
            return None, None
        conc_pre = _to_oxy_deoxy(convert_od_to_concentration(
            od_pre, channel_map=channel_map, metadata=metadata, events=events, fs=self.fs,
        ))
        quality = self._compute_quality_metrics(df_od, conc_pre, channel_map)

        od_corrected = self._apply_tddr(df_od)
        od_only = od_corrected.filter(regex="WL")
        if od_only.empty:
            logger.error("No wavelength columns after TDDR; aborting pipeline")
            return None, None
        conc = _to_oxy_deoxy(convert_od_to_concentration(
            od_only, channel_map=channel_map, metadata=metadata, events=events, fs=self.fs,
        ))

        if apply_filter and quality["excluded_columns"]:
            logger.info(
                f"{self.criterion.name}: excluding {len(quality['excluded_channels'])} channels "
                f"({len(quality['excluded_columns'])} columns) - {self.criterion.description}"
            )
            conc = conc.drop(columns=quality["excluded_columns"], errors="ignore")
        elif not apply_filter:
            logger.info(
                f"Channel exclusion disabled (would have excluded "
                f"{len(quality['excluded_channels'])} channels under {self.criterion.description})"
            )

        conc = self._apply_scr(conc, quality["excluded_channels"], apply_filter)

        filtered = pd.DataFrame(
            butterworth_bandpass(conc, order=_BUTTERWORTH_ORDER, Wn=list(_BANDPASS_BAND), fs=int(self.fs)),
            columns=conc.columns, index=conc.index,
        )
        return self._apply_baseline_correction(filtered, events), quality

    def _apply_tddr(self, df_od: pd.DataFrame) -> pd.DataFrame:
        signals = df_od.filter(regex="WL|Sample")
        corrected = tddr(signals, sample_rate=self.fs)
        for c in df_od.columns:
            if c not in corrected.columns:
                corrected[c] = df_od[c]
        return corrected

    # ---- quality metrics --------------------------------------------------

    def _compute_quality_metrics(self, df_od: pd.DataFrame, df_conc: pd.DataFrame, channel_map: Dict) -> Dict:
        """SQI/SCI/PSP/CardiacSD per channel, then a precondition-gated chain.

        Gate 0: a channel must clear SCI >= 0.75 and PSP >= 0.1 outright, or
        none of steps 1-4 below apply to it (it's left to the active
        criterion only). Among channels that clear gate 0:
            1. flag if weak vs. the dataset median or the channel's own
               history (flagging alone excludes nothing)
            2/3. discard a flagged channel that's dead within this file,
               whole-recording or in >=50% of 5s windows
            4. discard a flagged channel that survived 2/3 if SQI <= the
               hard floor
        The active criterion always runs independently on every channel.
        """
        metrics: List[Dict] = []
        ch_cols: Dict[int, Tuple[str, str]] = {}
        ch_windows: Dict[int, np.ndarray] = {}

        for ch_idx, info in channel_map.items():
            wls = sorted(info["columns"].keys())
            if len(wls) != 2:
                continue
            wl1, wl2 = wls
            cols_wl1 = info["columns"].get(wl1) or []
            cols_wl2 = info["columns"].get(wl2) or []
            if not cols_wl1 or not cols_wl2:
                continue

            od1_col, od2_col = cols_wl1[0], cols_wl2[0]
            oxy_col, deoxy_col = f"CH{ch_idx}{OXY}", f"CH{ch_idx}{DEOXY}"
            if (od1_col not in df_od.columns or od2_col not in df_od.columns
                    or oxy_col not in df_conc.columns or deoxy_col not in df_conc.columns):
                continue
            ch_cols[ch_idx] = (oxy_col, deoxy_col)

            od1, od2 = df_od[od1_col].values, df_od[od2_col].values
            oxy, deoxy = df_conc[oxy_col].values, df_conc[deoxy_col].values

            if not (np.all(np.isfinite(od1)) and np.all(np.isfinite(od2))
                    and np.all(np.isfinite(oxy)) and np.all(np.isfinite(deoxy))):
                metrics.append(self._nan_metric_row(ch_idx, status="fail"))
                continue

            try:
                sqi_val = float(SQI(od1, od2, oxy, deoxy, Fs=self.fs))
            except Exception as e:
                logger.warning(f"SQI failed on channel {ch_idx}: {e}")
                metrics.append(self._nan_metric_row(ch_idx, status="fail"))
                continue

            sd_wl1, sd_wl2 = cardiac_band_sd(od1, od2, fs=self.fs)
            metrics.append({
                "channel": ch_idx,
                "SQI": sqi_val,
                "SCI": scalp_coupling_index(od1, od2, fs=self.fs),
                "PSP": peak_spectral_power(od1, od2, fs=self.fs),
                "CardiacSD_WL1": sd_wl1,
                "CardiacSD_WL2": sd_wl2,
                "CardiacSD": (float("nan") if (np.isnan(sd_wl1) and np.isnan(sd_wl2))
                              else float(np.nanmax([sd_wl1, sd_wl2]))),
                "SD_oxy": float(np.std(oxy)),
                "SD_deoxy": float(np.std(deoxy)),
            })
            ch_windows[ch_idx] = _windowed_liveliness(oxy, deoxy, self.fs, _DEAD_WINDOW_S)

        dead_threshold = self._dead_threshold(metrics)
        bad_cols: set = set()
        bad_chs: set = set()

        for m in metrics:
            # Stage-1 auto-fails (non-finite input, SQI exception) already
            # carry status="fail" and have no SD_oxy/SCI/etc. to evaluate --
            # preserve that instead of letting the chain below re-derive a
            # (wrong) pass from missing data.
            pre_fail = m.get("status") == "fail"

            # Gate 0 (precondition): none of steps 1-4 run unless the channel
            # already clears SCI/PSP outright. A channel that fails this is
            # left entirely to the active criterion below.
            sci_val, psp_val, cardiac_sd = m.get("SCI"), m.get("PSP"), m.get("CardiacSD")
            sci_psp_pass = (
                isinstance(sci_val, (int, float)) and np.isfinite(sci_val) and sci_val >= _SCI_MIN
                and isinstance(psp_val, (int, float)) and np.isfinite(psp_val) and psp_val >= _PSP_MIN
            )
            m["SciPspPass"] = sci_psp_pass

            flagged = dead_step2 = dead_step3 = hard_sqi_fail = False
            frac = float("nan")

            if sci_psp_pass:
                # Step 1: flag on weak cardiac amplitude vs. the dataset or
                # this channel's own history. Flagging alone excludes nothing.
                have_cardiac_sd = isinstance(cardiac_sd, (int, float)) and np.isfinite(cardiac_sd)
                weak_vs_dataset = (
                    have_cardiac_sd and self.dataset_median_cardiac_sd is not None
                    and cardiac_sd < _FLAG_DATASET_RATIO * self.dataset_median_cardiac_sd
                )
                own_median = self.channel_median_cardiac_sd.get(m["channel"])
                weak_vs_own_history = (
                    have_cardiac_sd and own_median is not None and np.isfinite(own_median)
                    and cardiac_sd < _FLAG_CHANNEL_RATIO * own_median
                )
                flagged = bool(weak_vs_dataset or weak_vs_own_history)

                if flagged:
                    # Steps 2 & 3: only evaluated on a flagged channel. Both
                    # ask "is the livelier chromophore dead relative to this
                    # file's median liveliness," just over different windows.
                    o, d = m.get("SD_oxy"), m.get("SD_deoxy")
                    have_sd = (isinstance(o, (int, float)) and isinstance(d, (int, float))
                               and np.isfinite(o) and np.isfinite(d))
                    liveliness = max(o, d) if have_sd else float("nan")
                    dead_step2 = (not have_sd) or (dead_threshold is not None and liveliness < dead_threshold)

                    windows = ch_windows.get(m["channel"])
                    if windows is not None and windows.size and dead_threshold is not None:
                        frac = float(np.mean(windows < dead_threshold))
                        dead_step3 = frac >= _DEAD_WINDOW_FRACTION
                    else:
                        dead_step3 = False

                    # Step 4: the tiebreaker. Only reached by a flagged
                    # channel that survived steps 2 and 3.
                    if not (dead_step2 or dead_step3):
                        sqi_val = m.get("SQI")
                        hard_sqi_fail = (
                            self.hard_sqi_discard is not None
                            and isinstance(sqi_val, (int, float))
                            and np.isfinite(sqi_val)
                            and sqi_val <= self.hard_sqi_discard
                        )

            m["GateAFlagged"] = flagged
            m["Flatlined"] = bool(dead_step2)
            m["PartialDropout"] = bool(dead_step3)
            m["PartialDropoutFraction"] = frac
            m["HardSQIDiscard"] = bool(hard_sqi_fail)

            # Stored separately from the combined status below so each gate
            # can be audited independently -- without this, an OR of several
            # booleans collapses into one Status and you can't tell which
            # gate(s) actually fired for a given channel. The active
            # criterion always runs, regardless of the precondition/flag
            # chain above.
            criterion_fail = bool(self.criterion.excludes(m))
            m["CriterionExcluded"] = criterion_fail

            chain_fail = flagged and (dead_step2 or dead_step3 or hard_sqi_fail)
            fail = pre_fail or chain_fail or criterion_fail
            m["status"] = "fail" if fail else "pass"
            if fail:
                oxy_col, deoxy_col = ch_cols.get(m["channel"], (None, None))
                if oxy_col is not None:
                    bad_cols.update([oxy_col, deoxy_col])
                bad_chs.add(m["channel"])

        self._log_quality_summary(metrics)

        return {
            "metrics": metrics,
            "excluded_columns": sorted(bad_cols),
            "excluded_channels": sorted(bad_chs),
            "num_passed": sum(1 for m in metrics if m["status"] == "pass"),
            "num_failed": sum(1 for m in metrics if m["status"] == "fail"),
        }

    def _log_quality_summary(self, metrics: List[Dict]) -> None:
        if not metrics:
            return
        passed = sum(1 for m in metrics if m["status"] == "pass")
        n = len(metrics)
        n_sci_psp = sum(1 for m in metrics if m.get("SciPspPass"))
        n_gate_a = sum(1 for m in metrics if m.get("GateAFlagged"))
        n_flat = sum(1 for m in metrics if m.get("Flatlined"))
        n_partial = sum(1 for m in metrics if m.get("PartialDropout"))
        n_hard_sqi = sum(1 for m in metrics if m.get("HardSQIDiscard"))
        stats = {}
        for col in _METRIC_COLS:
            vals = [m[col] for m in metrics if not (m[col] is None or (isinstance(m[col], float) and np.isnan(m[col])))]
            if vals:
                stats[col] = (min(vals), float(np.mean(vals)), max(vals))
        bits = [f"{passed}/{n} pass under {self.criterion.description}",
                f"sci_psp_pass={n_sci_psp}/{n} (>= {_SCI_MIN} SCI, >= {_PSP_MIN} PSP -- gate 0)",
                f"gate_a_flagged={n_gate_a}"
                + (" (no dataset baseline supplied)" if self.dataset_median_cardiac_sd is None
                   and not self.channel_median_cardiac_sd else ""),
                f"dead_whole_file={n_flat}",
                f"dead_windowed={n_partial}"
                + (f" (>={_DEAD_WINDOW_FRACTION:.0%} of {_DEAD_WINDOW_S:g}s windows)"
                   if _DEAD_WINDOW_FRACTION > 0 else " (off)"),
                f"hard_sqi_discard={n_hard_sqi}"
                + (f" (threshold<={self.hard_sqi_discard})" if self.hard_sqi_discard is not None else " (off)")]
        for col, (lo, mean, hi) in stats.items():
            bits.append(f"{col}=[{lo:.3g}, {mean:.3g}, {hi:.3g}]")
        logger.info("Quality: " + "; ".join(bits))

    @staticmethod
    def _nan_metric_row(channel: int, status: str) -> Dict:
        return {"channel": channel, **{c: float("nan") for c in _METRIC_COLS}, "status": status}

    @staticmethod
    def _dead_threshold(metrics: List[Dict]) -> Optional[float]:
        """_DEAD_RATIO x median liveliness for this recording, or None if
        nothing usable. Liveliness = max(SD_oxy, SD_deoxy), computed only
        over channels with finite SDs so dead channels don't drag the
        reference down. Deliberately a much stricter ratio than a standalone
        flatline check would use -- see the _DEAD_RATIO comment above.
        """
        if _DEAD_RATIO <= 0:
            return None
        live = [max(m["SD_oxy"], m["SD_deoxy"]) for m in metrics
                if isinstance(m.get("SD_oxy"), (int, float))
                and isinstance(m.get("SD_deoxy"), (int, float))
                and np.isfinite(m["SD_oxy"]) and np.isfinite(m["SD_deoxy"])]
        live = [v for v in live if v > 0]
        return _DEAD_RATIO * float(np.median(live)) if live else None

    def _save_quality_report(self, quality: Optional[Dict], out_dir: Path, basename: str, applied: bool) -> None:
        if not quality or not quality["metrics"]:
            return

        rows = [
            {
                "Channel": m["channel"],
                **{c: m[c] for c in _METRIC_COLS},
                "Status": m["status"],
                "SciPspPass": bool(m.get("SciPspPass", False)),
                "GateAFlagged": bool(m.get("GateAFlagged", False)),
                "Flatlined": bool(m.get("Flatlined", False)),
                "PartialDropout": bool(m.get("PartialDropout", False)),
                "PartialDropoutFraction": m.get("PartialDropoutFraction", float("nan")),
                "HardSQIDiscard": bool(m.get("HardSQIDiscard", False)),
                "CriterionExcluded": bool(m.get("CriterionExcluded", False)),
                "SD_oxy": m.get("SD_oxy", float("nan")),
                "SD_deoxy": m.get("SD_deoxy", float("nan")),
                "Excluded": (m["channel"] in quality["excluded_channels"]) if applied else False,
                "Method": self.criterion.name,
                "Criterion": self.criterion.description,
            }
            for m in quality["metrics"]
        ]
        df = pd.DataFrame(rows).sort_values("Channel")

        summary = {"Channel": "SUMMARY"}
        for col in _METRIC_COLS:
            vals = [m[col] for m in quality["metrics"]]
            summary[col] = float(np.nanmean(vals)) if vals else float("nan")
        summary["Status"] = f"pass={quality['num_passed']} fail={quality['num_failed']}"
        summary["SciPspPass"] = sum(1 for m in quality["metrics"] if m.get("SciPspPass"))
        summary["GateAFlagged"] = sum(1 for m in quality["metrics"] if m.get("GateAFlagged"))
        summary["Flatlined"] = sum(1 for m in quality["metrics"] if m.get("Flatlined"))
        summary["PartialDropout"] = sum(1 for m in quality["metrics"] if m.get("PartialDropout"))
        partial_vals = [m["PartialDropoutFraction"] for m in quality["metrics"]]
        summary["PartialDropoutFraction"] = float(np.nanmean(partial_vals)) if partial_vals else float("nan")
        summary["HardSQIDiscard"] = sum(1 for m in quality["metrics"] if m.get("HardSQIDiscard"))
        summary["CriterionExcluded"] = sum(1 for m in quality["metrics"] if m.get("CriterionExcluded"))
        summary["SD_oxy"] = float("nan")
        summary["SD_deoxy"] = float("nan")
        summary["Excluded"] = f"applied ({len(quality['excluded_channels'])} excluded)" if applied else "not applied"
        summary["Method"] = self.criterion.name
        summary["Criterion"] = self.criterion.description

        path = out_dir / f"{basename}_quality_report.csv"
        pd.concat([df, pd.DataFrame([summary])], ignore_index=True).to_csv(path, index=False)

    # ---- regression / filtering / baseline --------------------------------

    def _apply_scr(self, conc: pd.DataFrame, excluded_channels: List[int], filter_applied: bool) -> pd.DataFrame:
        """Regress each long channel against its mapped short(s).

        PFC longs (SCR_PRIORITY_LONGS) use ONE short at a time: the
        ipsilateral OctaMon short, falling back to the contralateral short
        only if the ipsilateral one failed the active quality criterion.
        They're never pooled together -- averaging a right-PFC superficial
        signal into a left-PFC long's regressor (or vice versa) isn't a
        meaningful correction.

        Motor longs (SMA/M1/S1) pool whichever of the four Brite24 shorts
        passed quality into one averaged regressor, so one bad short doesn't
        knock out SCR for every long that borrows from it. V1 longs aren't in
        the map and pass through unchanged (no Brite24 short is close enough
        to be a meaningful reference).

        A short that failed the active criterion is never used -- dropped
        both from the PFC priority pick and the motor pool. A long is skipped
        only if every one of its short refs failed (nothing left to regress
        against).
        """
        result = conc.copy()
        excluded = set(excluded_channels) if filter_applied else set()
        applied = fallback = skipped_excluded = skipped_missing = 0

        for long_ch, short_refs in LONG_TO_SHORT_MAP.items():
            if long_ch in excluded:
                skipped_excluded += 1
                continue

            usable = [s for s in short_refs if s not in excluded]
            if not usable:
                skipped_excluded += 1
                continue

            if long_ch in SCR_PRIORITY_LONGS:
                # PFC: take the first still-usable short in priority order
                # (ipsilateral, then contralateral) -- never averaged.
                chosen = [usable[0]]
                if usable[0] != short_refs[0]:
                    fallback += 1
            else:
                # Motor: pool everything that passed quality.
                chosen = usable

            long_cols = [f"CH{long_ch}{OXY}", f"CH{long_ch}{DEOXY}"]
            short_oxy = [f"CH{s}{OXY}" for s in chosen]
            short_deoxy = [f"CH{s}{DEOXY}" for s in chosen]
            if not all(c in result.columns for c in long_cols + short_oxy + short_deoxy):
                skipped_missing += 1
                continue

            regressor = pd.DataFrame({
                f"CH_shortmean{OXY}": result[short_oxy].mean(axis=1),
                f"CH_shortmean{DEOXY}": result[short_deoxy].mean(axis=1),
            }, index=result.index)

            result.loc[:, long_cols] = np.asarray(scr_regression(result[long_cols], regressor))
            applied += 1

        logger.info(
            f"SCR: applied to {applied}/{len(LONG_TO_SHORT_MAP)} long channels "
            f"({fallback} PFC longs used their contralateral fallback short; "
            f"skipped {skipped_excluded} excluded, {skipped_missing} missing columns)"
        )
        return result

    def _apply_baseline_correction(
        self, df: pd.DataFrame, events: pd.DataFrame, baseline_duration: float = _DEFAULT_BASELINE_S,
    ) -> pd.DataFrame:
        """Baseline window priority: S1->W1 (the standing pre-walk baseline;
        S1 marks its START, W1 marks its END/task start) -> S1->S2 -> S1
        alone -> Task*Start/Baseline*End (these DO mark the end of baseline
        directly, unlike bare S1) -> first baseline_duration seconds if
        nothing matches. Bare S1 must never be treated as an end-of-baseline
        marker the way Task*Start/Baseline*End are -- doing so silently
        references to the device warm-up/setup period before S1, not the
        real standing baseline between S1 and W1 (this was a real, verified
        bug: every file was referencing to [4, S1) instead of [S1, W1)).
        """
        max_end = int(baseline_duration * self.fs)
        min_span = max(1, int(1.0 * self.fs))
        start, end = _BASELINE_START_SAMPLE, max_end
        matched_desc = f"no marker; first {baseline_duration}s"

        def _clamp(a, b):
            a = int(max(0, a))
            b = int(min(b, len(df)))
            return (a, b) if b - a >= min_span else None

        if not events.empty and "Event" in events.columns:
            ev = events.copy()
            ev["Event"] = ev["Event"].astype(str).str.strip()
            ev["Sample number"] = pd.to_numeric(ev["Sample number"], errors="coerce")
            ev = ev.dropna(subset=["Sample number"]).sort_values("Sample number")

            s1 = ev[ev["Event"].str.match(r"^S1$", case=False, na=False)]
            resolved = None
            if not s1.empty:
                s1_pos = s1.iloc[0]["Sample number"]
                w1 = ev[ev["Event"].str.match(r"^W1$", case=False, na=False) & (ev["Sample number"] > s1_pos)]
                s2 = ev[ev["Event"].str.match(r"^S2$", case=False, na=False) & (ev["Sample number"] > s1_pos)]
                if not w1.empty and (w := _clamp(s1_pos, w1.iloc[0]["Sample number"])):
                    resolved, matched_desc = w, f"S1->W1 ({s1_pos}-{int(w1.iloc[0]['Sample number'])})"
                elif not s2.empty and (w := _clamp(s1_pos, s2.iloc[0]["Sample number"])):
                    resolved, matched_desc = w, f"S1->S2 ({s1_pos}-{int(s2.iloc[0]['Sample number'])})"
                elif (w := _clamp(s1_pos, s1_pos + max_end)):
                    resolved, matched_desc = w, f"S1 + {baseline_duration}s ({s1_pos}-{s1_pos + max_end})"

            if resolved is None:
                task_end = ev[ev["Event"].str.match(r"^Task.*Start$|^Baseline.*End$", case=False, na=False)]
                if not task_end.empty:
                    marker_pos = int(task_end.iloc[0]["Sample number"])
                    if w := _clamp(_BASELINE_START_SAMPLE, min(marker_pos - 1, max_end)):
                        resolved = w
                        matched_desc = f"marker '{task_end.iloc[0]['Event']}' at {marker_pos}"

            if resolved is not None:
                start, end = resolved

        logger.info(f"Baseline: samples {start}-{end} ({matched_desc})")

        bdf = pd.DataFrame({"Sample number": [start, end], "Event": ["BaselineStart", "BaselineEnd"]})
        try:
            return baseline_subtraction(df, bdf)
        except Exception as e:
            logger.warning(f"Baseline correction failed: {e}")
            return df

    def _add_aggregates(self, conc: pd.DataFrame) -> pd.DataFrame:
        """Regional and per-hemisphere regional averages, long channels only.

        Short channels are SCR references, not part of the cortical response,
        so they're dropped before averaging. Each region is computed straight
        from the underlying long-channel series, never from already-averaged
        columns. Output carries three views per region (bilateral, left,
        right) x oxy/deoxy. No whole-cap or whole-hemisphere aggregate --
        averaging PFC with motor channels isn't a meaningful quantity.
        """
        short_cols = {f"CH{ch}{sfx}" for ch in SHORT_CHANNEL_LIST for sfx in (OXY, DEOXY)}
        long_only = conc.drop(columns=[c for c in short_cols if c in conc.columns])

        out = pd.DataFrame(index=conc.index)
        for region, channels in CH_REGION_MAP_COMBINED.items():
            self._add_region_mean(out, long_only, region, channels)
        for region, channels in CH_REGION_MAP.items():
            self._add_region_mean(out, long_only, region, channels)
        return out

    @staticmethod
    def _add_region_mean(out: pd.DataFrame, long_only: pd.DataFrame, region: str, channels: List[int]) -> None:
        for sfx in (OXY, DEOXY):
            cols = [f"CH{ch}{sfx}" for ch in channels if f"CH{ch}{sfx}" in long_only.columns]
            if cols:
                out[f"{region}{sfx}"] = long_only[cols].mean(axis=1)

    # ---- I/O helpers --------------------------------------------------------

    def _pre_tddr_concentration(
        self, df_od: pd.DataFrame, channel_map: Dict, metadata: Dict, events: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        od_pre = df_od.filter(regex="WL")
        if od_pre.empty:
            return None
        return _to_oxy_deoxy(convert_od_to_concentration(
            od_pre, channel_map=channel_map, metadata=metadata, events=events, fs=self.fs,
        ))

    def _plot_channel_quality(
        self, df_od, df_conc, channel_map, quality, out_dir: Path, basename: str, events=None,
    ) -> None:
        self._safe("Per-channel quality plot", lambda: plot_channel_quality_pdf(
            df_od=df_od, df_conc=df_conc, channel_map=channel_map, metrics=quality["metrics"],
            criteria=self.comparison_criteria, fs=self.fs,
            output_path=out_dir / f"{basename}_channel_quality.pdf", title=basename, events=events,
        ))

    def _make_output_dir(self, output_base: str, input_base: str, file_path: Path, apply_filter: bool) -> Path:
        rel = file_path.parent.relative_to(input_base)
        sub = f"with_{self.criterion.method}_filtering" if apply_filter else "no_filtering"
        out = Path(output_base) / rel / sub
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _make_quality_only_output_dir(self, output_base: str, input_base: str, file_path: Path) -> Path:
        rel = file_path.parent.relative_to(input_base)
        out = Path(output_base) / rel / "channel_quality"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _drop_initial_seconds(
        self, df: pd.DataFrame, events: pd.DataFrame, seconds: float = _DROP_INITIAL_S,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n = int(seconds * self.fs)
        if len(df) <= n:
            return df, events
        df = df.iloc[n:].reset_index(drop=True)
        df["Sample number"] = np.arange(len(df))
        if not events.empty:
            events = events[events["Sample number"] >= n].copy()
            events["Sample number"] -= n
        return df, events

    def _inject_events(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        if "Sample number" not in df.columns:
            df.insert(0, "Sample number", np.arange(len(df)))
        df["Event"] = ""
        if events.empty or "Sample number" not in events.columns:
            return df
        for _, row in events.iterrows():
            i = int(row["Sample number"])
            if 0 <= i < len(df):
                existing = df.at[i, "Event"]
                df.at[i, "Event"] = f"{existing};{row['Event']}" if existing else str(row["Event"])
        logger.info(f"Mapped {len(events)} event markers into output CSV")
        return df

    # ---- plotting -------------------------------------------------------

    def _safe(self, label: str, fn: Callable[[], None]) -> None:
        """Run a plot call, logging and swallowing failure instead of aborting
        the rest of the file's plots.
        """
        try:
            fn()
        except Exception as e:
            logger.warning(f"{label} failed: {e}")

    def _plot_raw(self, df_od, channel_map, metadata, out_dir: Path, basename: str, events) -> None:
        self._safe("Raw OD plot", lambda: self.viz_od.plot_raw_od(
            data=df_od, output_path=str(out_dir / f"{basename}_raw_OD.pdf"), y_limits=None,
        ))

        try:
            od_only = df_od.filter(regex="WL")
            conc = _to_oxy_deoxy(convert_od_to_concentration(
                od_only, channel_map=channel_map, metadata=metadata, events=events, fs=self.fs,
            ))
        except Exception as e:
            logger.warning(f"Raw concentration conversion failed: {e}")
            return

        self._safe("Raw all-channels plot", lambda: self.viz_conc.plot_raw_all_channels(
            data=conc, output_path=str(out_dir / f"{basename}_raw_all_channels.pdf"), y_limits=None,
        ))
        self._safe("Raw regions plot", lambda: self.viz_conc.plot_raw_regions(
            regional_data=_to_oxy_deoxy(self.averager.average_regions(conc)),
            output_path=str(out_dir / f"{basename}_raw_regions.pdf"), y_limits=None, channel_map=channel_map,
        ))

        def _overall():
            oxy = [c for c in conc.columns if c.endswith(OXY)]
            deoxy = [c for c in conc.columns if c.endswith(DEOXY)]
            if not (oxy and deoxy):
                return
            with_grand = conc.copy()
            with_grand["grand_oxy"] = conc[oxy].mean(axis=1)
            with_grand["grand_deoxy"] = conc[deoxy].mean(axis=1)
            self.viz_conc.plot_raw_overall(
                data=with_grand, output_path=str(out_dir / f"{basename}_raw_overall.pdf"),
                y_limits=None, events=events,
            )
        self._safe("Raw overall plot", _overall)

    def _plot_processed(self, df_final: pd.DataFrame, out_dir: Path, basename: str, events) -> None:
        self._safe("Processed overall plot", lambda: self.viz_conc.plot_processed_overall(
            data=df_final, output_path=str(out_dir / f"{basename}_processed_overall.pdf"),
            y_limits=None, events=events,
        ))
        self._safe("Processed regions plot", lambda: self.viz_conc.plot_processed_regions(
            data=df_final, output_path=str(out_dir / f"{basename}_processed_regions.pdf"), y_limits=None,
        ))
