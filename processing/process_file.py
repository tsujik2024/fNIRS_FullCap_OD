"""fNIRS full-cap processing pipeline.

Order: quality metrics -> TDDR -> OD-to-concentration -> SCR -> bandpass -> baseline.

Quality metrics run on the pre-TDDR signal.

Channel exclusion is a chain.
A channel only enters the SD-based steps if it clears SCI >= 0.75 outright;
otherwise it is left to the active criterion alone. 
    1. flag if cardiac-band amplitude is weak vs. the dataset or its own history
    2/3. discard a flagged channel that is dead within this file
         (whole recording, or >= half of its 5 s windows)
    4. discard a flagged channel that survived 2/3 if SQI <= 1
The active criterion runs on every channel regardless.

Entry points:
    process_file               full pipeline; the criterion drops channels
                               unless apply_filter=False.
    process_file_quality_only  quality metrics only.

Per-file output:
    *_processed.csv        long-channel concentration series, three views per
                           region (bilateral / left / right). No whole-cap or
                           whole-hemisphere aggregate (it would mix PFC and motor).
    *_quality_report.csv   per-channel SQI/SCI/PSP and the exclusion decision.
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

# Every export in this study is 50 Hz. It's a constant rather than a flag
# because each frequency-based step (TDDR, bandpass, the SQI and cardiac-band
# filters, window sizing) derives from it, and a wrong rate would quietly point
# all of them at the wrong part of the spectrum.
FS_HZ: float = 50.0

# Order is a pole count (2-4 typical), not a tap count. The earlier FIR version
# couldn't form a clean edge at 0.01 Hz with these sample counts.
_BANDPASS_BAND = (0.01, 0.1)
_BUTTERWORTH_ORDER = 4

_DEFAULT_BASELINE_S = 20.0
_BASELINE_START_SAMPLE = 4
_DROP_INITIAL_S = 1.0

_METRIC_COLS: Tuple[str, ...] = ("SQI", "SCI", "PSP", "CardiacSD_WL1", "CardiacSD_WL2", "CardiacSD")

# Gate 0: a channel enters steps 1-4 only if SCI clears this.
_SCI_MIN: float = 0.75

# Step 1 flags a channel when CardiacSD falls below either fraction of a
# corpus-wide median (the dataset's, or that channel's own). A single file can't
# produce those medians, so they're passed in; without them nothing is flagged.
_FLAG_DATASET_RATIO: float = 0.05
_FLAG_CHANNEL_RATIO: float = 0.20

# Steps 2/3 only see flagged channels, which is why the ratio is far stricter
# than a standalone flatline test would use (0.01% vs. ~0.1% of the median).
_DEAD_RATIO: float = 1e-4
_DEAD_WINDOW_S: float = 5.0
_DEAD_WINDOW_FRACTION: float = 0.5

_HARD_SQI_DISCARD: Optional[float] = 1.0
# Public aliases, imported by batch.py and main.py so defaults live in one place.
DEFAULT_HARD_SQI_DISCARD: Optional[float] = _HARD_SQI_DISCARD

# Order drives the badge/column layout on the summary heatmap.
_COMPARISON_METHODS: Tuple[str, ...] = ("sqi", "sci", "psp", "sci_psp")
COMPARISON_METHODS: Tuple[str, ...] = _COMPARISON_METHODS


def _to_oxy_deoxy(df: pd.DataFrame) -> pd.DataFrame:
    """'CH{n} HbO' -> 'CH{n}_oxy', 'CH{n} HHb' -> 'CH{n}_deoxy'."""
    return df.rename(columns=lambda c: c.replace(" HbO", OXY).replace(" HHb", DEOXY))


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and bool(np.isfinite(value))


def _windowed_liveliness(oxy: np.ndarray, deoxy: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Per-window max(std_oxy, std_deoxy) over non-overlapping windows.

    A short trailing remainder is dropped, not padded, so a partial last
    window can't read as artificially dead.
    """
    win = max(1, int(window_s * fs))
    return np.array([
        max(np.std(oxy[i * win:(i + 1) * win]), np.std(deoxy[i * win:(i + 1) * win]))
        for i in range(len(oxy) // win)
    ])


class FullCapProcessor:
    def __init__(
        self,
        criterion: Optional[ExclusionCriterion] = None,
        plot_channel_quality: bool = False,
        comparison_criteria: Optional[Sequence[ExclusionCriterion]] = None,
        skip_plots: bool = False,
        hard_sqi_discard: Optional[float] = _HARD_SQI_DISCARD,
        dataset_median_cardiac_sd: Optional[float] = None,
        channel_median_cardiac_sd: Optional[Dict[int, float]] = None,
    ):
        self.fs = FS_HZ
        self.criterion = criterion or make_criterion("sci")
        self.plot_channel_quality = plot_channel_quality
        self.comparison_criteria: List[ExclusionCriterion] = (
            list(comparison_criteria) if comparison_criteria is not None
            else [make_criterion(m) for m in _COMPARISON_METHODS]
        )
        self.skip_plots = skip_plots
        self.hard_sqi_discard = hard_sqi_discard
        self.dataset_median_cardiac_sd = dataset_median_cardiac_sd
        self.channel_median_cardiac_sd: Dict[int, float] = channel_median_cardiac_sd or {}
        self.viz_od = FNIRSVisualizer(fs=self.fs, data_type="od")
        self.viz_conc = FNIRSVisualizer(fs=self.fs, data_type="concentration")
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

        With apply_filter=True, channels that fail the criterion or the gates
        behind it are dropped before SCR/bandpass/baseline and output goes to
        with_<method>_filtering/. Otherwise nothing is dropped and output goes
        to no_filtering/.
        """
        try:
            fp = Path(file_path)
            out_dir = self._make_output_dir(output_base_dir, input_base_dir, fp, apply_filter)
            basename = f"{fp.stem}_{'filtered' if apply_filter else 'unfiltered'}"

            loaded = read_txt_file(fp)
            df_od, events, metadata, channel_map = (
                loaded["data"], loaded["events"], loaded["metadata"], loaded["channel_map"]
            )
            self._check_sample_rate(metadata, fp)
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
        """Pre-TDDR quality metrics only; no motion correction, SCR, bandpass or baseline."""
        try:
            fp = Path(file_path)
            out_dir = self._make_quality_only_output_dir(output_base_dir, input_base_dir, fp)
            basename = fp.stem

            loaded = read_txt_file(fp)
            df_od, events, metadata, channel_map = (
                loaded["data"], loaded["events"], loaded["metadata"], loaded["channel_map"]
            )
            self._check_sample_rate(metadata, fp)
            df_od, events = self._drop_initial_seconds(df_od, events)

            conc = self._pre_tddr_concentration(df_od, channel_map, metadata, events)
            if conc is None:
                logger.error("No wavelength columns in input; aborting quality assessment")
                return None

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
        conc_pre = self._pre_tddr_concentration(df_od, channel_map, metadata, events)
        if conc_pre is None:
            logger.error("No wavelength columns in input; aborting pipeline")
            return None, None
        quality = self._compute_quality_metrics(df_od, conc_pre, channel_map)

        od_only = self._apply_tddr(df_od).filter(regex="WL")
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
        corrected = tddr(df_od.filter(regex="WL|Sample"), sample_rate=self.fs)
        for col in df_od.columns:
            if col not in corrected.columns:
                corrected[col] = df_od[col]
        return corrected

    # ---- quality metrics --------------------------------------------------

    def _compute_quality_metrics(self, df_od: pd.DataFrame, df_conc: pd.DataFrame, channel_map: Dict) -> Dict:
        """Per-channel metrics, then the gate chain and the active criterion.

        See the module docstring for the chain. The criterion runs on every
        channel independently of it.
        """
        metrics, ch_cols, ch_windows = self._measure_channels(df_od, df_conc, channel_map)
        dead_threshold = self._dead_threshold(metrics)

        bad_cols: set = set()
        bad_chs: set = set()

        for m in metrics:
            failed_measurement = m.get("status") == "fail"
            m.update(self._run_gate_chain(m, ch_windows.get(m["channel"]), dead_threshold))

            # Kept apart from the chain flags so the report shows which gate fired.
            m["CriterionExcluded"] = bool(self.criterion.excludes(m))

            chain_fail = m["GateAFlagged"] and (m["Flatlined"] or m["PartialDropout"] or m["HardSQIDiscard"])
            failed = failed_measurement or chain_fail or m["CriterionExcluded"]
            m["status"] = "fail" if failed else "pass"

            if failed:
                bad_chs.add(m["channel"])
                bad_cols.update(ch_cols.get(m["channel"], ()))

        self._log_quality_summary(metrics)

        return {
            "metrics": metrics,
            "excluded_columns": sorted(bad_cols),
            "excluded_channels": sorted(bad_chs),
            "num_passed": sum(1 for m in metrics if m["status"] == "pass"),
            "num_failed": sum(1 for m in metrics if m["status"] == "fail"),
        }

    def _measure_channels(
        self, df_od: pd.DataFrame, df_conc: pd.DataFrame, channel_map: Dict,
    ) -> Tuple[List[Dict], Dict[int, Tuple[str, str]], Dict[int, np.ndarray]]:
        metrics: List[Dict] = []
        ch_cols: Dict[int, Tuple[str, str]] = {}
        ch_windows: Dict[int, np.ndarray] = {}

        for ch_idx, info in channel_map.items():
            columns = self._channel_columns(ch_idx, info, df_od, df_conc)
            if columns is None:
                continue
            od1_col, od2_col, oxy_col, deoxy_col = columns
            ch_cols[ch_idx] = (oxy_col, deoxy_col)

            od1, od2 = df_od[od1_col].values, df_od[od2_col].values
            oxy, deoxy = df_conc[oxy_col].values, df_conc[deoxy_col].values

            if not all(np.all(np.isfinite(sig)) for sig in (od1, od2, oxy, deoxy)):
                metrics.append(self._nan_metric_row(ch_idx, status="fail"))
                continue

            try:
                sqi_val = float(SQI(od1, od2, oxy, deoxy, Fs=self.fs))
            except Exception as exc:
                logger.warning(f"SQI failed on channel {ch_idx}: {exc}")
                metrics.append(self._nan_metric_row(ch_idx, status="fail"))
                continue

            sd_wl1, sd_wl2 = cardiac_band_sd(od1, od2, fs=self.fs)
            both_nan = np.isnan(sd_wl1) and np.isnan(sd_wl2)
            metrics.append({
                "channel": ch_idx,
                "SQI": sqi_val,
                "SCI": scalp_coupling_index(od1, od2, fs=self.fs),
                "PSP": peak_spectral_power(od1, od2, fs=self.fs),
                "CardiacSD_WL1": sd_wl1,
                "CardiacSD_WL2": sd_wl2,
                "CardiacSD": float("nan") if both_nan else float(np.nanmax([sd_wl1, sd_wl2])),
                "SD_oxy": float(np.std(oxy)),
                "SD_deoxy": float(np.std(deoxy)),
            })
            ch_windows[ch_idx] = _windowed_liveliness(oxy, deoxy, self.fs, _DEAD_WINDOW_S)

        return metrics, ch_cols, ch_windows

    @staticmethod
    def _channel_columns(
        ch_idx: int, info: Dict, df_od: pd.DataFrame, df_conc: pd.DataFrame,
    ) -> Optional[Tuple[str, str, str, str]]:
        """OD and concentration column names for a two-wavelength channel, or None if incomplete."""
        wavelengths = sorted(info["columns"])
        if len(wavelengths) != 2:
            return None

        od_candidates = [info["columns"].get(wl) or [] for wl in wavelengths]
        if not all(od_candidates):
            return None
        od1_col, od2_col = od_candidates[0][0], od_candidates[1][0]
        oxy_col, deoxy_col = f"CH{ch_idx}{OXY}", f"CH{ch_idx}{DEOXY}"

        present = (
            od1_col in df_od.columns and od2_col in df_od.columns
            and oxy_col in df_conc.columns and deoxy_col in df_conc.columns
        )
        return (od1_col, od2_col, oxy_col, deoxy_col) if present else None

    def _run_gate_chain(self, m: Dict, windows: Optional[np.ndarray], dead_threshold: Optional[float]) -> Dict:
        """Gate 0 and steps 1-4 for one channel; excludes nothing by itself."""
        result = {
            "SciPass": False,
            "GateAFlagged": False,
            "Flatlined": False,
            "PartialDropout": False,
            "PartialDropoutFraction": float("nan"),
            "HardSQIDiscard": False,
        }

        sci = m.get("SCI")
        if not (_is_finite_number(sci) and sci >= _SCI_MIN):
            return result
        result["SciPass"] = True

        if not self._has_weak_cardiac_signal(m):
            return result
        result["GateAFlagged"] = True

        sd_oxy, sd_deoxy = m.get("SD_oxy"), m.get("SD_deoxy")
        have_sd = _is_finite_number(sd_oxy) and _is_finite_number(sd_deoxy)
        flatlined = (not have_sd) or (dead_threshold is not None and max(sd_oxy, sd_deoxy) < dead_threshold)

        partial_dropout = False
        if windows is not None and windows.size and dead_threshold is not None:
            fraction = float(np.mean(windows < dead_threshold))
            partial_dropout = fraction >= _DEAD_WINDOW_FRACTION
            result["PartialDropoutFraction"] = fraction

        result["Flatlined"] = bool(flatlined)
        result["PartialDropout"] = partial_dropout

        if not (flatlined or partial_dropout):
            sqi = m.get("SQI")
            result["HardSQIDiscard"] = bool(
                self.hard_sqi_discard is not None
                and _is_finite_number(sqi)
                and sqi <= self.hard_sqi_discard
            )
        return result

    def _has_weak_cardiac_signal(self, m: Dict) -> bool:
        cardiac_sd = m.get("CardiacSD")
        if not _is_finite_number(cardiac_sd):
            return False

        if (self.dataset_median_cardiac_sd is not None
                and cardiac_sd < _FLAG_DATASET_RATIO * self.dataset_median_cardiac_sd):
            return True

        own_median = self.channel_median_cardiac_sd.get(m["channel"])
        return bool(
            own_median is not None and np.isfinite(own_median)
            and cardiac_sd < _FLAG_CHANNEL_RATIO * own_median
        )

    def _log_quality_summary(self, metrics: List[Dict]) -> None:
        if not metrics:
            return

        def count(key: str) -> int:
            return sum(1 for m in metrics if m.get(key))

        n = len(metrics)
        passed = sum(1 for m in metrics if m["status"] == "pass")
        no_baseline = self.dataset_median_cardiac_sd is None and not self.channel_median_cardiac_sd

        bits = [
            f"{passed}/{n} pass under {self.criterion.description}",
            f"sci_pass={count('SciPass')}/{n} (SCI >= {_SCI_MIN} -- gate 0)",
            f"gate_a_flagged={count('GateAFlagged')}" + (" (no dataset baseline supplied)" if no_baseline else ""),
            f"dead_whole_file={count('Flatlined')}",
            f"dead_windowed={count('PartialDropout')}"
            + (f" (>={_DEAD_WINDOW_FRACTION:.0%} of {_DEAD_WINDOW_S:g}s windows)"
               if _DEAD_WINDOW_FRACTION > 0 else " (off)"),
            f"hard_sqi_discard={count('HardSQIDiscard')}"
            + (f" (threshold<={self.hard_sqi_discard})" if self.hard_sqi_discard is not None else " (off)"),
        ]

        for col in _METRIC_COLS:
            vals = [m[col] for m in metrics if not (m[col] is None or (isinstance(m[col], float) and np.isnan(m[col])))]
            if vals:
                bits.append(f"{col}=[{min(vals):.3g}, {float(np.mean(vals)):.3g}, {max(vals):.3g}]")

        logger.info("Quality: " + "; ".join(bits))

    @staticmethod
    def _nan_metric_row(channel: int, status: str) -> Dict:
        return {"channel": channel, **{c: float("nan") for c in _METRIC_COLS}, "status": status}

    @staticmethod
    def _dead_threshold(metrics: List[Dict]) -> Optional[float]:
        """_DEAD_RATIO x this recording's median liveliness, or None if nothing usable.

        Liveliness is max(SD_oxy, SD_deoxy). Only channels with finite, positive
        SDs count, so dead channels can't drag the reference down.
        """
        if _DEAD_RATIO <= 0:
            return None
        live = [
            max(m["SD_oxy"], m["SD_deoxy"]) for m in metrics
            if _is_finite_number(m.get("SD_oxy")) and _is_finite_number(m.get("SD_deoxy"))
        ]
        live = [v for v in live if v > 0]
        return _DEAD_RATIO * float(np.median(live)) if live else None

    def _save_quality_report(self, quality: Optional[Dict], out_dir: Path, basename: str, applied: bool) -> None:
        if not quality or not quality["metrics"]:
            return

        metrics = quality["metrics"]
        excluded = set(quality["excluded_channels"])
        rows = [
            {
                "Channel": m["channel"],
                **{c: m[c] for c in _METRIC_COLS},
                "Status": m["status"],
                "SciPass": bool(m.get("SciPass", False)),
                "GateAFlagged": bool(m.get("GateAFlagged", False)),
                "Flatlined": bool(m.get("Flatlined", False)),
                "PartialDropout": bool(m.get("PartialDropout", False)),
                "PartialDropoutFraction": m.get("PartialDropoutFraction", float("nan")),
                "HardSQIDiscard": bool(m.get("HardSQIDiscard", False)),
                "CriterionExcluded": bool(m.get("CriterionExcluded", False)),
                "SD_oxy": m.get("SD_oxy", float("nan")),
                "SD_deoxy": m.get("SD_deoxy", float("nan")),
                "Excluded": applied and m["channel"] in excluded,
                "Method": self.criterion.name,
                "Criterion": self.criterion.description,
            }
            for m in metrics
        ]
        df = pd.DataFrame(rows).sort_values("Channel")

        def total(key: str) -> int:
            return sum(1 for m in metrics if m.get(key))

        summary = {"Channel": "SUMMARY"}
        summary.update({col: float(np.nanmean([m[col] for m in metrics])) for col in _METRIC_COLS})
        summary.update({
            "Status": f"pass={quality['num_passed']} fail={quality['num_failed']}",
            "SciPass": total("SciPass"),
            "GateAFlagged": total("GateAFlagged"),
            "Flatlined": total("Flatlined"),
            "PartialDropout": total("PartialDropout"),
            "PartialDropoutFraction": float(np.nanmean([m["PartialDropoutFraction"] for m in metrics])),
            "HardSQIDiscard": total("HardSQIDiscard"),
            "CriterionExcluded": total("CriterionExcluded"),
            "SD_oxy": float("nan"),
            "SD_deoxy": float("nan"),
            "Excluded": f"applied ({len(excluded)} excluded)" if applied else "not applied",
            "Method": self.criterion.name,
            "Criterion": self.criterion.description,
        })

        path = out_dir / f"{basename}_quality_report.csv"
        pd.concat([df, pd.DataFrame([summary])], ignore_index=True).to_csv(path, index=False)

    # ---- regression / filtering / baseline --------------------------------

    def _apply_scr(self, conc: pd.DataFrame, excluded_channels: List[int], filter_applied: bool) -> pd.DataFrame:
        """Regress each long channel against its mapped short channel(s).

        PFC longs (SCR_PRIORITY_LONGS) use one short: the ipsilateral OctaMon
        short, or the contralateral one if the ipsilateral failed quality. The
        two are never pooled, since averaging the opposite hemisphere's
        superficial signal into the regressor isn't a meaningful correction.

        Motor longs pool every Brite24 short that passed quality into one
        averaged regressor, so a single bad short doesn't disable SCR for every
        long that borrows from it. V1 longs aren't in the map and pass through
        unchanged; no Brite24 short sits close enough to serve as a reference.

        A long is skipped only when all of its shorts failed.
        """
        result = conc.copy()
        excluded = set(excluded_channels) if filter_applied else set()
        applied = fallback = skipped_excluded = skipped_missing = 0

        for long_ch, short_refs in LONG_TO_SHORT_MAP.items():
            usable = [s for s in short_refs if s not in excluded]
            if long_ch in excluded or not usable:
                skipped_excluded += 1
                continue

            if long_ch in SCR_PRIORITY_LONGS:
                chosen = usable[:1]
                fallback += chosen[0] != short_refs[0]
            else:
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
        """Pick the baseline window and subtract it.

        Priority: S1->W1 (standing pre-walk baseline; S1 starts it, W1 ends it),
        then S1->S2, then S1 + baseline_duration, then a Task*Start /
        Baseline*End marker, then the first baseline_duration seconds.

        Only Task*Start / Baseline*End mark the end of a baseline. Bare S1 marks
        the start, so treating it as an end would reference the data to the
        warm-up period before S1.
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

        window = pd.DataFrame({"Sample number": [start, end], "Event": ["BaselineStart", "BaselineEnd"]})
        try:
            return baseline_subtraction(df, window)
        except Exception as exc:
            logger.warning(f"Baseline correction failed: {exc}")
            return df

    def _add_aggregates(self, conc: pd.DataFrame) -> pd.DataFrame:
        """Regional and per-hemisphere regional means over long channels.

        Short channels are SCR references, not cortical signal, so they're
        dropped first. Every region is averaged from the long-channel series
        directly, never from other averages. Each region gets bilateral, left
        and right views for oxy and deoxy. There is no whole-cap aggregate;
        averaging PFC with motor channels isn't a meaningful quantity.
        """
        short_cols = {f"CH{ch}{sfx}" for ch in SHORT_CHANNEL_LIST for sfx in (OXY, DEOXY)}
        long_only = conc.drop(columns=[c for c in short_cols if c in conc.columns])

        out = pd.DataFrame(index=conc.index)
        for region_map in (CH_REGION_MAP_COMBINED, CH_REGION_MAP):
            for region, channels in region_map.items():
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

    def _check_sample_rate(self, metadata: Dict, file_path: Path) -> None:
        """Warn when a header disagrees with FS_HZ, without adopting its value.

        A disagreeing header is more likely garbled than a genuinely different
        recording rate, and trusting it would feed a wrong band into every
        filter downstream.
        """
        for key in ("sample_rate", "export_rate"):
            val = metadata.get(key)
            if val is not None and float(val) != FS_HZ:
                logger.warning(
                    f"{file_path.name}: header {key}={val} Hz, but this pipeline assumes "
                    f"{FS_HZ} Hz for every file. Processing proceeds at {FS_HZ} Hz regardless; "
                    f"if this file really was recorded at {val} Hz, its results will be wrong."
                )

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
        """Log a failed plot instead of letting it abort the file's remaining plots."""
        try:
            fn()
        except Exception as exc:
            logger.warning(f"{label} failed: {exc}")

    def _plot_raw(self, df_od, channel_map, metadata, out_dir: Path, basename: str, events) -> None:
        self._safe("Raw OD plot", lambda: self.viz_od.plot_raw_od(
            data=df_od, output_path=str(out_dir / f"{basename}_raw_OD.pdf"), y_limits=None,
        ))

        try:
            conc = self._pre_tddr_concentration(df_od, channel_map, metadata, events)
        except Exception as exc:
            logger.warning(f"Raw concentration conversion failed: {exc}")
            return
        if conc is None:
            logger.warning("Raw concentration conversion skipped: no wavelength columns")
            return

        self._safe("Raw all-channels plot", lambda: self.viz_conc.plot_raw_all_channels(
            data=conc, output_path=str(out_dir / f"{basename}_raw_all_channels.pdf"), y_limits=None,
        ))
        self._safe("Raw regions plot", lambda: self.viz_conc.plot_raw_regions(
            regional_data=_to_oxy_deoxy(self.averager.average_regions(conc)),
            output_path=str(out_dir / f"{basename}_raw_regions.pdf"), y_limits=None, channel_map=channel_map,
        ))

        def plot_overall():
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
        self._safe("Raw overall plot", plot_overall)

    def _plot_processed(self, df_final: pd.DataFrame, out_dir: Path, basename: str, events) -> None:
        # There is no processed-overall plot: it only draws when
        # grand_oxy/grand_deoxy exist, which _add_aggregates never produces, so
        # it would write a blank PDF for every file.
        self._safe("Processed regions plot", lambda: self.viz_conc.plot_processed_regions(
            data=df_final, output_path=str(out_dir / f"{basename}_processed_regions.pdf"), y_limits=None,
        ))
