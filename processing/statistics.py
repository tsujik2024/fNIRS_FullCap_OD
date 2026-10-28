from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from read.channel_config import CH_REGION_MAP, CH_REGION_MAP_COMBINED, CHANNEL_TO_REGION
from channel_quality.exclusion import (
    DEFAULT_SQI_THRESHOLD,
    DEFAULT_SCI_THRESHOLD,
    DEFAULT_PSP_THRESHOLD,
    COMBINED_SCI_THRESHOLD,
)

logger = logging.getLogger(__name__)

OXY = "_oxy"
DEOXY = "_deoxy"

_METRIC_COLS = ("SQI", "SCI", "PSP")
# Older reports predate these columns; the cardiac sheets are skipped without them.
_CARDIAC_COLS = ("CardiacSD_WL1", "CardiacSD_WL2", "CardiacSD")
# Gate 0 and the step-1 flag; neither excludes a channel on its own.
_CHAIN_COLS = ("SciPass", "GateAFlagged")
_GATE_COLS = ("Flatlined", "PartialDropout", "HardSQIDiscard", "CriterionExcluded")
_DIAGNOSTIC_COLS = ("PartialDropoutFraction",)

# Diagnostics for the cardiac sheets only; none of these affect exclusion.
_CARDIAC_SCI_THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
_CARDIAC_FLATLINE_RATIO = 0.05  # of the dataset median
_CARDIAC_LOW_SD_RATIO = 0.20    # of the channel's own median

_CH_TO_BASE_REGION = {ch: region.split("_", 1)[0] for ch, region in CHANNEL_TO_REGION.items()}

_QUALITY_SHEET_FILENAMES = {
    "exclusion_comparison": "exclusion_comparison.csv",
    "exclusion_by_condition": "exclusion_by_condition.csv",
    "exclusion_by_region": "exclusion_by_region.csv",
    "exclusion_by_gate": "exclusion_by_gate.csv",
    "gate_a_funnel": "gate_a_funnel.csv",
    "cardiac_sd_by_threshold": "cardiac_sd_by_sci_threshold.csv",
    "cardiac_sd_by_channel": "cardiac_sd_by_channel.csv",
    "cardiac_sd_flagged": "cardiac_sd_flagged_high_sci_low_sd.csv",
    "cardiac_sd_by_retention_rule": "cardiac_sd_by_retention_rule.csv",
}

_FILENAME_SUFFIX_RE = re.compile(r"(?:_(?:filtered|unfiltered))?_(?:processed|quality_report)\.csv$")
_RAW_EXT_RE = re.compile(r"\.(?:txt|csv|oxy4|oxyproj)$", re.I)
_FILTERED_DIR_RE = re.compile(r"^with_.+_filtering$")
_UNFILTERED_DIR = "no_filtering"
_QUALITY_ONLY_DIR = "channel_quality"
_UNKNOWN = "Unknown"


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning(f"Failed to read {path}: {exc}")
        return None


def _is_fail(df: pd.DataFrame) -> pd.Series:
    return df["Status"].astype(str).str.lower().eq("fail")


def _sd_stats(sd: pd.Series, with_p95: bool = True) -> Dict[str, float]:
    keys = ["CardiacSD_Median", "CardiacSD_P05"] + (["CardiacSD_P95"] if with_p95 else []) + ["CardiacSD_Min"]
    if sd.empty:
        return dict.fromkeys(keys, np.nan)
    values = {
        "CardiacSD_Median": sd.median(),
        "CardiacSD_P05": sd.quantile(0.05),
        "CardiacSD_P95": sd.quantile(0.95),
        "CardiacSD_Min": sd.min(),
    }
    return {k: round(float(values[k]), 6) for k in keys}


def _has_cardiac(df: pd.DataFrame) -> bool:
    return not df.empty and "SCI" in df.columns and "CardiacSD" in df.columns


class StatisticsCalculator:
    def __init__(self, input_base_dir: str | Path | None = None):
        self.input_base_dir = Path(input_base_dir) if input_base_dir else None

    # ---- entry points ------------------------------------------------------

    def collect_dual_pass_statistics(self, output_base_dir: str | Path) -> pd.DataFrame:
        """One row per *_processed.csv under output_base_dir.

        "Dual pass" is a leftover name; this collects whichever filtered and
        unfiltered runs exist.
        """
        filtered, unfiltered = self._find_processed_files(Path(output_base_dir))
        logger.info(f"Found {len(filtered)} filtered and {len(unfiltered)} unfiltered processed CSVs")

        rows = [self._summarize_file(p, True) for p in filtered]
        rows += [self._summarize_file(p, False) for p in unfiltered]
        rows = [r for r in rows if r]
        if not rows:
            logger.warning("No statistics generated")
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def collect_quality_summary(
        self,
        output_base_dir: str | Path,
        sqi_threshold: float = DEFAULT_SQI_THRESHOLD,
        sci_threshold: float = DEFAULT_SCI_THRESHOLD,
        psp_threshold: float = DEFAULT_PSP_THRESHOLD,
        combined_sci_threshold: float = COMBINED_SCI_THRESHOLD,
    ) -> Dict[str, pd.DataFrame]:
        """Aggregate per-file quality reports into summary tables.

        Reports are read from with_<method>_filtering/, no_filtering/ or
        channel_quality/; a recording found in several keeps the filtered one.

        The thresholds only drive the cross-method comparison sheets. They do
        not change which channels were excluded at processing time.

        Keys: long, per_channel, per_subject, exclusion_comparison,
        exclusion_by_condition, exclusion_by_region, plus, when the columns
        exist, exclusion_by_gate, gate_a_funnel and the cardiac_sd_* sheets.
        Cardiac amplitude uses the livelier wavelength (max of WL1/WL2), so a
        channel only counts as dead when both are flat. cardiac_sd_by_channel is
        the file to pass as --channel-median-cardiac-sd-file on the next run.
        """
        root = Path(output_base_dir)
        reports = self._find_quality_reports(root)
        if not reports:
            logger.warning(f"No quality reports found under {root}")
            return {}

        long_df = self._build_long_frame(reports)
        if long_df is None:
            return {}

        thresholds = {
            "sqi_threshold": sqi_threshold,
            "sci_threshold": sci_threshold,
            "psp_threshold": psp_threshold,
            "combined_sci_threshold": combined_sci_threshold,
        }
        result = {
            "long": long_df,
            "per_channel": self._per_channel_quality_summary(long_df),
            "per_subject": self._per_subject_quality_summary(long_df),
            "exclusion_comparison": self._exclusion_comparison(long_df, **thresholds),
            "exclusion_by_condition": self._exclusion_by_group(long_df, "Condition", thresholds),
            "exclusion_by_region": self._exclusion_by_group(long_df, "Region", thresholds),
        }
        if any(c in long_df.columns for c in _GATE_COLS):
            result["exclusion_by_gate"] = self._exclusion_by_gate(long_df)
        if any(c in long_df.columns for c in _CHAIN_COLS):
            result["gate_a_funnel"] = self._gate_a_funnel(long_df)
        if "CardiacSD" in long_df.columns:
            cardiac_thresholds = sorted(set(_CARDIAC_SCI_THRESHOLDS) | {sci_threshold})
            result.update({
                "cardiac_sd_by_threshold": self._cardiac_sd_by_sci_threshold(long_df, cardiac_thresholds),
                "cardiac_sd_by_channel": self._cardiac_sd_by_channel(long_df, sci_threshold),
                "cardiac_sd_flagged": self._cardiac_sd_flagged(long_df, sci_threshold),
                "cardiac_sd_by_retention_rule": self._cardiac_sd_by_retention_rule(
                    long_df, sci_threshold, psp_threshold, combined_sci_threshold),
            })
        return result

    def _build_long_frame(self, reports: List[Path]) -> Optional[pd.DataFrame]:
        chunks = []
        for path in reports:
            df = self._read_quality_report(path)
            if df is None:
                continue
            for key, value in self._extract_metadata(path).items():
                df[key] = value
            chunks.append(df)
        if not chunks:
            return None

        long_df = pd.concat(chunks, ignore_index=True)
        if "Channel" in long_df.columns:
            long_df["Region"] = long_df["Channel"].map(_CH_TO_BASE_REGION).fillna(_UNKNOWN)
        if {"CardiacSD_WL1", "CardiacSD_WL2"}.issubset(long_df.columns):
            long_df["CardiacSD"] = long_df[["CardiacSD_WL1", "CardiacSD_WL2"]].max(axis=1)

        column_order = (
            ["Subject", "Timepoint", "Condition", "Region", "SourceFile", "Channel"]
            + list(_METRIC_COLS) + list(_CARDIAC_COLS) + list(_CHAIN_COLS) + list(_GATE_COLS)
            + list(_DIAGNOSTIC_COLS) + ["Status", "Excluded", "Method", "Criterion"]
        )
        return long_df[[c for c in column_order if c in long_df.columns]]

    # ---- writers -----------------------------------------------------------

    def write_summary_sheets(
        self, stats_df: pd.DataFrame, output_folder: str | Path, group_by: str = "Condition",
    ) -> None:
        """group_by: "Condition" (summary_<condition>.csv), "Timepoint", or "both"."""
        if stats_df.empty:
            logger.warning("No statistics to write")
            return

        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)

        meta = [c for c in ("Subject", "Timepoint", "Condition", "Filtered", "SourceFile", "TotalSamples")
                if c in stats_df.columns]
        cols = meta + sorted(c for c in stats_df.columns if "Mean" in c)

        stats_df[cols].to_csv(out / "all_subjects_statistics.csv", index=False)
        if "Filtered" in stats_df.columns:
            for value, name in (("Yes", "filtered_statistics.csv"), ("No", "unfiltered_statistics.csv")):
                subset = stats_df[stats_df["Filtered"] == value]
                if not subset.empty:
                    subset[cols].to_csv(out / name, index=False)

        self._write_grouped_summaries(stats_df, out, cols, group_by)
        logger.info(f"Wrote summary sheets to {out}")

    @staticmethod
    def _write_grouped_summaries(stats_df: pd.DataFrame, out: Path, cols: List[str], group_by: str) -> None:
        """Condition -> summary_<cond>.csv, Timepoint -> summary_<tp>.csv, both -> summary_<tp>_<cond>.csv."""
        mode = (group_by or "Condition").strip().lower()
        if mode not in {"condition", "timepoint", "both"}:
            logger.warning(f"Unknown group_by={group_by!r}; using 'Condition'")
            mode = "condition"

        keys = {"condition": ["Condition"], "timepoint": ["Timepoint"], "both": ["Timepoint", "Condition"]}[mode]
        missing = [k for k in keys if k not in stats_df.columns]
        if missing:
            logger.warning(f"Can't group by {keys}: missing {missing}; skipping per-group sheets")
            return

        for key, subset in stats_df.dropna(subset=keys).groupby(keys, dropna=True):
            label = str(key) if len(keys) == 1 else "_".join(str(v) for v in key)
            subset[cols].to_csv(out / f"summary_{label}.csv", index=False)

    def write_quality_sheets(self, quality_summary: Dict[str, pd.DataFrame], output_folder: str | Path) -> None:
        if not quality_summary:
            return
        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in quality_summary.items():
            if df is None or df.empty:
                continue
            path = out / _QUALITY_SHEET_FILENAMES.get(name, f"quality_{name}.csv")
            df.to_csv(path, index=False)
            logger.info(f"Wrote {name} ({len(df)} rows) to {path}")

    # ---- file discovery ----------------------------------------------------

    @staticmethod
    def _find_processed_files(root: Path) -> Tuple[List[Path], List[Path]]:
        filtered, unfiltered = [], []
        for path in sorted(root.rglob("*_processed.csv")):
            if any(_FILTERED_DIR_RE.match(part) for part in path.parts):
                filtered.append(path)
            elif _UNFILTERED_DIR in path.parts:
                unfiltered.append(path)
            else:
                logger.warning(f"Can't classify {path} (no pipeline subdirectory); skipping")
        return filtered, unfiltered

    @staticmethod
    def _find_quality_reports(root: Path) -> List[Path]:
        """One report per recording, keyed on recording directory + source stem.

        When a recording has several, filtered beats quality-only beats unfiltered.
        """
        chosen: Dict[Tuple[str, str], Tuple[int, Path]] = {}
        for path in sorted(root.rglob("*_quality_report.csv")):
            if any(_FILTERED_DIR_RE.match(part) for part in path.parts):
                priority = 0
            elif _QUALITY_ONLY_DIR in path.parts:
                priority = 1
            elif _UNFILTERED_DIR in path.parts:
                priority = 2
            else:
                continue
            key = (str(path.parent.parent), _FILENAME_SUFFIX_RE.sub("", path.name))
            if key not in chosen or chosen[key][0] > priority:
                chosen[key] = (priority, path)
        return sorted(path for _, path in chosen.values())

    # ---- per-file stats ----------------------------------------------------

    def _summarize_file(self, path: Path, filtered: bool) -> Optional[Dict]:
        df = _read_csv(path)
        if df is None:
            return None
        if df.empty:
            logger.warning(f"Empty CSV: {path}")
            return None

        meta = self._extract_metadata(path)
        meta["Filtered"] = "Yes" if filtered else "No"
        meta["TotalSamples"] = len(df)

        half = len(df) // 2
        return {**meta, **self._grand_stats(df, half), **self._region_stats(df, half)}

    @staticmethod
    def _means(series: pd.Series, half: int, label: str) -> Dict[str, float]:
        return {
            f"{label} Overall Mean": series.mean(),
            f"{label} First Half Mean": series.iloc[:half].mean(),
            f"{label} Second Half Mean": series.iloc[half:].mean(),
        }

    @classmethod
    def _grand_stats(cls, df: pd.DataFrame, half: int) -> Dict[str, float]:
        # process_file.py no longer writes grand_oxy/grand_deoxy, so this is
        # empty for its output; it still covers older processed CSVs.
        stats: Dict[str, float] = {}
        for kind, col in (("Oxy", "grand_oxy"), ("Deoxy", "grand_deoxy")):
            if col in df.columns:
                stats.update(cls._means(df[col], half, f"Grand {kind}"))
        return stats

    @classmethod
    def _region_stats(cls, df: pd.DataFrame, half: int) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for region in list(CH_REGION_MAP) + list(CH_REGION_MAP_COMBINED):
            for kind, suffix in (("Oxy", OXY), ("Deoxy", DEOXY)):
                if f"{region}{suffix}" in df.columns:
                    stats.update(cls._means(df[f"{region}{suffix}"], half, f"{region} {kind}"))
        return stats

    # ---- quality-report aggregation ----------------------------------------

    @staticmethod
    def _read_quality_report(path: Path) -> Optional[pd.DataFrame]:
        df = _read_csv(path)
        if df is None:
            return None
        df["Channel"] = pd.to_numeric(df["Channel"], errors="coerce").astype("Int64")
        for col in (*_METRIC_COLS, *_CARDIAC_COLS, *_DIAGNOSTIC_COLS):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # The SUMMARY row holds integer counts, which makes pandas read the
        # boolean flag columns back as 1/0 rather than True/False.
        for col in (*_CHAIN_COLS, *_GATE_COLS):
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().isin(("true", "1", "1.0"))
        return df.dropna(subset=["Channel"])  # also drops the SUMMARY row

    @staticmethod
    def _per_channel_quality_summary(long_df: pd.DataFrame) -> pd.DataFrame:
        metrics = [c for c in _METRIC_COLS if c in long_df.columns]
        grouped = long_df.groupby("Channel")
        agg = grouped[metrics].agg(["mean", "median", "min", "max"])
        agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
        agg["n_observations"] = grouped.size()
        if "Status" in long_df.columns:
            failed = long_df.assign(_fail=_is_fail(long_df)).groupby("Channel")["_fail"].mean()
            agg["pct_failed_active_criterion"] = failed.mul(100).round(2)
        return agg.reset_index()

    @staticmethod
    def _per_subject_quality_summary(long_df: pd.DataFrame) -> pd.DataFrame:
        metrics = [c for c in _METRIC_COLS if c in long_df.columns]
        grouped = long_df.groupby("Subject")
        agg = grouped[metrics].agg(["mean", "median"])
        agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
        agg["n_files"] = grouped["SourceFile"].nunique() if "SourceFile" in long_df.columns else grouped.size()
        agg["n_channel_observations"] = grouped.size()
        if "Status" in long_df.columns:
            failed = long_df.assign(_fail=_is_fail(long_df)).groupby("Subject")["_fail"].sum()
            agg["n_failed_active_criterion"] = failed.astype(int)
            agg["pct_failed_active_criterion"] = (
                agg["n_failed_active_criterion"] / agg["n_channel_observations"] * 100
            ).round(2)
        return agg.reset_index().sort_values("Subject")

    @staticmethod
    def _exclusion_comparison(
        long_df: pd.DataFrame, sqi_threshold: float, sci_threshold: float,
        psp_threshold: float, combined_sci_threshold: float,
    ) -> pd.DataFrame:
        """Exclusion rate under each literature criterion, recomputed from the raw metrics.

        Independent of what actually ran at processing time (that's
        Status/Excluded; see exclusion_by_gate and gate_a_funnel). NaN counts as
        a fail. Counts are (file, channel) observations.
        """
        if long_df.empty:
            return pd.DataFrame()

        n = len(long_df)
        rows: List[Dict] = []

        def add(method: str, description: str, fail: pd.Series) -> None:
            excluded = int(fail.sum())
            rows.append({
                "Method": method, "Criterion": description, "N_Channels": n,
                "N_Excluded": excluded, "Pct_Excluded": round(100.0 * excluded / n, 2),
            })

        if "SQI" in long_df.columns:
            sqi = long_df["SQI"]
            add("SQI", f"SQI < {sqi_threshold}", (sqi < sqi_threshold) | sqi.isna())
        if "SCI" in long_df.columns:
            sci = long_df["SCI"]
            add("SCI", f"SCI < {sci_threshold}", (sci < sci_threshold) | sci.isna())
        if "PSP" in long_df.columns:
            psp = long_df["PSP"]
            add("PSP", f"PSP < {psp_threshold}", (psp < psp_threshold) | psp.isna())
        if "SCI" in long_df.columns and "PSP" in long_df.columns:
            sci, psp = long_df["SCI"], long_df["PSP"]
            add("SCI+PSP", f"SCI < {combined_sci_threshold} OR PSP < {psp_threshold}",
                (sci < combined_sci_threshold) | (psp < psp_threshold) | sci.isna() | psp.isna())

        return pd.DataFrame(rows)

    @classmethod
    def _exclusion_by_group(cls, long_df: pd.DataFrame, group_col: str, thresholds: Dict[str, float]) -> pd.DataFrame:
        if long_df.empty or group_col not in long_df.columns:
            return pd.DataFrame()

        rows: List[Dict] = []
        for value in sorted(long_df[group_col].dropna().unique()):
            comparison = cls._exclusion_comparison(long_df[long_df[group_col] == value], **thresholds)
            rows += [{group_col: value, **record} for record in comparison.to_dict("records")]
        return pd.DataFrame(rows)

    @staticmethod
    def _exclusion_by_gate(long_df: pd.DataFrame) -> pd.DataFrame:
        """How many observations each exclusion flag caught, and how many only that flag caught.

        The chain flags aren't independent (see the module docstring), and
        N_Caught won't sum to the final fail count because a channel can trip
        several gates.
        """
        n = len(long_df)
        if n == 0:
            return pd.DataFrame()

        labels = {
            "Flatlined": "Flatlined (step 2: whole-file dead check)",
            "PartialDropout": "PartialDropout (step 3: windowed dead check)",
            "HardSQIDiscard": "HardSQIDiscard (step 4: SQI tiebreaker)",
            "CriterionExcluded": "Active criterion (independent of the chain)",
        }
        present = [c for c in _GATE_COLS if c in long_df.columns]
        flags = {c: long_df[c].fillna(False) for c in present}

        rows: List[Dict] = []
        for col in present:
            caught = flags[col]
            others = [flags[other] for other in present if other != col]
            alone = (caught & ~pd.concat(others, axis=1).any(axis=1)) if others else caught
            rows.append({
                "Gate": labels.get(col, col),
                "N_Channels": n,
                "N_Caught": int(caught.sum()),
                "Pct_Caught": round(100.0 * caught.sum() / n, 2),
                "N_Caught_By_This_Gate_Alone": int(alone.sum()),
            })

        if "Status" in long_df.columns:
            failed = int(_is_fail(long_df).sum())
            rows.append({
                "Gate": "Any (final Status == fail)", "N_Channels": n, "N_Caught": failed,
                "Pct_Caught": round(100.0 * failed / n, 2), "N_Caught_By_This_Gate_Alone": np.nan,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _gate_a_funnel(long_df: pd.DataFrame) -> pd.DataFrame:
        """Observations reaching each stage of the chain.

        The place to look when tuning the SCI minimum or the flag ratios;
        exclusion_by_gate can't show channels that never entered the chain.
        """
        n = len(long_df)
        if n == 0:
            return pd.DataFrame()

        def stage(label: str, count: int) -> Dict:
            return {"Stage": label, "N": count, "Pct_Of_Total": round(100.0 * count / n, 2)}

        rows = [{"Stage": "Total channel observations", "N": n, "Pct_Of_Total": 100.0}]
        if "SciPass" not in long_df.columns:
            return pd.DataFrame(rows)
        rows.append(stage("Cleared gate 0 (SCI precondition)", int(long_df["SciPass"].fillna(False).sum())))

        if "GateAFlagged" not in long_df.columns:
            return pd.DataFrame(rows)
        flagged = long_df["GateAFlagged"].fillna(False)
        n_flagged = int(flagged.sum())
        rows.append(stage("Flagged by step 1 (weak vs. dataset/own history)", n_flagged))

        chain = [c for c in ("Flatlined", "PartialDropout", "HardSQIDiscard") if c in long_df.columns]
        if chain and flagged.any():
            via_chain = pd.concat([long_df[c].fillna(False) for c in chain], axis=1).any(axis=1)
            n_excluded = int((flagged & via_chain).sum())
            rows.append(stage("  -> excluded by the chain (steps 2-4)", n_excluded))
            rows.append(stage("  -> kept despite the flag", n_flagged - n_excluded))
        return pd.DataFrame(rows)

    # ---- cardiac-band amplitude (SD) diagnostics ---------------------------

    @staticmethod
    def _cardiac_sd_by_sci_threshold(long_df: pd.DataFrame, thresholds) -> pd.DataFrame:
        """CardiacSD distribution among channels kept at each SCI cutoff.

        Tightening the cutoff drops the dead and flat channels, so the low
        percentiles climb; that's the retention/quality tradeoff.
        """
        if not _has_cardiac(long_df):
            return pd.DataFrame()
        valid = long_df.dropna(subset=["SCI"])
        total = len(valid)

        rows = []
        for threshold in sorted({float(t) for t in thresholds}):
            kept = valid[valid["SCI"] > threshold]
            rows.append({
                "SCI_Threshold": threshold,
                "N_Channels": total,
                "N_Retained": len(kept),
                "Pct_Retained": round(100.0 * len(kept) / total, 2) if total else np.nan,
                "N_Discarded": total - len(kept),
                **_sd_stats(kept["CardiacSD"].dropna()),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _cardiac_sd_by_channel(long_df: pd.DataFrame, sci_threshold: float) -> pd.DataFrame:
        """Per-channel cardiac amplitude among SCI > threshold observations, weakest first.

        Shows channels that couple well enough to pass SCI but are too flat to
        carry signal. Channel + CardiacSD_Median is exactly Gate A's per-channel
        baseline, so this sheet can be fed straight to --channel-median-cardiac-sd-file.
        """
        if not _has_cardiac(long_df):
            return pd.DataFrame()
        coupled = long_df.dropna(subset=["SCI", "CardiacSD"])
        coupled = coupled[coupled["SCI"] > sci_threshold]
        if coupled.empty:
            return pd.DataFrame()

        out = coupled.groupby("Channel")["CardiacSD"].agg(
            N="size", CardiacSD_Median="median",
            CardiacSD_P05=lambda s: s.quantile(0.05), CardiacSD_Min="min",
        ).reset_index()
        dataset_median = float(coupled["CardiacSD"].median())
        out["Pct_Of_Dataset_Median"] = (
            (out["CardiacSD_Median"] / dataset_median * 100).round(1) if dataset_median else np.nan
        )
        return out.sort_values("CardiacSD_Median").reset_index(drop=True)

    @staticmethod
    def _cardiac_sd_flagged(
        long_df: pd.DataFrame, sci_threshold: float,
        flatline_ratio: float = _CARDIAC_FLATLINE_RATIO, low_sd_ratio: float = _CARDIAC_LOW_SD_RATIO,
    ) -> pd.DataFrame:
        """SCI > threshold observations with dead or weak cardiac SD.

        flat_line and weak_for_channel are this sheet's own diagnostics,
        evaluated on every SCI-passing row whether or not the pipeline flagged
        it. The Excluded_By_* columns show whether the real gates fired, so a
        row flagged here that tripped none of them is the borderline case: odd
        on this diagnostic, never caught by what actually removes channels.
        """
        if not _has_cardiac(long_df):
            return pd.DataFrame()
        kept = long_df.dropna(subset=["SCI", "CardiacSD"])
        kept = kept[kept["SCI"] > sci_threshold].copy()
        if kept.empty:
            return pd.DataFrame()

        dataset_median = float(kept["CardiacSD"].median())
        kept["Channel_Median_SD"] = kept.groupby("Channel")["CardiacSD"].transform("median")
        kept["Dataset_Median_SD"] = dataset_median
        kept["flat_line"] = kept["CardiacSD"] < flatline_ratio * dataset_median
        kept["weak_for_channel"] = kept["CardiacSD"] < low_sd_ratio * kept["Channel_Median_SD"]

        flagged = kept[kept["flat_line"] | kept["weak_for_channel"]].rename(columns={
            "GateAFlagged": "Flagged_By_Step1",
            "Flatlined": "Excluded_By_Flatline_Gate",
            "PartialDropout": "Excluded_By_PartialDropout_Gate",
            "HardSQIDiscard": "Excluded_By_HardSQI_Gate",
            "CriterionExcluded": "Excluded_By_Active_Criterion",
        })
        output_cols = [c for c in (
            "Subject", "Timepoint", "Condition", "Region", "SourceFile", "Channel",
            "SCI", "PSP", "SQI", "CardiacSD", "CardiacSD_WL1", "CardiacSD_WL2",
            "Channel_Median_SD", "Dataset_Median_SD", "flat_line", "weak_for_channel",
            "Flagged_By_Step1", "Excluded_By_Flatline_Gate", "Excluded_By_PartialDropout_Gate",
            "Excluded_By_HardSQI_Gate", "Excluded_By_Active_Criterion", "Status", "Excluded",
        ) if c in flagged.columns]
        return flagged[output_cols].sort_values("CardiacSD").reset_index(drop=True)

    @staticmethod
    def _cardiac_sd_by_retention_rule(
        long_df: pd.DataFrame, sci_threshold: float, psp_threshold: float,
        combined_sci_threshold: float, flatline_ratio: float = _CARDIAC_FLATLINE_RATIO,
    ) -> pd.DataFrame:
        """Does adding PSP to an SCI-only cutoff remove the flat channels it leaves in?

        Each row is a candidate retention rule; N_Flatline counts retained
        channels with almost no cardiac pulsation (CardiacSD below
        flatline_ratio x the dataset median). NaN metrics count as not
        retained. The last row is the full pipeline outcome (chain plus active
        criterion), not the criterion alone.
        """
        if not _has_cardiac(long_df):
            return pd.DataFrame()
        valid = long_df.dropna(subset=["SCI", "CardiacSD"])
        if valid.empty:
            return pd.DataFrame()

        total = len(valid)
        flat_cutoff = flatline_ratio * float(valid["CardiacSD"].median())

        def summarize(rule: str, description: str, retained: pd.Series) -> Dict:
            kept = valid[retained]
            sd = kept["CardiacSD"]
            return {
                "Rule": rule, "Criterion": description, "N_Total": total, "N_Retained": len(kept),
                "Pct_Retained": round(100.0 * len(kept) / total, 2),
                "N_Flatline": int((sd < flat_cutoff).sum()),
                **_sd_stats(sd, with_p95=False),
            }

        rows = [summarize("SCI", f"SCI >= {sci_threshold}", valid["SCI"] >= sci_threshold)]
        if "PSP" in valid.columns:
            rows.append(summarize("PSP", f"PSP >= {psp_threshold}", valid["PSP"] >= psp_threshold))
            rows.append(summarize(
                "SCI+PSP", f"SCI >= {combined_sci_threshold} AND PSP >= {psp_threshold}",
                (valid["SCI"] >= combined_sci_threshold) & (valid["PSP"] >= psp_threshold),
            ))
        if "Status" in valid.columns:
            rows.append(summarize(
                "Full pipeline", "Status == pass (as processed)",
                valid["Status"].astype(str).str.lower().eq("pass"),
            ))
        return pd.DataFrame(rows)

    # ---- metadata from paths -----------------------------------------------

    def _extract_metadata(self, path: Path) -> Dict[str, str]:
        return {
            "Subject": self._subject(path),
            "Timepoint": self._visit(path),
            "Condition": self._condition(path.name),
            "SourceFile": path.name,
        }

    @staticmethod
    def _subject(path: Path) -> str:
        match = re.search(r"AUT_\d{3}", str(path))
        return match.group(0) if match else _UNKNOWN

    @staticmethod
    def _visit(path: Path) -> str:
        """V<digits> anywhere in the path, ignoring letter-adjacent hits like WL1 or MOVE."""
        match = re.search(r"(?<![A-Za-z])V(\d+)(?![A-Za-z])", str(path))
        if match:
            return f"V{match.group(1)}"
        logger.warning(f"No timepoint in {path.name!r}; using {_UNKNOWN!r}")
        return _UNKNOWN

    @staticmethod
    def _condition(filename: str) -> str:
        name = _FILENAME_SUFFIX_RE.sub("", filename)
        name = _RAW_EXT_RE.sub("", name)
        name = re.sub(r"^[A-Za-z]+_\d+_", "", name, count=1)    # subject prefix
        name = re.sub(r"^[Vv]\d+_", "", name, count=1)          # visit prefix
        name = re.split(r"_O[DS](?:_|$)", name, maxsplit=1, flags=re.I)[0]  # everything after the optics marker

        previous = None
        while previous != name:
            previous = name
            name = re.sub(r"_(?:correct(?:ed)?|[Vv]\d+)$", "", name, flags=re.I)

        tokens = [t for t in name.split("_") if t]
        if not tokens:
            logger.warning(f"No condition in {filename!r}; using {_UNKNOWN!r}")
            return _UNKNOWN
        return "_".join(StatisticsCalculator._normalize_condition_token(t) for t in tokens)

    @staticmethod
    def _normalize_condition_token(token: str) -> str:
        if re.fullmatch(r"(?:DT|ST)\d*", token, re.I):
            return token.upper()
        return token[:1].upper() + token[1:].lower()
