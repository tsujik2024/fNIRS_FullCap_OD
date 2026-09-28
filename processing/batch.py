"""Batch driver for FullCapProcessor.

Walks a directory of OxySoft .txt exports, groups them by subject (AUT_<NNN>)
and runs each through the pipeline.

    --quality-only   metrics/report only  -> <rel>/channel_quality/
    (default)        full pipeline        -> <rel>/with_<method>_filtering/,
                     or <rel>/no_filtering/ when the criterion is "none"

Gate A (the SD chain in process_file.py) needs two corpus-wide numbers a single
file can't give it: dataset_median_cardiac_sd and channel_median_cardiac_sd.
Without them it stays inert. Typical workflow: run once to get quality reports,
aggregate with statistics.py, then feed its cardiac_sd_by_channel.csv back in
via --channel-median-cardiac-sd-file for a second pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from processing.process_file import FullCapProcessor, DEFAULT_HARD_SQI_DISCARD, COMPARISON_METHODS
from channel_quality.exclusion import (
    ExclusionCriterion,
    METHODS,
    DEFAULT_SQI_THRESHOLD,
    DEFAULT_SCI_THRESHOLD,
    DEFAULT_PSP_THRESHOLD,
    COMBINED_SCI_THRESHOLD,
    make_criterion,
)

logger = logging.getLogger(__name__)

_SUBJECT_RE = re.compile(r"AUT_\d{3}")


class BatchProcessor:
    def __init__(
        self,
        fs: float = 50.0,
        criterion: Optional[ExclusionCriterion] = None,
        quality_only: bool = False,
        plot_channel_quality: bool = False,
        comparison_criteria: Optional[Sequence[ExclusionCriterion]] = None,
        skip_plots: bool = False,
        hard_sqi_discard: Optional[float] = DEFAULT_HARD_SQI_DISCARD,
        dataset_median_cardiac_sd: Optional[float] = None,
        channel_median_cardiac_sd: Optional[Dict[int, float]] = None,
    ):
        self.fs = fs
        self.criterion = criterion if criterion is not None else make_criterion("sqi")
        self.quality_only = quality_only
        self.plot_channel_quality = plot_channel_quality
        self.comparison_criteria: List[ExclusionCriterion] = (
            list(comparison_criteria) if comparison_criteria is not None
            else [make_criterion(m) for m in COMPARISON_METHODS]
        )
        self.skip_plots = skip_plots
        self.hard_sqi_discard = hard_sqi_discard
        self.dataset_median_cardiac_sd = dataset_median_cardiac_sd
        self.channel_median_cardiac_sd: Dict[int, float] = channel_median_cardiac_sd or {}
        self.warning_files: List[Tuple[str, str]] = []

    def process_batch(self, input_dir: str | Path, output_dir: str | Path) -> Dict[str, List[str]]:
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.rglob("*.txt"))
        if not files:
            logger.warning(f"No .txt files found under {input_dir}")
            return {}

        gate_a = bool(self.dataset_median_cardiac_sd is not None or self.channel_median_cardiac_sd)
        mode = "quality-only" if self.quality_only else self.criterion.description
        logger.info(f"{len(files)} files under {input_dir} | {mode} | "
                    f"hard SQI {self.hard_sqi_discard} | Gate A {'on' if gate_a else 'inert'}")

        subjects = defaultdict(list)
        for f in files:
            subjects[self._subject_id(f)].append(f)

        results = {name: self._process_subject(name, paths, input_dir, output_dir)
                   for name, paths in subjects.items()}
        self._save_warnings(output_dir)
        return results

    @staticmethod
    def _subject_id(path: Path) -> str:
        """AUT_<3 digits> anywhere in the path, else the parent dir name."""
        m = _SUBJECT_RE.search(str(path))
        return m.group(0) if m else path.parent.name

    def _process_subject(self, subject: str, files: List[Path], input_dir: Path, output_dir: Path) -> List[str]:
        proc = FullCapProcessor(
            fs=self.fs,
            criterion=self.criterion,
            plot_channel_quality=self.plot_channel_quality,
            comparison_criteria=self.comparison_criteria,
            skip_plots=self.skip_plots,
            hard_sqi_discard=self.hard_sqi_discard,
            dataset_median_cardiac_sd=self.dataset_median_cardiac_sd,
            channel_median_cardiac_sd=self.channel_median_cardiac_sd,
        )
        done = [str(f) for f in files if self._run_one(proc, f, input_dir, output_dir)]
        self.warning_files.extend(proc.warning_files)
        logger.info(f"{subject}: {len(done)}/{len(files)} ok")
        return done

    def _run_one(self, proc: FullCapProcessor, path: Path, input_dir: Path, output_dir: Path) -> bool:
        args = dict(file_path=str(path), output_base_dir=str(output_dir), input_base_dir=str(input_dir))
        if self.quality_only:
            return proc.process_file_quality_only(**args) is not None
        return proc.process_file(**args, apply_filter=self.criterion.method != "none", plot_raw=True) is not None

    def _save_warnings(self, output_dir: Path) -> None:
        if self.warning_files:
            path = output_dir / "processing_warnings.txt"
            path.write_text("\n".join(f"{fp}: {msg}" for fp, msg in self.warning_files) + "\n")
            logger.info(f"Saved {len(self.warning_files)} warnings to {path}")


def load_channel_median_cardiac_sd(path: str | Path) -> Dict[int, float]:
    """Per-channel CardiacSD baseline for Gate A, by extension:
      .json  {"1": 0.004, "2": 0.0038, ...}
      .csv   'Channel' + 'CardiacSD_Median' columns, i.e. the cardiac_sd_by_channel
             sheet from statistics.py can be fed straight back in
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        return {int(k): float(v) for k, v in json.loads(p.read_text()).items()}

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = next((c for c in ("CardiacSD_Median", "MedianCardiacSD", "Median") if c in df.columns), None)
        if "Channel" not in df.columns or col is None:
            raise ValueError(f"{p}: need a 'Channel' column and one of "
                             f"CardiacSD_Median/MedianCardiacSD/Median, found {list(df.columns)}")
        df = df[["Channel", col]].apply(pd.to_numeric, errors="coerce").dropna()  # skip junk rows
        return dict(zip(df["Channel"].astype(int), df[col].astype(float)))

    raise ValueError(f"{p}: unsupported format {p.suffix!r} (use .json or .csv)")


# ---- CLI (add_pipeline_args and the *_from_args helpers are shared with cli.main) ----

def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    add("--fs", type=float, default=50.0, help="Sampling frequency (Hz)")
    add("--exclusion-method", choices=METHODS, default="sqi",
        help="Exclusion criterion: sqi (Sappia 2020), sci (Pollonini 2014), psp (PHOEBE 2016), "
             "sci_psp (NIRSplot), or none")
    add("--sqi-threshold", type=float, default=DEFAULT_SQI_THRESHOLD,
        help="SQI floor (1-5 scale; 2.5 loose, 3.0 standard, 3.5 strict)")
    add("--sci-threshold", type=float, default=DEFAULT_SCI_THRESHOLD, help="SCI floor for 'sci'")
    add("--psp-threshold", type=float, default=DEFAULT_PSP_THRESHOLD, help="PSP floor for 'psp' and 'sci_psp'")
    add("--combined-sci-threshold", type=float, default=COMBINED_SCI_THRESHOLD,
        help="SCI floor for the 'sci_psp' rule")
    add("--quality-only", action="store_true", help="Metrics/report only; skip TDDR/SCR/bandpass/baseline")
    add("--plot-channels", action="store_true", help="Write a per-channel quality PDF next to each report")
    add("--skip-plots", action="store_true", help="Skip the overview PDFs (independent of --plot-channels)")
    add("--hard-sqi-discard", type=float, default=DEFAULT_HARD_SQI_DISCARD,
        help="Step 4 of the Gate A chain: a Gate-A-flagged channel that passed the dead checks "
             "is dropped if SQI <= this")
    add("--disable-hard-sqi", action="store_true", help="Turn step 4 off (wins over --hard-sqi-discard)")
    add("--dataset-median-cardiac-sd", type=float, default=None,
        help="Corpus-wide median CardiacSD for Gate A. Without this and "
             "--channel-median-cardiac-sd-file Gate A never flags anything.")
    add("--channel-median-cardiac-sd-file", type=Path, default=None,
        help=".json or .csv of each channel's own median CardiacSD (Channel + CardiacSD_Median columns)")


def criterion_from_args(args: argparse.Namespace) -> ExclusionCriterion:
    return make_criterion(
        args.exclusion_method,
        sqi_threshold=args.sqi_threshold,
        sci_threshold=args.sci_threshold,
        psp_threshold=args.psp_threshold,
        combined_sci_threshold=args.combined_sci_threshold,
    )


def comparison_criteria_from_args(args: argparse.Namespace) -> List[ExclusionCriterion]:
    """The four standard criteria at the user's thresholds, independent of --exclusion-method."""
    return [
        make_criterion(m, sqi_threshold=args.sqi_threshold, sci_threshold=args.sci_threshold,
                       psp_threshold=args.psp_threshold, combined_sci_threshold=args.combined_sci_threshold)
        for m in COMPARISON_METHODS
    ]


def hard_sqi_discard_from_args(args: argparse.Namespace) -> Optional[float]:
    return None if args.disable_hard_sqi else args.hard_sqi_discard


def channel_median_cardiac_sd_from_args(args: argparse.Namespace) -> Optional[Dict[int, float]]:
    if args.channel_median_cardiac_sd_file is None:
        return None
    return load_channel_median_cardiac_sd(args.channel_median_cardiac_sd_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-process fNIRS OD TXT files.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Root directory containing .txt files")
    parser.add_argument("output_dir", type=Path, help="Directory for processed output")
    add_pipeline_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    batch = BatchProcessor(
        fs=args.fs,
        criterion=criterion_from_args(args),
        quality_only=args.quality_only,
        plot_channel_quality=args.plot_channels,
        comparison_criteria=comparison_criteria_from_args(args),
        skip_plots=args.skip_plots,
        hard_sqi_discard=hard_sqi_discard_from_args(args),
        dataset_median_cardiac_sd=args.dataset_median_cardiac_sd,
        channel_median_cardiac_sd=channel_median_cardiac_sd_from_args(args),
    )
    results = batch.process_batch(args.input_dir, args.output_dir)
    print(f"Done. Processed {sum(map(len, results.values()))} files across {len(results)} subjects.")


if __name__ == "__main__":
    main()
