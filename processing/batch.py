"""Batch driver for FullCapProcessor.

Walks a directory of OxySoft .txt exports, groups files by subject (the
AUT_<NNN> ID convention) and runs each through the pipeline with the chosen
ExclusionCriterion.

Two modes:
    quality-only  SQI/SCI/PSP only. Output: <input_rel>/channel_quality/.
    full          the whole pipeline with the active criterion. Output:
                  <input_rel>/with_<method>_filtering/, or
                  <input_rel>/no_filtering/ when the criterion is "none".

Gate A (the SD-based flag-then-discard chain in process_file.py) needs two
corpus-wide numbers a single file can't provide: dataset_median_cardiac_sd and
channel_median_cardiac_sd. Both optional; without them Gate A stays inert.
The per-channel one is what StatisticsCalculator.collect_quality_summary writes
to its "cardiac_sd_by_channel" sheet (Channel, CardiacSD_Median), so the usual
workflow is a first pass without them, aggregate with statistics.py, then a
second pass using that sheet via --channel-median-cardiac-sd-file.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
_CARDIAC_SD_COLUMNS = ("CardiacSD_Median", "MedianCardiacSD", "Median")


class BatchProcessor:
    def __init__(
        self,
        criterion: Optional[ExclusionCriterion] = None,
        quality_only: bool = False,
        plot_channel_quality: bool = False,
        comparison_criteria: Optional[Sequence[ExclusionCriterion]] = None,
        skip_plots: bool = False,
        hard_sqi_discard: Optional[float] = DEFAULT_HARD_SQI_DISCARD,
        dataset_median_cardiac_sd: Optional[float] = None,
        channel_median_cardiac_sd: Optional[Dict[int, float]] = None,
    ):
        self.criterion = criterion if criterion is not None else make_criterion("sci")
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
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.rglob("*.txt"))
        if not files:
            logger.warning(f"No .txt files found under {input_dir}")
            return {}
        logger.info(f"Found {len(files)} TXT files under {input_dir} (mode: {self._describe_mode()})")

        subjects = self._group_by_subject(files)
        logger.info(f"Grouped into {len(subjects)} subjects")

        results = {
            subject: self._process_subject(subject, subject_files, input_dir, output_dir)
            for subject, subject_files in subjects.items()
        }
        self._save_warnings(output_dir)
        return results

    # ---- internals ------------------------------------------------------

    def _describe_mode(self) -> str:
        if self.quality_only:
            mode = "quality-only"
        elif self.criterion.method == "none":
            mode = "full (no filtering)"
        else:
            mode = f"full ({self.criterion.description})"

        sqi_floor = (
            f"hard SQI floor <= {self.hard_sqi_discard}" if self.hard_sqi_discard is not None
            else "hard SQI floor off"
        )
        gate_a = (
            "Gate A active" if self.dataset_median_cardiac_sd is not None or self.channel_median_cardiac_sd
            else "Gate A inert (no dataset/channel median CardiacSD supplied)"
        )
        return f"{mode}, {sqi_floor}, {gate_a}"

    @staticmethod
    def _group_by_subject(files: List[Path]) -> Dict[str, List[Path]]:
        grouped: Dict[str, List[Path]] = {}
        for f in files:
            grouped.setdefault(BatchProcessor._subject_id(f), []).append(f)
        return grouped

    @staticmethod
    def _subject_id(path: Path) -> str:
        """AUT_<3 digits> from anywhere in the path, else the parent directory name."""
        match = _SUBJECT_RE.search(str(path))
        return match.group(0) if match else path.parent.name

    def _process_subject(self, subject: str, files: List[Path], input_dir: Path, output_dir: Path) -> List[str]:
        logger.info(f"--- Subject {subject}: {len(files)} files ---")
        processor = FullCapProcessor(
            criterion=self.criterion,
            plot_channel_quality=self.plot_channel_quality,
            comparison_criteria=self.comparison_criteria,
            skip_plots=self.skip_plots,
            hard_sqi_discard=self.hard_sqi_discard,
            dataset_median_cardiac_sd=self.dataset_median_cardiac_sd,
            channel_median_cardiac_sd=self.channel_median_cardiac_sd,
        )

        processed = [str(f) for f in files if self._run_one(processor, f, input_dir, output_dir)]

        self.warning_files.extend(processor.warning_files)
        logger.info(f"Subject {subject}: {len(processed)}/{len(files)} succeeded")
        return processed

    def _run_one(self, processor: FullCapProcessor, file_path: Path, input_dir: Path, output_dir: Path) -> bool:
        paths = dict(
            file_path=str(file_path), output_base_dir=str(output_dir), input_base_dir=str(input_dir),
        )
        if self.quality_only:
            return processor.process_file_quality_only(**paths) is not None

        result = processor.process_file(
            **paths, apply_filter=self.criterion.method != "none", plot_raw=True,
        )
        return result is not None

    def _save_warnings(self, output_dir: Path) -> None:
        if not self.warning_files:
            return
        path = output_dir / "processing_warnings.txt"
        path.write_text("\n".join(f"{fp}: {msg}" for fp, msg in self.warning_files) + "\n")
        logger.info(f"Saved {len(self.warning_files)} warnings to {path}")


# ---- Gate A baseline loading ----------------------------------------------

def load_channel_median_cardiac_sd(path: str | Path) -> Dict[int, float]:
    """Per-channel CardiacSD baseline for Gate A, from a .json or .csv file.

    .json  {"1": 0.004, "2": 0.0038, ...} (int keys also fine)
    .csv   a "Channel" column plus "CardiacSD_Median", the shape of
           StatisticsCalculator's cardiac_sd_by_channel sheet
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_cardiac_sd_json(path)
    if suffix == ".csv":
        return _load_cardiac_sd_csv(path)
    raise ValueError(f"{path}: unsupported format {path.suffix!r} -- use .json or .csv")


def _load_cardiac_sd_json(path: Path) -> Dict[int, float]:
    with open(path) as f:
        raw = json.load(f)
    return {int(channel): float(value) for channel, value in raw.items()}


def _load_cardiac_sd_csv(path: Path) -> Dict[int, float]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if columns is None or "Channel" not in columns:
            raise ValueError(f"{path}: expected a 'Channel' column, found {columns}")

        value_col = next((c for c in _CARDIAC_SD_COLUMNS if c in columns), None)
        if value_col is None:
            raise ValueError(f"{path}: expected one of {'/'.join(_CARDIAC_SD_COLUMNS)}, found {columns}")

        medians: Dict[int, float] = {}
        for row in reader:
            try:
                medians[int(row["Channel"])] = float(row[value_col])
            except (TypeError, ValueError):
                continue  # blank or non-numeric rows, e.g. a trailing summary line
    return medians


# ---- standalone CLI -------------------------------------------------------

def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Pipeline configuration flags; shared with cli.main."""
    parser.add_argument(
        "--exclusion-method", choices=METHODS, default="sci",
        help=(
            "Channel-exclusion criterion: 'sqi' (Sappia 2020), 'sci' (Pollonini 2014), "
            "'psp' (PHOEBE 2016), 'sci_psp' (NIRSplot combined rule), or 'none' to skip exclusion"
        ),
    )
    parser.add_argument("--sqi-threshold", type=float, default=DEFAULT_SQI_THRESHOLD,
                        help="SQI floor (1-5 scale; 2.5 loose, 3.0 standard, 3.5 strict)")
    parser.add_argument("--sci-threshold", type=float, default=DEFAULT_SCI_THRESHOLD,
                        help="SCI floor for standalone 'sci' method (Pollonini 2014 uses 0.75)")
    parser.add_argument("--psp-threshold", type=float, default=DEFAULT_PSP_THRESHOLD,
                        help="PSP floor for 'psp' and 'sci_psp' methods (PHOEBE uses 0.1)")
    parser.add_argument("--combined-sci-threshold", type=float, default=COMBINED_SCI_THRESHOLD,
                        help="SCI floor used by 'sci_psp' combined rule (NIRSplot uses 0.5)")
    parser.add_argument("--quality-only", action="store_true",
                        help="Compute only channel-quality metrics; skip TDDR / SCR / bandpass / baseline")
    parser.add_argument(
        "--plot-channels", action="store_true",
        help=(
            "Write a per-channel quality PDF alongside each report: a summary heatmap of all "
            "four methods' decisions plus one page per channel with pass/fail badges"
        ),
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help=(
            "Skip the overview PDFs (raw_OD, raw_all_channels, raw_regions, raw_overall, "
            "processed_regions). Independent of --plot-channels."
        ),
    )
    parser.add_argument(
        "--hard-sqi-discard", type=float, default=DEFAULT_HARD_SQI_DISCARD,
        help=(
            "Tiebreaker for a channel Gate A flagged (weak cardiac-band amplitude) that also "
            "survived the within-file dead checks: SQI at or below this value discards it, "
            "otherwise it is kept. Has no effect on unflagged channels."
        ),
    )
    parser.add_argument(
        "--disable-hard-sqi", action="store_true",
        help="Turn off the hard SQI tiebreaker regardless of --hard-sqi-discard.",
    )
    parser.add_argument(
        "--dataset-median-cardiac-sd", type=float, default=None,
        help=(
            "Corpus-wide median CardiacSD, one of Gate A's two reference numbers. Without this "
            "and --channel-median-cardiac-sd-file, Gate A never flags anything."
        ),
    )
    parser.add_argument(
        "--channel-median-cardiac-sd-file", type=Path, default=None,
        help=(
            "A .json or .csv file with each channel's own median CardiacSD across the dataset. "
            "A .csv with 'Channel' and 'CardiacSD_Median' columns (the cardiac_sd_by_channel "
            "sheet) works as-is."
        ),
    )


def criterion_from_args(args: argparse.Namespace) -> ExclusionCriterion:
    return make_criterion(
        args.exclusion_method,
        sqi_threshold=args.sqi_threshold,
        sci_threshold=args.sci_threshold,
        psp_threshold=args.psp_threshold,
        combined_sci_threshold=args.combined_sci_threshold,
    )


def comparison_criteria_from_args(args: argparse.Namespace) -> List[ExclusionCriterion]:
    """The four standard criteria at the user's thresholds, independent of --exclusion-method.

    Feeds the per-channel plot and the cross-method comparison sheets.
    """
    return [
        make_criterion(
            method,
            sqi_threshold=args.sqi_threshold,
            sci_threshold=args.sci_threshold,
            psp_threshold=args.psp_threshold,
            combined_sci_threshold=args.combined_sci_threshold,
        )
        for method in COMPARISON_METHODS
    ]


def hard_sqi_discard_from_args(args: argparse.Namespace) -> Optional[float]:
    return None if args.disable_hard_sqi else args.hard_sqi_discard


def channel_median_cardiac_sd_from_args(args: argparse.Namespace) -> Optional[Dict[int, float]]:
    if args.channel_median_cardiac_sd_file is None:
        return None
    return load_channel_median_cardiac_sd(args.channel_median_cardiac_sd_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process fNIRS OD TXT files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Root directory containing .txt files")
    parser.add_argument("output_dir", type=Path, help="Directory for processed output")
    add_pipeline_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    batch = BatchProcessor(
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
    total = sum(len(files) for files in results.values())
    print(f"Done. Processed {total} files across {len(results)} subjects.")


if __name__ == "__main__":
    main()
