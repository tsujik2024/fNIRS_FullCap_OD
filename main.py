"""Command-line entry point for the full-cap fNIRS pipeline.

Usage::

    python -m cli.main <input_dir> <output_dir>
    python -m cli.main <input_dir> <output_dir> --exclusion-method sci_psp
    python -m cli.main <input_dir> <output_dir> --quality-only
    python -m cli.main <input_dir> <output_dir> --exclusion-method none
    python -m cli.main <input_dir> <output_dir> --hard-sqi-discard 1.0
    python -m cli.main <input_dir> <output_dir> --disable-hard-sqi
    python -m cli.main <input_dir> <output_dir> \\
        --dataset-median-cardiac-sd 0.004023 \\
        --channel-median-cardiac-sd-file summary/cardiac_sd_by_channel.csv

The pipeline runs once per file under the selected criterion. A cross-method
exclusion comparison (SQI / SCI / PSP / SCI+PSP) is always written to the
summary directory regardless of which criterion ran, so the choice can be
revisited without reprocessing.

Gate A (see processing.process_file) is inert unless the last two flags above
are supplied, and the pipeline then behaves as if it didn't exist. The
per-channel file normally comes from a previous run's cardiac_sd_by_channel.csv:
run once without it, aggregate with run_statistics, then rerun with it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from processing.batch import (
    BatchProcessor,
    add_pipeline_args,
    criterion_from_args,
    comparison_criteria_from_args,
    hard_sqi_discard_from_args,
    channel_median_cardiac_sd_from_args,
)
from processing.statistics import StatisticsCalculator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-cap fNIRS processing pipeline (OD -> concentration -> stats).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path,
                        help="Directory containing OxySoft OD .txt exports")
    parser.add_argument("output_dir", type=Path,
                        help="Directory for processed CSVs, plots, and summaries")
    add_pipeline_args(parser)
    parser.add_argument("--skip-stats", action="store_true",
                        help="Skip batch-level statistics aggregation")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _describe_run(args: argparse.Namespace, criterion, hard_sqi_discard, gate_a_active: bool) -> str:
    if args.quality_only:
        mode = "quality-only"
    elif criterion.method == "none":
        mode = "full pipeline, no exclusion"
    else:
        mode = f"full pipeline ({criterion.description})"

    sqi_floor = f"hard SQI floor <= {hard_sqi_discard}" if hard_sqi_discard is not None else "hard SQI floor off"
    gate_a = "Gate A active" if gate_a_active else "Gate A inert"
    return f"{mode}, {sqi_floor}, {gate_a}"


def run_processing(args: argparse.Namespace) -> int:
    criterion = criterion_from_args(args)
    hard_sqi_discard = hard_sqi_discard_from_args(args)
    channel_median_cardiac_sd = channel_median_cardiac_sd_from_args(args)

    batch = BatchProcessor(
        criterion=criterion,
        quality_only=args.quality_only,
        plot_channel_quality=args.plot_channels,
        comparison_criteria=comparison_criteria_from_args(args),
        skip_plots=args.skip_plots,
        hard_sqi_discard=hard_sqi_discard,
        dataset_median_cardiac_sd=args.dataset_median_cardiac_sd,
        channel_median_cardiac_sd=channel_median_cardiac_sd,
    )
    results = batch.process_batch(args.input_dir, args.output_dir)
    total = sum(len(files) for files in results.values())

    gate_a_active = args.dataset_median_cardiac_sd is not None or bool(channel_median_cardiac_sd)
    mode = _describe_run(args, criterion, hard_sqi_discard, gate_a_active)
    logger.info(f"Processed {total} files across {len(results)} subjects ({mode})")
    return total


def run_statistics(args: argparse.Namespace) -> None:
    """Aggregate per-file outputs into batch-level summary sheets.

    The cross-method exclusion comparison is always included, at the configured
    thresholds, so the active criterion can be reviewed without a second
    processing run. This is also where cardiac_sd_by_channel.csv is written;
    pass it back as --channel-median-cardiac-sd-file to activate Gate A.
    """
    calc = StatisticsCalculator()
    summary_dir = args.output_dir / "summary"

    if not args.quality_only:
        stats_df = calc.collect_dual_pass_statistics(args.output_dir)
        if not stats_df.empty:
            calc.write_summary_sheets(stats_df, summary_dir)

    quality_summary = calc.collect_quality_summary(
        args.output_dir,
        sqi_threshold=args.sqi_threshold,
        sci_threshold=args.sci_threshold,
        psp_threshold=args.psp_threshold,
        combined_sci_threshold=args.combined_sci_threshold,
    )
    if quality_summary:
        calc.write_quality_sheets(quality_summary, summary_dir)


def main() -> None:
    args = parse_args()
    configure_logging()

    total = run_processing(args)
    if total == 0:
        logger.warning("No files processed; skipping statistics")
        return

    if not args.skip_stats:
        run_statistics(args)


if __name__ == "__main__":
    main()
