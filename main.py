"""CLI for the full-cap fNIRS pipeline.

    python -m cli.main <input_dir> <output_dir> [--exclusion-method sqi|sci|psp|sci_psp|none]
                       [--quality-only] [--hard-sqi-discard 1.0 | --disable-hard-sqi]
                       [--dataset-median-cardiac-sd X --channel-median-cardiac-sd-file F]

Each file is processed under the selected criterion; a cross-method exclusion
comparison always goes to <output_dir>/summary so the choice can be revisited
without reprocessing.

Gate A (see processing.process_file) stays inert unless a median cardiac SD is
supplied. Workflow: run once without it, take summary/cardiac_sd_by_channel.csv from
the stats output, then rerun with --channel-median-cardiac-sd-file pointing at it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from processing.batch import (
    BatchProcessor,
    add_pipeline_args,
    channel_median_cardiac_sd_from_args,
    comparison_criteria_from_args,
    criterion_from_args,
    hard_sqi_discard_from_args,
)
from processing.statistics import StatisticsCalculator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-cap fNIRS processing pipeline (OD -> concentration -> stats).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir", type=Path, help="Directory containing OxySoft OD .txt exports")
    p.add_argument("output_dir", type=Path, help="Directory for processed CSVs, plots, and summaries")
    add_pipeline_args(p)
    p.add_argument("--skip-stats", action="store_true", help="Skip batch-level statistics aggregation")
    return p.parse_args()


def run_processing(args: argparse.Namespace) -> int:
    criterion = criterion_from_args(args)
    hard_sqi = hard_sqi_discard_from_args(args)
    channel_sd = channel_median_cardiac_sd_from_args(args)

    batch = BatchProcessor(
        fs=args.fs,
        criterion=criterion,
        quality_only=args.quality_only,
        plot_channel_quality=args.plot_channels,
        comparison_criteria=comparison_criteria_from_args(args),
        skip_plots=args.skip_plots,
        hard_sqi_discard=hard_sqi,
        dataset_median_cardiac_sd=args.dataset_median_cardiac_sd,
        channel_median_cardiac_sd=channel_sd,
    )
    results = batch.process_batch(args.input_dir, args.output_dir)
    total = sum(len(v) for v in results.values())

    if args.quality_only:
        mode = "quality-only"
    elif criterion.method == "none":
        mode = "full pipeline, no exclusion"
    else:
        mode = f"full pipeline ({criterion.description})"
    mode += f", hard SQI floor <= {hard_sqi}" if hard_sqi is not None else ", hard SQI floor off"
    gate_a = args.dataset_median_cardiac_sd is not None or bool(channel_sd)
    mode += ", Gate A active" if gate_a else ", Gate A inert"

    logger.info(f"Processed {total} files across {len(results)} subjects ({mode})")
    return total


def run_statistics(args: argparse.Namespace) -> None:
    """Also writes cardiac_sd_by_channel.csv, the input for a Gate A rerun."""
    calc = StatisticsCalculator()
    summary_dir = args.output_dir / "summary"

    if not args.quality_only:
        stats = calc.collect_dual_pass_statistics(args.output_dir)
        if not stats.empty:
            calc.write_summary_sheets(stats, summary_dir)

    quality = calc.collect_quality_summary(
        args.output_dir,
        sqi_threshold=args.sqi_threshold,
        sci_threshold=args.sci_threshold,
        psp_threshold=args.psp_threshold,
        combined_sci_threshold=args.combined_sci_threshold,
    )
    if quality:
        calc.write_quality_sheets(quality, summary_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if run_processing(args) == 0:
        logger.warning("No files processed; skipping statistics")
        return
    if not args.skip_stats:
        run_statistics(args)


if __name__ == "__main__":
    main()
