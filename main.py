#!/usr/bin/env python3
"""Minimal CLI entry point for the OD → concentration FullCap fNIRS pipeline"""

import argparse
import logging
from processing.pipeline_manager import PipelineManager


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-cap fNIRS processing pipeline (OD → concentration → preprocessing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing raw OD-based OxySoft .txt files",
    )

    parser.add_argument(
        "output_dir",
        help="Directory where processed outputs (CSV + plots) will be written",
    )

    parser.add_argument(
        "--fs",
        type=float,
        default=50.0,
        help="Sampling frequency (Hz)",
    )

    parser.add_argument(
        "--sci_threshold",
        type=float,
        default=2.0,
        help="Signal Quality Index threshold (valid range 1–5; recommended ≥2.0)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging()

    pipeline = PipelineManager()
    pipeline.run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fs=args.fs,
        sci_threshold=args.sci_threshold,
    )


if __name__ == "__main__":
    main()
