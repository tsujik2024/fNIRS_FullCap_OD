# pipeline_manager.py — FULLY UPDATED FOR OD PIPELINE
import os
import logging
from typing import List, Dict, Optional, Tuple

from processing.process_file import FullCapProcessor
from preprocessing.od_to_concentration import convert_od_to_concentration
from read.loaders import read_txt_file
from processing.statistics import StatisticsCalculator

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    High-level orchestrator for the full-cap OD→concentration pipeline.
    Performs a two-pass workflow:
       1) First pass: compute subject-level y-limits from OD→concentration conversion
       2) Second pass: run full processing pipeline with consistent y-limits
    """

    def __init__(self):
        self.input_base_dir = ""
        self.stats_calc = None
        self.warning_files = []

    # ----------------------------------------------------------------------
    # ENTRY POINT
    # ----------------------------------------------------------------------
    def run_pipeline(
            self,
            input_dir: str,
            output_dir: str,
            fs: float = 50.0,
            sci_threshold: float = 0.6
    ) -> None:

        self.input_base_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)

        self.stats_calc = StatisticsCalculator(input_base_dir=input_dir)

        # 1) Find input files
        txt_files = self._discover_txt_files(input_dir)
        subject_map = self._group_by_subject(txt_files)

        total_processed = 0

        for subject, files in subject_map.items():

            logger.info(f"\n===============================")
            logger.info(f"Subject: {subject}")
            logger.info(f"===============================")

            subject_outdir = os.path.join(output_dir, subject)
            os.makedirs(subject_outdir, exist_ok=True)

            # ----- FIRST PASS: compute y-limits for the subject -----
            y_limits = self._estimate_subject_y_limits(files, fs)
            logger.info(f"Subject {subject} y-limits: {y_limits}")

            # Create a processor for this subject
            processor = FullCapProcessor(fs=fs, sci_threshold=sci_threshold)

            # ----- SECOND PASS: DUAL PROCESSING (WITH and WITHOUT SQI) -----
            for fp in files:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing: {os.path.basename(fp)}")
                logger.info(f"{'=' * 60}")

                # Pass 1: WITH SQI filtering
                logger.info("\n>>> PASS 1: WITH SQI FILTERING")
                result_with_sqi = processor.process_file(
                    file_path=fp,
                    output_base_dir=subject_outdir,
                    input_base_dir=input_dir,
                    y_limits=y_limits,
                    apply_sqi_filter=True  # NEW PARAMETER
                )

                if result_with_sqi is not None:
                    total_processed += 1
                    logger.info(f"  ✔ Pass 1 (WITH SQI) Success")
                else:
                    logger.warning(f"  ✖ Pass 1 (WITH SQI) FAILED")

                # Pass 2: WITHOUT SQI filtering
                logger.info("\n>>> PASS 2: WITHOUT SQI FILTERING")
                result_no_sqi = processor.process_file(
                    file_path=fp,
                    output_base_dir=subject_outdir,
                    input_base_dir=input_dir,
                    y_limits=y_limits,
                    apply_sqi_filter=False  # NEW PARAMETER
                )

                if result_no_sqi is not None:
                    total_processed += 1
                    logger.info(f"  ✔ Pass 2 (NO SQI) Success")
                else:
                    logger.warning(f"  ✖ Pass 2 (NO SQI) FAILED")

                # accumulate warnings
                self.warning_files.extend(processor.warning_files)

                logger.info(f"\n{'=' * 60}")
                logger.info(f"Completed both passes for: {os.path.basename(fp)}")
                logger.info(f"{'=' * 60}")

        # ----- PROCESSING COMPLETE -----
        if total_processed > 0:
            logger.info(f"\nSuccessfully processed {total_processed} files.")
            logger.info("Now generating summary statistics...")
            self._collect_statistics_from_output_dir(output_dir)
        else:
            logger.warning("No files were successfully processed!")

        self._save_warnings(output_dir)

    # ----------------------------------------------------------------------
    # FIRST PASS: Y-LIMIT ESTIMATION
    # ----------------------------------------------------------------------
    def _estimate_subject_y_limits(self, file_paths: List[str], fs: float) -> Optional[Tuple[float, float]]:
        """
        Minimal conversion from OD→concentration for each file,
        used ONLY to estimate subject-level concentration amplitude.
        """
        conc_maxes = []

        for fp in file_paths:
            try:
                raw = read_txt_file(fp)
                df = raw["data"]
                channel_map = raw["channel_map"]

                conc_df = convert_od_to_concentration(df, channel_map)

                conc_cols = [c for c in conc_df.columns if "Hb" in c]
                if not conc_cols:
                    continue

                vmax = conc_df[conc_cols].abs().max().max()
                conc_maxes.append(vmax)

            except Exception as e:
                logger.warning(f"Y-limit estimation failed for {fp}: {e}")
                self.warning_files.append((fp, f"Y-limit estimation failed: {e}"))

        if not conc_maxes:
            return None

        max_val = max(conc_maxes) * 1.2
        return (-max_val, max_val)

    # ----------------------------------------------------------------------
    # STATISTICS COLLECTION
    # ----------------------------------------------------------------------
    def _collect_statistics_from_output_dir(self, output_dir: str) -> None:

        try:
            processed_csvs = []

            for root, _, files in os.walk(output_dir):
                for f in files:
                    if f.endswith("_processed.csv") and "bad_sci" not in f.lower():
                        processed_csvs.append(os.path.join(root, f))

            if not processed_csvs:
                logger.warning("No processed CSV files found for statistics.")
                return

            logger.info(f"Found {len(processed_csvs)} processed CSV files.")

            stats_df = self.stats_calc.collect_statistics_from_processed_files(processed_csvs)

            if stats_df is not None and not stats_df.empty:
                logger.info(f"Generated statistics for {len(stats_df)} processed entries")
                self.stats_calc.create_summary_sheets(stats_df, output_dir)
            else:
                logger.warning("Statistics calculator returned no data.")

        except Exception as e:
            logger.error(f"Statistics collection failed: {e}", exc_info=True)
            self.warning_files.append(("Statistics Collection", str(e)))

    # ----------------------------------------------------------------------
    # FILE DISCOVERY / GROUPING
    # ----------------------------------------------------------------------
    def _discover_txt_files(self, root_dir: str) -> List[str]:
        txt_files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".txt"):
                    txt_files.append(os.path.join(root, f))
        logger.info(f"Found {len(txt_files)} TXT files.")
        return sorted(txt_files)

    def _group_by_subject(self, paths: List[str]) -> Dict[str, List[str]]:
        groups = {}
        for p in paths:
            subject = os.path.basename(os.path.dirname(p))
            groups.setdefault(subject, []).append(p)
        logger.info(f"Grouped into {len(groups)} subjects.")
        return groups

    # ----------------------------------------------------------------------
    # WARNINGS
    # ----------------------------------------------------------------------
    def _save_warnings(self, output_dir: str) -> None:
        if not self.warning_files:
            return

        warn_path = os.path.join(output_dir, "warnings.txt")
        with open(warn_path, "w") as f:
            for fp, msg in self.warning_files:
                f.write(f"{fp}: {msg}\n")

        logger.info(f"Saved {len(self.warning_files)} warnings → {warn_path}")
