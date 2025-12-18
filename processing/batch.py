# batch.py
import os
import logging
from typing import List, Dict, Tuple, Optional

from processing.process_file import FullCapProcessor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch-level manager for processing all full-cap fNIRS OD-exported .txt files.
    Uses a two-pass strategy:
       1) First-pass: estimate y-limits from concentration values
       2) Second-pass: process files with consistent y-limits
    """

    def __init__(self, fs: float = 50.0, sci_threshold: float = 0.6):
        self.fs = fs
        self.sci_threshold = sci_threshold
        self.warning_files = []
        logger.info(f"Initialized BatchProcessor (fs={fs}, SCI threshold={sci_threshold})")

    # -------------------------------------------------------------------------
    #     PUBLIC ENTRY POINT
    # -------------------------------------------------------------------------
    def process_batch(self, input_base_dir: str, output_base_dir: str) -> Dict[str, List[str]]:
        """Process all .txt files under input_base_dir."""
        os.makedirs(output_base_dir, exist_ok=True)

        txt_files = self._find_input_files(input_base_dir)
        if not txt_files:
            logger.warning(f"No .txt files found in {input_base_dir}")
            return {}

        subject_files = self._organize_files_by_subject(txt_files)

        processed_files = self._process_subject_files(
            subject_files, input_base_dir, output_base_dir
        )

        self._save_warnings(output_base_dir)
        return processed_files

    # -------------------------------------------------------------------------
    #     PRIVATE HELPERS
    # -------------------------------------------------------------------------
    def _find_input_files(self, root: str) -> List[str]:
        """Return sorted list of all .txt files."""
        txt_files = []
        for r, _, files in os.walk(root):
            for f in files:
                if f.endswith(".txt"):
                    txt_files.append(os.path.join(r, f))
        txt_files.sort()
        logger.info(f"Found {len(txt_files)} TXT files.")
        return txt_files

    def _organize_files_by_subject(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Group files by subject folder."""
        subjects = {}
        for fp in file_paths:
            subj = self._extract_subject_id(fp)
            subjects.setdefault(subj, []).append(fp)
        return subjects

    @staticmethod
    def _extract_subject_id(path: str) -> str:
        """Extract subject ID from folder name."""
        parts = path.split(os.sep)
        for p in parts:
            if "OHSU_Turn" in p or "sub-" in p:
                return p
        return "Unknown"

    # -------------------------------------------------------------------------
    #     SUBJECT-LEVEL PROCESSING
    # -------------------------------------------------------------------------
    def _process_subject_files(
        self,
        subject_to_files: Dict[str, List[str]],
        input_base_dir: str,
        output_base_dir: str
    ) -> Dict[str, List[str]]:

        processed = {}

        for subject, files in subject_to_files.items():
            logger.info(f"\n===============================")
            logger.info(f"Processing subject: {subject}")
            logger.info(f"===============================")

            # ---------- FIRST PASS: extract concentration y-limits ----------
            y_limits = self._calculate_subject_y_limits(files)
            logger.info(f"Subject {subject}: y-limits = {y_limits}")

            subject_processed = []
            processor = FullCapProcessor(fs=self.fs, sci_threshold=self.sci_threshold)

            # ---------- SECOND PASS ----------
            for fp in files:
                out = processor.process_file(
                    file_path=fp,
                    output_base_dir=output_base_dir,
                    input_base_dir=input_base_dir,
                    y_limits=y_limits
                )
                if out is not None:
                    subject_processed.append(fp)

                self.warning_files.extend(processor.warning_files)

            processed[subject] = subject_processed
            logger.info(f"Subject {subject}: processed {len(subject_processed)} / {len(files)} files.")

        return processed

    # -------------------------------------------------------------------------
    #     FIRST-PASS Y-LIMIT ESTIMATION
    # -------------------------------------------------------------------------
    def _calculate_subject_y_limits(self, file_paths: List[str]) -> Optional[Tuple[float, float]]:
        """
        Run a *minimal* first-pass OD→concentration conversion on each file
        to determine consistent y-axis limits.
        """
        from read.loaders import read_txt_file
        from preprocessing.od_to_concentration import convert_od_to_concentration

        conc_ranges = []

        for fp in file_paths:
            try:
                raw = read_txt_file(fp)
                df = raw["data"]
                channel_map = raw["channel_map"]

                conc = convert_od_to_concentration(df, channel_map)

                # Identify concentration columns
                cols = [c for c in conc.columns if "_Hb" in c or "HbO" in c or "HbR" in c]
                if not cols:
                    continue

                vmax = conc[cols].abs().max().max()
                conc_ranges.append(vmax)

            except Exception as e:
                self.warning_files.append((fp, f"Y-limit error: {str(e)}"))
                logger.warning(f"Failed y-limit first-pass on {fp}: {e}")

        if not conc_ranges:
            return None

        max_val = max(conc_ranges) * 1.2
        return (-max_val, max_val)

    # -------------------------------------------------------------------------
    #     SAVE WARNING MESSAGES
    # -------------------------------------------------------------------------
    def _save_warnings(self, outdir: str) -> None:
        if not self.warning_files:
            return
        warn_path = os.path.join(outdir, "processing_warnings.txt")
        with open(warn_path, "w") as f:
            for fp, msg in self.warning_files:
                f.write(f"{fp}: {msg}\n")
        logger.info(f"Saved {len(self.warning_files)} warnings → {warn_path}")


# -------------------------------------------------------------------------
#     CLI ENTRY POINT
# -------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch process OD fNIRS TXT files")
    parser.add_argument("input_dir", help="Input directory containing .txt files")
    parser.add_argument("output_dir", help="Directory for processed output")
    parser.add_argument("--fs", type=float, default=50.0)
    parser.add_argument("--sci_threshold", type=float, default=0.6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    batch = BatchProcessor(fs=args.fs, sci_threshold=args.sci_threshold)
    results = batch.process_batch(args.input_dir, args.output_dir)

    print(f"\nBatch processing complete. Processed subjects: {len(results)}")


if __name__ == "__main__":
    main()
