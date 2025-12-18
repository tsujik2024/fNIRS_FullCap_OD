# process_file.py  (FULLY REWRITTEN FOR OD PIPELINE WITH DUAL-PASS SQI)
# FIXED VERSION - handles HbO/HHb column naming from od_to_concentration
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# Loaders and utilities
from read.loaders import read_txt_file  # returns OD + metadata + channel_map

# Conversion
from preprocessing.od_to_concentration import convert_od_to_concentration

# Preprocessing
from preprocessing.signalqualityindex import SQI
from preprocessing.fir_filter import fir_filter
from preprocessing.short_channel_regression import scr_regression
from preprocessing.tddr import tddr
from preprocessing.baseline_correction import baseline_subtraction

# Region-level tools
from preprocessing.average_channels import FullCapChannelAverager
from viz.visualizer import FNIRSVisualizer

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# MAIN PROCESSOR CLASS
# -------------------------------------------------------------------
class FullCapProcessor:
    """
    Updated full-cap processor for OD → concentration → SCR → FIR → baseline → TDDR pipeline.
    Supports dual-pass processing: with and without SQI filtering.
    """

    def __init__(self, fs: float = 50.0, sci_threshold: float = 0.6):
        self.fs = fs
        self.sci_threshold = sci_threshold
        self.visualizer_od = FNIRSVisualizer(fs=fs, data_type='od')  # For OD plots
        self.visualizer_conc = FNIRSVisualizer(fs=fs, data_type='concentration')  # For concentration plots
        self.channel_averager = FullCapChannelAverager()
        self.warning_files = []

    # -------------------------------------------------------------------
    # HELPER: Standardize column names
    # -------------------------------------------------------------------
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize concentration column names to use _oxy/_deoxy suffix.
        Handles both 'CH1 HbO'/'CH1 HHb' and 'CH1 oxy'/'CH1 deoxy' formats.
        """
        rename_map = {}
        for c in df.columns:
            new_name = c
            if ' HbO' in c:
                new_name = c.replace(' HbO', '_oxy')
            elif ' HHb' in c:
                new_name = c.replace(' HHb', '_deoxy')
            elif c.endswith(' oxy'):
                new_name = c.replace(' oxy', '_oxy')
            elif c.endswith(' deoxy'):
                new_name = c.replace(' deoxy', '_deoxy')

            if new_name != c:
                rename_map[c] = new_name

        if rename_map:
            logger.debug(f"Standardizing {len(rename_map)} column names")

        return df.rename(columns=rename_map)

    def _get_hbo_hhb_colnames(self, df: pd.DataFrame, ch_idx: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the HbO and HHb column names for a channel, handling both naming conventions.
        Returns (hbo_col, hhb_col) or (None, None) if not found.
        """
        # Try HbO/HHb naming first (from od_to_concentration)
        hbo_col = f"CH{ch_idx} HbO"
        hhb_col = f"CH{ch_idx} HHb"

        if hbo_col in df.columns and hhb_col in df.columns:
            return hbo_col, hhb_col

        # Try _oxy/_deoxy naming (standardized)
        hbo_col = f"CH{ch_idx}_oxy"
        hhb_col = f"CH{ch_idx}_deoxy"

        if hbo_col in df.columns and hhb_col in df.columns:
            return hbo_col, hhb_col

        return None, None

    # -------------------------------------------------------------------
    # DUAL-PASS ENTRY POINT
    # -------------------------------------------------------------------
    def process_file_dual_pass(
            self,
            file_path: str,
            output_base_dir: str,
            input_base_dir: str,
            y_limits: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Process a file twice: once with SQI filtering and once without.

        Returns:
            Dictionary with keys 'with_sqi' and 'without_sqi', each containing
            the processed DataFrame (or None if processing failed).
        """
        results = {}

        # Pass 1: With SQI filtering
        logger.info(f"=== DUAL PASS 1/2: Processing WITH SQI filtering ===")
        results['with_sqi'] = self.process_file(
            file_path=file_path,
            output_base_dir=output_base_dir,
            input_base_dir=input_base_dir,
            y_limits=y_limits,
            apply_sqi_filter=True,
            plot_raw=True  # Only plot raw data once
        )

        # Pass 2: Without SQI filtering
        logger.info(f"=== DUAL PASS 2/2: Processing WITHOUT SQI filtering ===")
        results['without_sqi'] = self.process_file(
            file_path=file_path,
            output_base_dir=output_base_dir,
            input_base_dir=input_base_dir,
            y_limits=y_limits,
            apply_sqi_filter=False,
            plot_raw=False  # Don't re-plot raw data
        )

        # Log summary
        logger.info(f"=== DUAL PASS COMPLETE ===")
        logger.info(f"  With SQI: {'Success' if results['with_sqi'] is not None else 'Failed'}")
        logger.info(f"  Without SQI: {'Success' if results['without_sqi'] is not None else 'Failed'}")

        return results

    # -------------------------------------------------------------------
    # PUBLIC ENTRY POINT (SINGLE PASS)
    # -------------------------------------------------------------------
    def process_file(
            self,
            file_path: str,
            output_base_dir: str,
            input_base_dir: str,
            y_limits: Optional[Tuple[float, float]] = None,
            apply_sqi_filter: bool = True,
            plot_raw: bool = True  # New parameter to control raw plotting
    ) -> Optional[pd.DataFrame]:

        try:
            # Prepare output directory (includes subfolder for SQI vs no-SQI)
            out_dir = self._create_output_dir(output_base_dir, input_base_dir, file_path, apply_sqi_filter)
            basename = Path(file_path).stem

            # Add suffix to distinguish SQI vs no-SQI outputs
            if apply_sqi_filter:
                basename_suffix = f"{basename}_SQI"
            else:
                basename_suffix = f"{basename}_NoSQI"

            # ---------------------------------------------
            # Load raw OD + metadata + events + channel_map
            # ---------------------------------------------
            loaded = read_txt_file(file_path)
            df_od = loaded["data"]  # OD values only
            events = loaded["events"]
            metadata = loaded["metadata"]
            channel_map = loaded["channel_map"]

            # Drop 1 second of startup artifact
            df_od, events = self._drop_initial_second(df_od, events)

            # -------------------------------------------------------------
            # RAW PLOTS (OD) - only plot if requested (avoids duplication in dual-pass)
            # -------------------------------------------------------------
            if plot_raw:
                self._plot_raw_od(df_od, out_dir, basename_suffix, events)

            # -------------------------------------------------------------
            # CONVERT TO CONCENTRATION FOR RAW CONCENTRATION PLOTS
            # -------------------------------------------------------------
            # We need raw concentration for raw plots
            od_data_for_conversion = df_od.filter(regex="WL")
            conc_raw = convert_od_to_concentration(
                od_data_for_conversion,
                channel_map=channel_map,
                metadata=metadata
            )

            # Standardize column naming (HbO/HHb -> _oxy/_deoxy)
            conc_raw = self._standardize_column_names(conc_raw)

            # Debug: log column names after standardization
            logger.debug(f"Concentration columns after standardization: {conc_raw.columns.tolist()[:10]}...")

            # -------------------------------------------------------------
            # RAW CONCENTRATION PLOTS - only plot if requested
            # -------------------------------------------------------------
            if plot_raw:
                self._plot_raw_all_channels(conc_raw, out_dir, basename_suffix, events)
                self._plot_raw_regions(conc_raw, out_dir, basename_suffix, events, channel_map)
                self._plot_raw_overall(conc_raw, out_dir, basename_suffix, events)

            # -------------------------------------------------------------
            # PROCESSING PIPELINE (on the original OD data)
            # -------------------------------------------------------------
            df_conc, sqi_info = self._run_pipeline(df_od, events, metadata, channel_map, apply_sqi_filter)

            if df_conc is None:
                return None

            # -------------------------------------------------------------
            # REGION AVERAGING (HbO/HbR)
            # -------------------------------------------------------------
            reg_df = self.channel_averager.average_regions(df_conc)
            reg_df = self.channel_averager.average_hemispheres(reg_df)

            # Standardize region column naming
            reg_df = self._standardize_column_names(reg_df)

            # Merge region signals into main dataframe
            df_final = pd.concat([df_conc, reg_df], axis=1)

            # Add grand mean
            oxy_cols = [c for c in df_final.columns if c.endswith("_oxy")]
            deoxy_cols = [c for c in df_final.columns if c.endswith("_deoxy")]

            if oxy_cols:
                df_final["grand_oxy"] = df_final[oxy_cols].mean(axis=1)
            else:
                logger.warning("No _oxy columns found for grand mean calculation")
                df_final["grand_oxy"] = 0

            if deoxy_cols:
                df_final["grand_deoxy"] = df_final[deoxy_cols].mean(axis=1)
            else:
                logger.warning("No _deoxy columns found for grand mean calculation")
                df_final["grand_deoxy"] = 0

            # -------------------------------------------------------------
            # PROCESSED PLOTS (with events!)
            # -------------------------------------------------------------
            self._plot_processed(df_final, out_dir, basename_suffix, events)

            # -------------------------------------------------------------
            # SAVE CSV
            # -------------------------------------------------------------
            df_final.to_csv(os.path.join(out_dir, f"{basename_suffix}_processed.csv"), index=False)

            # -------------------------------------------------------------
            # SAVE SQI REPORT (if SQI was computed)
            # -------------------------------------------------------------
            if sqi_info is not None:
                self._save_sqi_report(sqi_info, out_dir, basename_suffix, apply_sqi_filter)

            return df_final

        except Exception as exc:
            logger.error(f"Error processing {file_path}: {exc}", exc_info=True)
            self.warning_files.append((file_path, str(exc)))
            return None

    # -------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------

    def _create_output_dir(self, output_base: str, input_base: str, file_path: str, apply_sqi_filter: bool) -> str:
        """Create output directory with subfolder for SQI vs no-SQI."""
        rel = os.path.relpath(os.path.dirname(file_path), start=input_base)

        # Add subdirectory for SQI vs no-SQI
        if apply_sqi_filter:
            out = os.path.join(output_base, rel, "with_SQI_filtering")
        else:
            out = os.path.join(output_base, rel, "no_SQI_filtering")

        os.makedirs(out, exist_ok=True)
        return out

    def _drop_initial_second(self, df: pd.DataFrame, events: pd.DataFrame):
        drop = int(self.fs)
        if len(df) > drop:
            df = df.iloc[drop:].reset_index(drop=True)
            df["Sample number"] = np.arange(len(df))

            if not events.empty:
                events = events[events["Sample number"] >= drop].copy()
                events["Sample number"] -= drop

        return df, events

    # -------------------------------------------------------------------
    # RAW OD PLOTTING
    # -------------------------------------------------------------------
    def _plot_raw_od(self, df_od, out_dir, basename, events):
        try:
            self.visualizer_od.plot_raw_od(
                data=df_od,
                output_path=os.path.join(out_dir, f"{basename}_raw_OD.pdf"),
                y_limits=None  # Let matplotlib auto-scale
            )
        except Exception as e:
            logger.warning(f"Raw OD plot failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # -------------------------------------------------------------------
    # MAIN PIPELINE LOGIC
    # -------------------------------------------------------------------
    def _run_pipeline(self, df_od, events, metadata, channel_map, apply_sqi_filter: bool = True) -> Tuple[
        Optional[pd.DataFrame], Optional[Dict]]:
        """
        Run the processing pipeline.

        Returns:
            Tuple of (processed_dataframe, sqi_info_dict)
            sqi_info_dict contains SQI scores and excluded channels for reporting
        """
        sqi_info = None

        # -------------------------------------------------------------
        # 1. TDDR on OD data (motion artifact correction)
        # -------------------------------------------------------------
        logger.info("Applying TDDR to OD data")
        df_od_corrected = tddr(df_od.filter(regex="WL|Sample"), sample_rate=self.fs)

        # Keep non-numeric columns from original
        for col in df_od.columns:
            if col not in df_od_corrected.columns:
                df_od_corrected[col] = df_od[col]

        # DEBUG: Check OD values after TDDR
        logger.info(f"DEBUG: After TDDR - OD value ranges (first 3 columns):")
        od_cols = [c for c in df_od_corrected.columns if c.startswith("D")]
        for col in od_cols[:3]:
            logger.info(
                f"  {col}: min={df_od_corrected[col].min():.4f}, max={df_od_corrected[col].max():.4f}, mean={df_od_corrected[col].mean():.4f}")

        # -------------------------------------------------------------
        # 2. OD → concentration
        # -------------------------------------------------------------
        try:
            # Filter for WL columns
            od_data = df_od_corrected.filter(regex="WL")
            if od_data.empty:
                logger.error(f"No WL columns found in OD data. Available columns: {df_od_corrected.columns.tolist()}")
                return None, None

            # DEBUG: Check OD values before conversion
            logger.info(f"DEBUG: OD data shape: {od_data.shape}")
            logger.info(f"DEBUG: OD value ranges (first 3 columns):")
            for col in list(od_data.columns)[:3]:
                logger.info(
                    f"  {col}: min={od_data[col].min():.4f}, max={od_data[col].max():.4f}, mean={od_data[col].mean():.4f}")

            conc = convert_od_to_concentration(
                od_data,
                channel_map=channel_map,
                metadata=metadata
            )

            logger.info(f"DEBUG: Concentration columns created: {conc.columns.tolist()[:10]}...")

            # DEBUG: Check concentration values
            logger.info(f"DEBUG: Concentration data shape: {conc.shape}")
            logger.info(f"DEBUG: Concentration value ranges (first 4 columns):")
            for col in list(conc.columns)[:4]:
                logger.info(
                    f"  {col}: min={conc[col].min():.4f}, max={conc[col].max():.4f}, mean={conc[col].mean():.4f}")

        except Exception as e:
            logger.error(f"OD to concentration conversion failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

        # -------------------------------------------------------------
        # 3. SQI: Evaluate signal quality for each channel
        # -------------------------------------------------------------
        # Always compute SQI for reporting, but only exclude channels if apply_sqi_filter=True
        sqi_info = self._compute_sqi(df_od_corrected, conc, channel_map)

        if apply_sqi_filter:
            bad_channels = sqi_info['excluded_columns']
            if bad_channels:
                logger.info(
                    f"Excluding {len(bad_channels)} columns ({len(bad_channels) // 2} channels) due to SQI < {self.sci_threshold}")
                conc = conc.drop(columns=bad_channels, errors="ignore")
        else:
            logger.info("Skipping SQI filtering (apply_sqi_filter=False) - keeping all channels")
            logger.info(f"  (Would have excluded {len(sqi_info['excluded_columns'])} columns if filtering was enabled)")

        # -------------------------------------------------------------
        # 4. Short-channel regression (SCR)
        # -------------------------------------------------------------
        conc = self._apply_scr(conc, channel_map, metadata)

        # DEBUG: After SCR
        logger.info(f"DEBUG: After SCR - value ranges (first 4 columns):")
        for col in list(conc.columns)[:4]:
            logger.info(f"  {col}: min={conc[col].min():.4f}, max={conc[col].max():.4f}, mean={conc[col].mean():.4f}")

        # -------------------------------------------------------------
        # 5. FIR bandpass filter (0.01–0.1 Hz)
        # -------------------------------------------------------------
        conc_filtered = pd.DataFrame(
            fir_filter(conc,
                       order=1000,
                       Wn=[0.01, 0.1],
                       fs=int(self.fs)),
            columns=conc.columns,
            index=conc.index,
        )

        # DEBUG: After filtering
        logger.info(f"DEBUG: After FIR filter - value ranges (first 4 columns):")
        for col in list(conc_filtered.columns)[:4]:
            logger.info(
                f"  {col}: min={conc_filtered[col].min():.4f}, max={conc_filtered[col].max():.4f}, mean={conc_filtered[col].mean():.4f}")

        # -------------------------------------------------------------
        # 6. Baseline correction
        # -------------------------------------------------------------
        conc_base = self._apply_baseline_correction(conc_filtered, events)

        # DEBUG: After baseline
        logger.info(f"DEBUG: After baseline - value ranges (first 4 columns):")
        for col in list(conc_base.columns)[:4]:
            logger.info(
                f"  {col}: min={conc_base[col].min():.4f}, max={conc_base[col].max():.4f}, mean={conc_base[col].mean():.4f}")

        # Standardize column naming (HbO/HHb -> _oxy/_deoxy)
        conc_base = self._standardize_column_names(conc_base)

        return conc_base, sqi_info

    # -------------------------------------------------------------------
    # SQI
    # -------------------------------------------------------------------
    def _compute_sqi(self, df_od, df_conc, channel_map) -> Dict:
        """
        Compute SQI for each channel.

        Returns:
            Dictionary containing:
            - 'scores': List of (channel_idx, score) tuples
            - 'excluded_columns': List of column names that would be excluded
            - 'excluded_channels': List of channel indices that would be excluded
            - 'threshold': The SQI threshold used
        """
        bad_channels = []
        bad_channel_indices = []
        sqi_scores = []

        for ch_idx, info in channel_map.items():
            if "columns" not in info:
                logger.warning(f"Channel {ch_idx} missing 'columns' key in channel_map")
                continue

            wavelengths = sorted(info["columns"].keys())
            if len(wavelengths) != 2:
                continue

            wl1, wl2 = wavelengths

            cols_wl1 = info["columns"].get(wl1, [])
            cols_wl2 = info["columns"].get(wl2, [])

            if not cols_wl1 or not cols_wl2:
                continue

            col1 = cols_wl1[0]
            col2 = cols_wl2[0]

            # Use helper to get correct column names (handles both naming conventions)
            hbo_col, hhb_col = self._get_hbo_hhb_colnames(df_conc, ch_idx)

            if hbo_col is None or hhb_col is None:
                logger.warning(f"Channel {ch_idx}: HbO/HHb columns not found in concentration data")
                continue

            OD1 = df_od[col1].values if col1 in df_od.columns else None
            OD2 = df_od[col2].values if col2 in df_od.columns else None

            if OD1 is None or OD2 is None:
                continue

            # Check for NaN/Inf in OD data
            if not np.all(np.isfinite(OD1)):
                logger.warning(f"CH{ch_idx}: OD1 contains NaN/Inf - marking as bad")
                bad_channels.extend([hbo_col, hhb_col])
                bad_channel_indices.append(ch_idx)
                sqi_scores.append((ch_idx, 0.0, "NaN/Inf in OD1"))
                continue

            if not np.all(np.isfinite(OD2)):
                logger.warning(f"CH{ch_idx}: OD2 contains NaN/Inf - marking as bad")
                bad_channels.extend([hbo_col, hhb_col])
                bad_channel_indices.append(ch_idx)
                sqi_scores.append((ch_idx, 0.0, "NaN/Inf in OD2"))
                continue

            oxy = df_conc[hbo_col].values
            dxy = df_conc[hhb_col].values

            # Check for NaN/Inf in concentration data
            if not np.all(np.isfinite(oxy)) or not np.all(np.isfinite(dxy)):
                logger.warning(f"CH{ch_idx}: Concentration contains NaN/Inf - marking as bad")
                bad_channels.extend([hbo_col, hhb_col])
                bad_channel_indices.append(ch_idx)
                sqi_scores.append((ch_idx, 0.0, "NaN/Inf in concentration"))
                continue

            try:
                score = SQI(OD1, OD2, oxy, dxy, Fs=self.fs)

                if score < self.sci_threshold:
                    bad_channels.extend([hbo_col, hhb_col])
                    bad_channel_indices.append(ch_idx)
                    sqi_scores.append((ch_idx, score, "Below threshold"))
                else:
                    sqi_scores.append((ch_idx, score, "Pass"))

            except Exception as e:
                logger.warning(f"CH{ch_idx}: SQI calculation failed: {e} - marking as bad")
                bad_channels.extend([hbo_col, hhb_col])
                bad_channel_indices.append(ch_idx)
                sqi_scores.append((ch_idx, 0.0, f"Error: {str(e)}"))

        # Log SQI score statistics
        passing_scores = [s for ch, s, status in sqi_scores if status == "Pass"]
        failing_scores = [s for ch, s, status in sqi_scores if status != "Pass"]

        if sqi_scores:
            all_scores = [s for _, s, _ in sqi_scores]
            logger.info(f"SQI scores - min: {min(all_scores):.2f}, max: {max(all_scores):.2f}, "
                        f"mean: {np.mean(all_scores):.2f}, median: {np.median(all_scores):.2f}")
            logger.info(
                f"Channels: {len(passing_scores)} passed SQI, {len(failing_scores)} failed (threshold={self.sci_threshold})")

        return {
            'scores': sqi_scores,
            'excluded_columns': list(set(bad_channels)),
            'excluded_channels': list(set(bad_channel_indices)),
            'threshold': self.sci_threshold,
            'num_passed': len(passing_scores),
            'num_failed': len(failing_scores),
        }

    # -------------------------------------------------------------------
    # SQI REPORT
    # -------------------------------------------------------------------
    def _save_sqi_report(self, sqi_info: Dict, out_dir: str, basename: str, filtering_applied: bool):
        """Save SQI scores and exclusion info to a CSV file."""
        try:
            # Create DataFrame from scores
            scores_data = []
            for ch_idx, score, status in sqi_info['scores']:
                scores_data.append({
                    'Channel': ch_idx,
                    'SQI_Score': score,
                    'Status': status,
                    'Excluded': ch_idx in sqi_info['excluded_channels'],
                    'Threshold': sqi_info['threshold'],
                })

            scores_df = pd.DataFrame(scores_data)
            scores_df = scores_df.sort_values('Channel')

            # Add summary row
            summary_df = pd.DataFrame([{
                'Channel': 'SUMMARY',
                'SQI_Score': np.mean([s for _, s, _ in sqi_info['scores']]) if sqi_info['scores'] else 0,
                'Status': f"Passed: {sqi_info['num_passed']}, Failed: {sqi_info['num_failed']}",
                'Excluded': f"Filtering {'Applied' if filtering_applied else 'NOT Applied'}",
                'Threshold': sqi_info['threshold'],
            }])

            final_df = pd.concat([scores_df, summary_df], ignore_index=True)

            # Save
            report_path = os.path.join(out_dir, f"{basename}_SQI_report.csv")
            final_df.to_csv(report_path, index=False)
            logger.info(f"SQI report saved to {report_path}")

        except Exception as e:
            logger.warning(f"Failed to save SQI report: {e}")

    # -------------------------------------------------------------------
    # SCR
    # -------------------------------------------------------------------
    def _apply_scr(self, conc_df, channel_map, metadata):
        """
        Apply short-channel regression using channel_map to identify short channels.

        NOTE: With the corrected loaders.py, short channels are now correctly
        identified as CH 4, 8, 14, 17, 18, 21 (based on XML template).
        """
        # Get short and long channel indices from channel_map
        short_channels = [
            idx for idx, info in channel_map.items()
            if info.get("short_channel", False)
        ]

        long_channels = [
            idx for idx in channel_map.keys()
            if idx not in short_channels
        ]

        logger.info(f"SCR channel classification:")
        logger.info(f"  Short channels (10mm): {sorted(short_channels)}")
        logger.info(f"  Long channels (30-35mm): {sorted(long_channels)}")

        if not short_channels:
            logger.info("No short channels found, skipping SCR")
            return conc_df

        # Build list of columns - handle both naming conventions
        long_cols = []
        short_cols = []

        for idx in long_channels:
            hbo, hhb = self._get_hbo_hhb_colnames(conc_df, idx)
            if hbo and hbo in conc_df.columns:
                long_cols.append(hbo)
            if hhb and hhb in conc_df.columns:
                long_cols.append(hhb)

        for idx in short_channels:
            hbo, hhb = self._get_hbo_hhb_colnames(conc_df, idx)
            if hbo and hbo in conc_df.columns:
                short_cols.append(hbo)
            if hhb and hhb in conc_df.columns:
                short_cols.append(hhb)

        if not (long_cols and short_cols):
            logger.info(f"SCR skipped: long_cols={len(long_cols)}, short_cols={len(short_cols)}")
            return conc_df

        logger.info(f"Running SCR with {len(long_cols)} long columns and {len(short_cols)} short columns")

        # Pass DataFrames, not numpy arrays
        scr_arr = scr_regression(
            conc_df[long_cols],
            conc_df[short_cols]
        )

        scr_df = pd.DataFrame(scr_arr, columns=long_cols, index=conc_df.index)

        # Keep only cleaned long-channel columns + untouched extras
        untouched = conc_df.drop(columns=long_cols + short_cols, errors="ignore")

        return pd.concat([scr_df, untouched], axis=1)

    # -------------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------------
    def _apply_baseline_correction(self, df, events, baseline_duration=20.0):
        """
        Apply baseline correction using event markers or first N seconds.
        Looks for markers like 'S1', 'baseline', or uses initial period.
        """
        try:
            if not events.empty and 'Event' in events.columns:
                # Clean up event strings (remove whitespace)
                events['Event'] = events['Event'].str.strip()

                # Look for common baseline markers
                # S1 = start of task (end of baseline)
                # Or explicit baseline markers
                baseline_end_markers = events[
                    events['Event'].str.match(r'^S1$|^Task.*Start$|^Baseline.*End$',
                                              case=False, na=False)
                ]

                if not baseline_end_markers.empty:
                    # Found a marker - use period BEFORE this marker as baseline
                    baseline_end = baseline_end_markers.iloc[0]["Sample number"]

                    # Baseline starts after initial dropout (we already dropped 1 second)
                    baseline_start = 4

                    # Use everything from start until the marker as baseline
                    # (but cap at baseline_duration if the marker is very late)
                    max_baseline_end = int(baseline_duration * self.fs)

                    if baseline_end > max_baseline_end:
                        # Marker is late, just use first N seconds
                        logger.info(f"Task marker 'S1' found at sample {baseline_end}, "
                                    f"but using first {baseline_duration}s as baseline")
                        end = max_baseline_end
                    else:
                        # Use period up to the marker
                        end = baseline_end - 1  # End just before task starts
                        actual_duration = (end - baseline_start) / self.fs
                        logger.info(f"Using pre-task baseline: {actual_duration:.1f}s "
                                    f"(samples {baseline_start} to {end})")

                    start = baseline_start
                else:
                    # No markers found, use first N seconds
                    start = 4
                    end = int(baseline_duration * self.fs)
                    logger.warning(f"No baseline markers found, using first {baseline_duration}s")
                    logger.info(f"Baseline period: samples {start} to {end}")
            else:
                # No events, use first N seconds
                start = 4
                end = int(baseline_duration * self.fs)
                logger.warning(f"No events DataFrame, using first {baseline_duration}s")
                logger.info(f"Baseline period: samples {start} to {end}")

            # Create baseline event markers for the correction function
            bdf = pd.DataFrame({
                "Sample number": [start, end],
                "Event": ["BaselineStart", "BaselineEnd"]
            })

            return baseline_subtraction(df, bdf)

        except Exception as e:
            logger.warning(f"Baseline correction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return df

    # -------------------------------------------------------------------
    # RAW CONCENTRATION PLOTTING
    # -------------------------------------------------------------------
    def _plot_raw_all_channels(self, conc_raw, out_dir, basename, events):
        try:
            self.visualizer_conc.plot_raw_all_channels(
                data=conc_raw,
                output_path=os.path.join(out_dir, f"{basename}_raw_all_channels.pdf"),
                y_limits=None
            )
        except Exception as e:
            logger.warning(f"Raw all channels plot failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _plot_raw_regions(self, conc_raw, out_dir, basename, events, channel_map):
        """Plot raw regions with better debugging."""
        try:
            logger.info(f"=== DEBUG: Plotting raw regions ===")
            logger.info(f"Data shape: {conc_raw.shape}")
            logger.info(f"Data columns: {conc_raw.columns.tolist()}")

            # First check if we have region columns
            region_cols = [c for c in conc_raw.columns if '_oxy' in c and not c.startswith('CH')]
            logger.info(f"Found {len(region_cols)} region oxy columns: {region_cols}")

            if not region_cols:
                logger.warning("No region columns found! Creating regional averages...")
                # Try to create regional averages from channel data
                reg_df = self.channel_averager.average_regions(conc_raw)
                logger.info(f"Created region dataframe with columns: {reg_df.columns.tolist()}")

                # Pass DataFrame directly to visualizer
                self.visualizer_conc.plot_raw_regions(
                    regional_data=reg_df,
                    output_path=os.path.join(out_dir, f"{basename}_raw_regions.pdf"),
                    y_limits=None,
                    channel_map=channel_map
                )
            else:
                # We have region columns, use them directly
                self.visualizer_conc.plot_raw_regions(
                    regional_data=conc_raw,
                    output_path=os.path.join(out_dir, f"{basename}_raw_regions.pdf"),
                    y_limits=None,
                    channel_map=channel_map
                )

        except Exception as e:
            logger.error(f"Raw regions plot failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _plot_raw_overall(self, conc_raw, out_dir, basename, events):
        try:
            # Need to add grand mean columns
            conc_with_grand = conc_raw.copy()

            # Get all oxy/deoxy columns (handle both naming conventions)
            oxy_cols = [c for c in conc_raw.columns if c.endswith("_oxy") or c.endswith(" oxy") or 'HbO' in c]
            deoxy_cols = [c for c in conc_raw.columns if c.endswith("_deoxy") or c.endswith(" deoxy") or 'HHb' in c]

            if not oxy_cols or not deoxy_cols:
                logger.warning(f"No oxy/deoxy columns found. Available columns: {conc_raw.columns.tolist()}")
                return

            conc_with_grand["grand_oxy"] = conc_raw[oxy_cols].mean(axis=1)
            conc_with_grand["grand_deoxy"] = conc_raw[deoxy_cols].mean(axis=1)

            # Debug: Check if grand means have data
            logger.info(
                f"Grand oxy range: {conc_with_grand['grand_oxy'].min():.2f} to {conc_with_grand['grand_oxy'].max():.2f}")
            logger.info(
                f"Grand deoxy range: {conc_with_grand['grand_deoxy'].min():.2f} to {conc_with_grand['grand_deoxy'].max():.2f}")

            self.visualizer_conc.plot_raw_overall(
                data=conc_with_grand,
                output_path=os.path.join(out_dir, f"{basename}_raw_overall.pdf"),
                y_limits=None,
                events=events  # Pass events for plotting
            )
        except Exception as e:
            logger.warning(f"Raw overall plot failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # -------------------------------------------------------------------
    # PROCESSED PLOTS
    # -------------------------------------------------------------------
    def _plot_processed(self, df_final, out_dir, basename, events):
        try:
            self.visualizer_conc.plot_processed_overall(
                data=df_final,
                output_path=os.path.join(out_dir, f"{basename}_processed_overall.pdf"),
                y_limits=None,
                events=events  # Pass events for plotting
            )
        except Exception as e:
            logger.warning(f"Processed overall plot failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        try:
            self.visualizer_conc.plot_processed_regions(
                data=df_final,
                output_path=os.path.join(out_dir, f"{basename}_processed_regions.pdf"),
                y_limits=None
            )
        except Exception as e:
            logger.warning(f"Processed regions plot failed: {e}")
            import traceback
            logger.error(traceback.format_exc())