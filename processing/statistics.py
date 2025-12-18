# statistics.py - UPDATED FOR DUAL-PASS PROCESSING WITH BACKWARD COMPATIBILITY
import os
import pandas as pd
from typing import List, Dict, Tuple
import logging
import traceback
import numpy as np

logger = logging.getLogger(__name__)

# Import your channel utils to get correct region names
try:
    from read.channel_utils import CH_REGION_MAP, CH_REGION_MAP_COMBINED

    HAS_CHANNEL_UTILS = True
except ImportError:
    HAS_CHANNEL_UTILS = False
    logger.warning("Could not import channel_utils, using default region names")

original_read_csv = pd.read_csv


def safe_read_csv(filepath_or_buffer, *args, **kwargs):
    """Wrapper for pd.read_csv that rejects .txt files"""
    if isinstance(filepath_or_buffer, str):
        if filepath_or_buffer.endswith('.txt'):
            error_msg = f"EMERGENCY PATCH: Blocked attempt to read .txt file: {filepath_or_buffer}"
            logger.error(error_msg)
            print("=" * 80)
            traceback.print_stack()
            print("=" * 80)
            raise ValueError(error_msg)
    return original_read_csv(filepath_or_buffer, *args, **kwargs)


pd.read_csv = safe_read_csv
print("EMERGENCY PATCH ACTIVE: Will block any .txt file access via pandas")


class StatisticsCalculator:
    """
    UPDATED: Handles dual-pass processing statistics (with and without SQI filtering).
    Also maintains backward compatibility with existing pipeline_manager.
    """

    def __init__(self, input_base_dir: str = ""):
        """
        Initialize the statistics calculator.

        Args:
            input_base_dir: Base directory for input files
        """
        self.input_base_dir = input_base_dir

        # Get actual region names from channel_utils if available
        if HAS_CHANNEL_UTILS:
            # Get hemisphere-specific regions from CH_REGION_MAP
            self.required_regions = list(CH_REGION_MAP.keys())  # PFC_R, PFC_L, SMA_L, etc.
            self.combined_regions = list(CH_REGION_MAP_COMBINED.keys())  # PFC, SMA, M1, etc.
            logger.info(f"Using regions from channel_utils: {self.required_regions}")
            logger.info(f"Combined regions: {self.combined_regions}")
        else:
            # Fallback to original list
            self.required_regions = ['pfc', 'sma', 'm1', 's1', 'v1']
            self.combined_regions = ['PFC', 'SMA', 'M1', 'S1', 'V1']
            logger.warning("Using default region names (channel_utils not available)")

    def calculate_subject_y_limits(self, subject_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate y-axis limits for plotting"""
        try:
            # Select only hemoglobin columns
            signal_cols = [col for col in subject_data.columns
                           if 'HbO' in col or 'HHb' in col or 'oxy' in col or 'deoxy' in col]

            if not signal_cols:
                return (-1, 1)

            # Convert to numeric and drop NA
            numeric_data = subject_data[signal_cols].apply(pd.to_numeric, errors='coerce')
            numeric_data = numeric_data.dropna()

            if numeric_data.empty:
                return (-1, 1)

            # Calculate max absolute value with buffer
            max_abs = max(abs(numeric_data.max().max()),
                          abs(numeric_data.min().min()))
            buffer = max_abs * 0.2
            return (-max_abs - buffer, max_abs + buffer)

        except Exception as e:
            logger.error(f"Error calculating y-limits: {str(e)}")
            return (-1, 1)

    # =========================================================================
    # MAIN NEW METHOD FOR DUAL-PASS PROCESSING
    # =========================================================================
    def collect_dual_pass_statistics(self, output_base_dir: str) -> pd.DataFrame:
        """
        NEW METHOD: Collect statistics for BOTH SQI-filtered and non-filtered data.

        This function traverses the output directory structure created by the dual-pass processor
        and generates comprehensive statistics for both passes.

        Args:
            output_base_dir: Base output directory containing 'with_SQI_filtering'
                            and 'no_SQI_filtering' subdirectories

        Returns:
            DataFrame with statistics for both processing passes
        """
        logger.info("=== COLLECTING DUAL-PASS STATISTICS ===")
        logger.info(f"Searching in: {output_base_dir}")

        # STEP 1: Find all processed CSV files in both SQI and no-SQI folders
        sqi_files = []
        no_sqi_files = []

        # Walk through the output directory structure
        for root, dirs, files in os.walk(output_base_dir):
            for file in files:
                if file.endswith('_processed.csv'):
                    full_path = os.path.join(root, file)

                    # Check if this is SQI-filtered or no-SQI
                    if 'with_SQI_filtering' in full_path or '_SQI_processed' in file:
                        sqi_files.append(full_path)
                        logger.debug(f"Found SQI file: {os.path.relpath(full_path, output_base_dir)}")
                    elif 'no_SQI_filtering' in full_path or '_NoSQI_processed' in file:
                        no_sqi_files.append(full_path)
                        logger.debug(f"Found no-SQI file: {os.path.relpath(full_path, output_base_dir)}")
                    else:
                        logger.warning(f"Could not determine processing type for: {full_path}")

        logger.info(f"Found {len(sqi_files)} SQI-filtered files")
        logger.info(f"Found {len(no_sqi_files)} non-SQI-filtered files")

        if not sqi_files and not no_sqi_files:
            logger.error("No processed CSV files found!")
            return pd.DataFrame()

        # STEP 2: Process SQI-filtered files
        sqi_stats = []
        if sqi_files:
            logger.info("Processing SQI-filtered files...")
            for csv_file in sqi_files:
                stats = self._process_single_csv_file(csv_file, sqi_filtered=True)
                if stats:
                    sqi_stats.append(stats)

        # STEP 3: Process non-SQI-filtered files
        no_sqi_stats = []
        if no_sqi_files:
            logger.info("Processing non-SQI-filtered files...")
            for csv_file in no_sqi_files:
                stats = self._process_single_csv_file(csv_file, sqi_filtered=False)
                if stats:
                    no_sqi_stats.append(stats)

        # STEP 4: Combine all statistics
        all_stats = sqi_stats + no_sqi_stats

        if not all_stats:
            logger.warning("No statistics were generated")
            return pd.DataFrame()

        result_df = pd.DataFrame(all_stats)
        logger.info(f"Generated statistics for {len(result_df)} files")
        logger.info(f"SQI-filtered: {len(sqi_stats)}, Non-SQI-filtered: {len(no_sqi_stats)}")

        return result_df

    # =========================================================================
    # BACKWARD COMPATIBILITY METHODS
    # =========================================================================
    def collect_statistics_from_processed_files(self, processed_files: List[str]) -> pd.DataFrame:
        """
        ORIGINAL METHOD: For backward compatibility with pipeline_manager.
        Now enhanced to handle dual-pass files.
        """
        logger.info("Using backward-compatible method: collect_statistics_from_processed_files")

        # Check if we received .txt files (old bug) or actual CSV files
        txt_files = [f for f in processed_files if f.endswith('.txt')]

        if txt_files:
            logger.error(f"ERROR: Received {len(txt_files)} .txt files!")
            logger.error("This indicates a bug in the calling code.")

            # Try to find CSV files independently
            csv_files = self._find_processed_csv_files()
            if csv_files:
                logger.info(f"Found {len(csv_files)} CSV files instead")
                return self._process_file_list(csv_files)
            else:
                logger.error("No CSV files found")
                return pd.DataFrame()

        # Process the list of files
        return self._process_file_list(processed_files)

    def collect_statistics(self, processed_files: List[str], output_base_dir: str) -> pd.DataFrame:
        """
        [Deprecated] Original method - now uses new dual-pass method
        """
        logger.warning("DEPRECATED collect_statistics method called - switching to dual-pass collection")
        return self.collect_dual_pass_statistics(output_base_dir)

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================
    def _process_file_list(self, file_list: List[str]) -> pd.DataFrame:
        """Process a list of CSV files (handles both SQI and non-SQI)"""
        all_stats = []

        for csv_file in file_list:
            if not csv_file.endswith('.csv') or not os.path.exists(csv_file):
                logger.warning(f"Skipping invalid file: {csv_file}")
                continue

            # Determine if this file is SQI-filtered or not
            if '_SQI_processed' in csv_file or 'with_SQI_filtering' in csv_file:
                sqi_filtered = True
            elif '_NoSQI_processed' in csv_file or 'no_SQI_filtering' in csv_file:
                sqi_filtered = False
            else:
                # Can't determine - default to assuming it's SQI-filtered
                # (this handles old single-pass files)
                sqi_filtered = True
                logger.warning(f"Could not determine SQI status for {csv_file}, assuming SQI-filtered")

            stats = self._process_single_csv_file(csv_file, sqi_filtered)
            if stats:
                all_stats.append(stats)

        if not all_stats:
            logger.warning("No statistics were generated from file list")
            return pd.DataFrame()

        result_df = pd.DataFrame(all_stats)
        logger.info(f"Generated statistics for {len(result_df)} files from file list")
        return result_df

    def _process_single_csv_file(self, csv_file: str, sqi_filtered: bool) -> Dict:
        """
        Process a single CSV file and extract statistics.

        Args:
            csv_file: Path to the CSV file
            sqi_filtered: Whether this file contains SQI-filtered data

        Returns:
            Dictionary of statistics or None if processing failed
        """
        try:
            logger.info(f"Processing: {os.path.basename(csv_file)}")

            # Read CSV file
            df = pd.read_csv(csv_file)
            if df.empty:
                logger.warning(f"Empty CSV file: {csv_file}")
                return None

            # Extract metadata
            filename = os.path.basename(csv_file)
            subject_id = self._extract_subject_id(filename)
            visit = self._extract_visit_from_path(csv_file)
            condition = self._extract_condition(filename)

            # Calculate statistics
            total_samples = len(df)
            half = total_samples // 2

            stats = {
                'Subject': subject_id,
                'Timepoint': visit,
                'Condition': condition,
                'SQI_Filtered': 'Yes' if sqi_filtered else 'No',
                'SourceFile': filename,
                'TotalSamples': total_samples
            }

            # Log what columns we have for debugging
            logger.debug(f"File columns: {df.columns.tolist()[:20]}...")

            # Grand averages
            grand_stats = self._calculate_grand_stats(df, half)
            stats.update(grand_stats)

            # Regional averages - now using actual region detection
            regional_stats = self._calculate_regional_stats(df, half)
            stats.update(regional_stats)

            logger.info(f"Successfully processed: {filename} (SQI: {sqi_filtered})")
            return stats

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _find_processed_csv_files(self) -> List[str]:
        """Find processed CSV files independently (legacy method)"""
        search_dirs = []

        if self.input_base_dir:
            parent_dir = os.path.dirname(self.input_base_dir)
            search_dirs.extend([
                os.path.join(parent_dir, 'output'),
                os.path.join(parent_dir, 'results'),
                os.path.join(parent_dir, 'processed'),
                os.path.join(parent_dir, 'auttest'),
                '/Users/tsujik/Documents/auttest'
            ])

        # Remove non-existent directories
        search_dirs = [d for d in search_dirs if os.path.exists(d)]

        # Find all processed CSV files
        csv_files = []
        for search_dir in search_dirs:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith("_processed.csv") and "bad_SCI" not in file:
                        csv_files.append(os.path.join(root, file))

        return list(set(csv_files))

    def _extract_subject_id(self, filename: str) -> str:
        """Extract subject ID from filename"""
        # Handle both naming conventions: with and without SQI suffix
        clean_name = filename.replace('_SQI_processed.csv', '').replace('_NoSQI_processed.csv', '')
        parts = clean_name.split('_')

        # Try to extract subject pattern (e.g., "Subject1_Visit1")
        if len(parts) >= 2:
            # Look for pattern like "Subject1" followed by "Visit1"
            if any(substr.lower().startswith('subject') for substr in parts):
                # Find subject and visit parts
                for i, part in enumerate(parts):
                    if part.lower().startswith('subject'):
                        if i + 1 < len(parts):
                            return f"{part}_{parts[i + 1]}"
                        else:
                            return part
                return f"{parts[0]}_{parts[1]}"
        return "Unknown"

    def _extract_visit_from_path(self, file_path: str) -> str:
        """Extract visit from file path"""
        parts = file_path.split(os.sep)
        for part in parts:
            if part.startswith('Visit'):
                return part
        return "Unknown"

    def _extract_condition(self, filename: str) -> str:
        """Extract condition from filename"""
        # Remove SQI suffix first
        clean_name = filename.replace('_SQI_processed.csv', '').replace('_NoSQI_processed.csv', '')
        clean_name = clean_name.replace('_processed.csv', '')

        # Convert to lowercase for matching
        filename_lower = clean_name.lower()

        if 'cue_walking' in filename_lower or 'cue walking' in filename_lower:
            if 'dt3' in filename_lower or 'dt' in filename_lower:
                return 'LongWalk_DT'
            elif 'st' in filename_lower:
                return 'LongWalk_ST'
            else:
                return 'Cue_Walking'
        elif 'walking_st' in filename_lower or 'walkingst' in filename_lower:
            return 'LongWalk_ST'
        elif 'walking_dt' in filename_lower or 'walkingdt' in filename_lower:
            return 'LongWalk_DT'
        elif 'sitting' in filename_lower:
            return 'Sitting'
        elif 'standing' in filename_lower:
            return 'Standing'
        else:
            # Try to extract from filename pattern
            parts = clean_name.split('_')
            for part in parts:
                if part.lower() in ['st', 'dt', 'dt3']:
                    return f'LongWalk_{part.upper()}'
            return 'Unknown'

    def _calculate_grand_stats(self, df: pd.DataFrame, half: int) -> Dict:
        """Calculate grand statistics"""
        stats = {}

        # Look for grand mean columns
        grand_cols = [col for col in df.columns if 'grand_' in col.lower()]
        logger.debug(f"Found grand columns: {grand_cols}")

        for col in grand_cols:
            if 'oxy' in col.lower() or 'hbo' in col.lower():
                try:
                    stats[f'Grand Oxy Overall Mean'] = df[col].mean()
                    stats[f'Grand Oxy First Half Mean'] = df[col].iloc[:half].mean()
                    stats[f'Grand Oxy Second Half Mean'] = df[col].iloc[half:].mean()
                except Exception as e:
                    logger.warning(f"Error calculating grand oxy stats from {col}: {e}")
            elif 'deoxy' in col.lower() or 'hhb' in col.lower():
                try:
                    stats[f'Grand Deoxy Overall Mean'] = df[col].mean()
                    stats[f'Grand Deoxy First Half Mean'] = df[col].iloc[:half].mean()
                    stats[f'Grand Deoxy Second Half Mean'] = df[col].iloc[half:].mean()
                except Exception as e:
                    logger.warning(f"Error calculating grand deoxy stats from {col}: {e}")

        return stats

    def _calculate_regional_stats(self, df: pd.DataFrame, half: int) -> Dict:
        """Calculate regional statistics"""
        stats = {}

        # Get all columns that might be region columns
        all_cols = df.columns.tolist()

        # Look for oxy/deoxy columns that aren't channel columns
        region_columns = []
        for col in all_cols:
            # Skip channel columns (start with CH) and grand columns
            if col.startswith('CH') or 'grand' in col.lower() or col in ['Sample number', 'Event']:
                continue

            # Look for oxy or deoxy indicators
            if '_oxy' in col or '_deoxy' in col or ' HbO' in col or ' HHb' in col:
                region_columns.append(col)

        logger.debug(f"Found {len(region_columns)} potential region columns")

        # Group by region name
        region_data = {}
        for col in region_columns:
            # Extract region name
            region_name = None

            # Try different patterns
            if '_oxy' in col:
                region_name = col.replace('_oxy', '')
            elif '_deoxy' in col:
                region_name = col.replace('_deoxy', '')
            elif ' HbO' in col:
                region_name = col.replace(' HbO', '')
            elif ' HHb' in col:
                region_name = col.replace(' HHb', '')

            if region_name:
                # Clean up region name (remove trailing underscores, etc.)
                region_name = region_name.strip('_')

                if region_name not in region_data:
                    region_data[region_name] = {'oxy': [], 'deoxy': []}

                if '_oxy' in col or ' HbO' in col:
                    region_data[region_name]['oxy'].append(col)
                elif '_deoxy' in col or ' HHb' in col:
                    region_data[region_name]['deoxy'].append(col)

        logger.debug(f"Organized into {len(region_data)} regions")

        # Calculate statistics for each region
        for region_name, columns in region_data.items():
            region_key = region_name.upper()

            # Oxy statistics
            if columns['oxy']:
                # If multiple oxy columns, average them
                if len(columns['oxy']) == 1:
                    oxy_col = columns['oxy'][0]
                    oxy_data = df[oxy_col]
                else:
                    # Average multiple oxy columns for this region
                    oxy_data = df[columns['oxy']].mean(axis=1)

                try:
                    stats[f'{region_key} Oxy Overall Mean'] = oxy_data.mean()
                    stats[f'{region_key} Oxy First Half Mean'] = oxy_data.iloc[:half].mean()
                    stats[f'{region_key} Oxy Second Half Mean'] = oxy_data.iloc[half:].mean()
                except Exception as e:
                    logger.warning(f"Error calculating {region_key} oxy stats: {e}")

            # Deoxy statistics
            if columns['deoxy']:
                # If multiple deoxy columns, average them
                if len(columns['deoxy']) == 1:
                    deoxy_col = columns['deoxy'][0]
                    deoxy_data = df[deoxy_col]
                else:
                    # Average multiple deoxy columns for this region
                    deoxy_data = df[columns['deoxy']].mean(axis=1)

                try:
                    stats[f'{region_key} Deoxy Overall Mean'] = deoxy_data.mean()
                    stats[f'{region_key} Deoxy First Half Mean'] = deoxy_data.iloc[:half].mean()
                    stats[f'{region_key} Deoxy Second Half Mean'] = deoxy_data.iloc[half:].mean()
                except Exception as e:
                    logger.warning(f"Error calculating {region_key} deoxy stats: {e}")

        # Also try to find combined regions (PFC, SMA, M1, etc.)
        for combined_region in self.combined_regions:
            # Look for columns that match this combined region
            oxy_cols = [col for col in region_columns if combined_region.upper() in col.upper()
                        and ('_oxy' in col or ' HbO' in col)]
            deoxy_cols = [col for col in region_columns if combined_region.upper() in col.upper()
                          and ('_deoxy' in col or ' HHb' in col)]

            if oxy_cols:
                oxy_data = df[oxy_cols].mean(axis=1)
                stats[f'{combined_region}_COMBINED Oxy Overall Mean'] = oxy_data.mean()
                stats[f'{combined_region}_COMBINED Oxy First Half Mean'] = oxy_data.iloc[:half].mean()
                stats[f'{combined_region}_COMBINED Oxy Second Half Mean'] = oxy_data.iloc[half:].mean()

            if deoxy_cols:
                deoxy_data = df[deoxy_cols].mean(axis=1)
                stats[f'{combined_region}_COMBINED Deoxy Overall Mean'] = deoxy_data.mean()
                stats[f'{combined_region}_COMBINED Deoxy First Half Mean'] = deoxy_data.iloc[:half].mean()
                stats[f'{combined_region}_COMBINED Deoxy Second Half Mean'] = deoxy_data.iloc[half:].mean()

        return stats

    def create_summary_sheets(self, stats_df: pd.DataFrame, output_folder: str) -> None:
        """
        Generate summary CSV files for different conditions and SQI filtering status.
        No individual region sheets - all regions are in the main stats sheet.
        """
        if stats_df.empty:
            logger.warning("No statistics to summarize")
            return

        try:
            os.makedirs(output_folder, exist_ok=True)

            # Define output columns
            basic_cols = ['Subject', 'Timepoint', 'Condition', 'SQI_Filtered', 'SourceFile', 'TotalSamples']

            # Find available columns
            available_cols = list(stats_df.columns)
            output_cols = [col for col in basic_cols if col in available_cols]

            # Add statistical columns
            stat_cols = [col for col in available_cols if 'Mean' in col]
            output_cols.extend(sorted(stat_cols))

            # Save full statistics
            full_stats_path = os.path.join(output_folder, 'all_subjects_statistics.csv')
            stats_df[output_cols].to_csv(full_stats_path, index=False)
            logger.info(f"Saved full statistics: {full_stats_path}")

            # Save SQI-filtered vs non-filtered comparisons
            if 'SQI_Filtered' in stats_df.columns:
                # SQI-filtered only
                sqi_df = stats_df[stats_df['SQI_Filtered'] == 'Yes']
                if not sqi_df.empty:
                    sqi_path = os.path.join(output_folder, 'SQI_filtered_statistics.csv')
                    sqi_df[output_cols].to_csv(sqi_path, index=False)
                    logger.info(f"Saved SQI-filtered statistics: {sqi_path}")

                # Non-SQI-filtered only
                no_sqi_df = stats_df[stats_df['SQI_Filtered'] == 'No']
                if not no_sqi_df.empty:
                    no_sqi_path = os.path.join(output_folder, 'non_SQI_filtered_statistics.csv')
                    no_sqi_df[output_cols].to_csv(no_sqi_path, index=False)
                    logger.info(f"Saved non-SQI-filtered statistics: {no_sqi_path}")

            # Save condition-specific summaries
            if 'Condition' in stats_df.columns:
                for condition in stats_df['Condition'].unique():
                    if pd.notna(condition):
                        cond_df = stats_df[stats_df['Condition'] == condition]
                        summary_path = os.path.join(output_folder, f'summary_{condition}.csv')
                        cond_df[output_cols].to_csv(summary_path, index=False)
                        logger.info(f"Saved condition summary: {summary_path}")

            logger.info(f"Statistics columns: {output_cols}")

        except Exception as e:
            logger.error(f"Error creating summary sheets: {str(e)}")
            logger.error(traceback.format_exc())