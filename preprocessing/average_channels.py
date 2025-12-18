# average_channels.py - FIXED VERSION
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import from your channel_utils to ensure consistency
try:
    from read.channel_utils import CH_REGION_MAP, CH_REGION_MAP_COMBINED, SHORT_CHANNEL_LIST, LONG_CHANNEL_LIST

    HAS_CHANNEL_UTILS = True
except ImportError:
    HAS_CHANNEL_UTILS = False
    logger.warning("Could not import channel_utils")


class FullCapChannelAverager:
    """
    Handles channel averaging for full-head fNIRS caps.
    UPDATED: Uses correct channel numbers from channel_utils.py
    """

    def __init__(self):
        """
        Initialize with full-cap specific channel configuration.
        Uses actual channel numbers from channel_utils.py
        """
        if HAS_CHANNEL_UTILS:
            # Use actual region mapping from channel_utils.py
            self.regions = {}
            for region, channels in CH_REGION_MAP.items():
                self.regions[region] = {
                    "channels": channels,
                    "hemisphere": "right" if region.endswith('_R') else "left" if region.endswith('_L') else "midline"
                }

            # Also store combined regions
            self.combined_regions = CH_REGION_MAP_COMBINED
            logger.info(f"Using regions from channel_utils: {list(self.regions.keys())}")
        else:
            # Fallback with CORRECT channel numbers (1-based, not 0-based!)
            self.regions = {
                "PFC_R": {"channels": [1, 2, 3], "hemisphere": "right"},
                "PFC_L": {"channels": [5, 6, 7], "hemisphere": "left"},
                "SMA_L": {"channels": [14, 15], "hemisphere": "left"},
                "SMA_R": {"channels": [16, 17], "hemisphere": "right"},
                "M1_L": {"channels": [12, 13], "hemisphere": "left"},
                "M1_R": {"channels": [18, 19], "hemisphere": "right"},
                "S1_L": {"channels": [10, 11], "hemisphere": "left"},
                "S1_R": {"channels": [20, 21], "hemisphere": "right"},
                "V1_L": {"channels": [9], "hemisphere": "left"},
                "V1_R": {"channels": [22], "hemisphere": "right"}
            }
            self.combined_regions = {
                "PFC": [1, 2, 3, 5, 6, 7],
                "SMA": [14, 15, 16, 17],
                "M1": [12, 13, 18, 19],
                "S1": [10, 11, 20, 21],
                "V1": [9, 22]
            }
            logger.warning("Using fallback region definitions (channel_utils not available)")

        # Short channel mapping - updated based on your actual short channels [4, 8, 23, 24, 25, 26]
        # Map each long channel to the nearest short channel in the same region
        self.short_channel_mapping = {
            # PFC_R long channels (1,2,3) -> PFC_R short (4)
            1: 4, 2: 4, 3: 4,
            # PFC_L long channels (5,6,7) -> PFC_L short (8)
            5: 8, 6: 8, 7: 8,
            # M1_L long channels (12,13) -> M1_L short (23)
            12: 23, 13: 23,
            # M1_R long channels (18,19) -> M1_R short (26)
            18: 26, 19: 26,
            # SMA_L long channels (14,15) -> SMA_L short (25)
            14: 25, 15: 25,
            # SMA_R long channels (16,17) -> SMA_R short (24)
            16: 24, 17: 24,
            # S1_L long channels (10,11) -> nearest short (23 for M1_L)
            10: 23, 11: 23,
            # S1_R long channels (20,21) -> nearest short (26 for M1_R)
            20: 26, 21: 26,
            # V1_L long channel (9) -> nearest short (23 for M1_L)
            9: 23,
            # V1_R long channel (22) -> nearest short (26 for M1_R)
            22: 26
        }

        self.current_naming_convention = None  # Will detect automatically

    def detect_naming_convention(self, column_names: List[str]) -> Optional[str]:
        """
        Detect the naming convention used in the data columns.

        Returns:
            Either 'HbO' or 'O2Hb' convention, or None if undetermined
        """
        # First check for standard naming
        if any('HbO' in col for col in column_names):
            return 'HbO'
        elif any('O2Hb' in col for col in column_names):
            return 'O2Hb'
        # Also check for _oxy/_deoxy naming
        elif any('_oxy' in col for col in column_names):
            return '_oxy'
        elif any('_deoxy' in col for col in column_names):
            return '_deoxy'
        return None

    def get_region_channels(self, region: str, hb_type: str) -> List[str]:
        """
        Get the proper column names for a region and hemoglobin type.

        Args:
            region: Region name (e.g., "PFC_R")
            hb_type: Either "HbO" or "HHb" or "oxy" or "deoxy"

        Returns:
            List of column names matching the current naming convention
        """
        if region not in self.regions:
            raise ValueError(f"Unknown region: {region}")

        # Get channel numbers for this region
        channel_nums = self.regions[region]["channels"]

        # Determine naming convention if not already done
        if self.current_naming_convention is None:
            return []  # Can't determine without seeing columns

        # Build column names based on convention
        columns = []

        for ch_num in channel_nums:
            if self.current_naming_convention == 'HbO':
                if hb_type == 'HbO':
                    columns.append(f"CH{ch_num} HbO")
                elif hb_type == 'HHb':
                    columns.append(f"CH{ch_num} HHb")
            elif self.current_naming_convention == 'O2Hb':
                if hb_type == 'HbO':
                    columns.append(f"CH{ch_num} O2Hb")
                elif hb_type == 'HHb':
                    columns.append(f"CH{ch_num} HHb")
            elif self.current_naming_convention == '_oxy':
                if hb_type in ['HbO', 'oxy']:
                    columns.append(f"CH{ch_num}_oxy")
                elif hb_type in ['HHb', 'deoxy']:
                    columns.append(f"CH{ch_num}_deoxy")

        return columns

    def average_regions(self, df: pd.DataFrame,
                        channels_to_exclude: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Average channels by region, handling naming conventions and exclusions.

        Returns:
            DataFrame with region averages
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Detect naming convention
        self.current_naming_convention = self.detect_naming_convention(df.columns.tolist())
        logger.info(f"Detected naming convention: {self.current_naming_convention}")

        channels_to_exclude = channels_to_exclude or []
        result_data = {}

        # Preserve metadata columns if present
        for col in ['Sample number', 'Event', 'grand_oxy', 'grand_deoxy']:
            if col in df.columns:
                result_data[col] = df[col]

        # Calculate region averages
        for region in self.regions:
            # Determine column naming based on convention
            if self.current_naming_convention == '_oxy':
                oxy_suffix = '_oxy'
                deoxy_suffix = '_deoxy'
            elif self.current_naming_convention in ['HbO', 'O2Hb']:
                oxy_suffix = ' HbO' if self.current_naming_convention == 'HbO' else ' O2Hb'
                deoxy_suffix = ' HHb'
            else:
                # Try to guess
                oxy_suffix = ' HbO'
                deoxy_suffix = ' HHb'

            # Get channel numbers for this region
            channel_nums = self.regions[region]["channels"]

            # Build column names
            oxy_cols = []
            deoxy_cols = []

            for ch_num in channel_nums:
                if ch_num not in channels_to_exclude:
                    if self.current_naming_convention == '_oxy':
                        oxy_col = f"CH{ch_num}_oxy"
                        deoxy_col = f"CH{ch_num}_deoxy"
                    else:
                        oxy_col = f"CH{ch_num}{oxy_suffix}"
                        deoxy_col = f"CH{ch_num}{deoxy_suffix}"

                    if oxy_col in df.columns:
                        oxy_cols.append(oxy_col)
                    if deoxy_col in df.columns:
                        deoxy_cols.append(deoxy_col)

            # Log what we found
            logger.debug(f"Region {region}: {len(oxy_cols)} oxy cols, {len(deoxy_cols)} deoxy cols")

            # Calculate averages
            if oxy_cols:
                result_data[f"{region}_oxy"] = df[oxy_cols].mean(axis=1)
            else:
                result_data[f"{region}_oxy"] = np.nan

            if deoxy_cols:
                result_data[f"{region}_deoxy"] = df[deoxy_cols].mean(axis=1)
            else:
                result_data[f"{region}_deoxy"] = np.nan

        # Also create combined region averages
        for combined_region, channels in self.combined_regions.items():
            oxy_cols = []
            deoxy_cols = []

            for ch_num in channels:
                if ch_num not in channels_to_exclude:
                    if self.current_naming_convention == '_oxy':
                        oxy_col = f"CH{ch_num}_oxy"
                        deoxy_col = f"CH{ch_num}_deoxy"
                    else:
                        oxy_suffix = ' HbO' if self.current_naming_convention == 'HbO' else ' O2Hb'
                        deoxy_suffix = ' HHb'
                        oxy_col = f"CH{ch_num}{oxy_suffix}"
                        deoxy_col = f"CH{ch_num}{deoxy_suffix}"

                    if oxy_col in df.columns:
                        oxy_cols.append(oxy_col)
                    if deoxy_col in df.columns:
                        deoxy_cols.append(deoxy_col)

            if oxy_cols:
                result_data[f"{combined_region}_oxy"] = df[oxy_cols].mean(axis=1)
            if deoxy_cols:
                result_data[f"{combined_region}_deoxy"] = df[deoxy_cols].mean(axis=1)

        result_df = pd.DataFrame(result_data, index=df.index)
        logger.info(f"Created region averages with columns: {result_df.columns.tolist()}")
        return result_df

    def average_hemispheres(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine left and right hemisphere regions into combined averages.

        Returns:
            DataFrame with additional combined hemisphere averages
        """
        result_df = df.copy()

        # Get unique region prefixes (e.g., "PFC" from "PFC_L" and "PFC_R")
        region_types = set()
        for region in self.regions:
            if '_' in region:
                base = region.split('_')[0]
                region_types.add(base)

        for region_type in region_types:
            left_region = f"{region_type}_L"
            right_region = f"{region_type}_R"

            for hb_suffix in ['_oxy', '_deoxy']:
                left_col = f"{left_region}{hb_suffix}"
                right_col = f"{right_region}{hb_suffix}"
                combined_col = f"{region_type}{hb_suffix}"

                if left_col in df.columns and right_col in df.columns:
                    result_df[combined_col] = (df[left_col] + df[right_col]) / 2
                    logger.debug(f"Created combined column: {combined_col} from {left_col} and {right_col}")
                elif left_col in df.columns:
                    result_df[combined_col] = df[left_col]
                elif right_col in df.columns:
                    result_df[combined_col] = df[right_col]

        # Also check for any existing combined columns and keep them
        for col in df.columns:
            if col not in result_df.columns:
                result_df[col] = df[col]

        return result_df

    def get_short_map(self, column_names: List[str]) -> Dict[str, str]:
        """
        Map long channels to their corresponding short channels.

        Returns:
            Dictionary mapping long channel column names to short channel column names
        """
        # Detect naming convention
        convention = self.detect_naming_convention(column_names)

        if convention == '_oxy':
            oxy_suffix = '_oxy'
            deoxy_suffix = '_deoxy'
        elif convention == 'HbO':
            oxy_suffix = ' HbO'
            deoxy_suffix = ' HHb'
        elif convention == 'O2Hb':
            oxy_suffix = ' O2Hb'
            deoxy_suffix = ' HHb'
        else:
            # Default to HbO/HHb
            oxy_suffix = ' HbO'
            deoxy_suffix = ' HHb'

        mapping = {}

        # Create mapping for both oxy and deoxy
        for long_num, short_num in self.short_channel_mapping.items():
            long_oxy = f"CH{long_num}{oxy_suffix}"
            long_deoxy = f"CH{long_num}{deoxy_suffix}"
            short_oxy = f"CH{short_num}{oxy_suffix}"
            short_deoxy = f"CH{short_num}{deoxy_suffix}"

            if (long_oxy in column_names and short_oxy in column_names):
                mapping[long_oxy] = short_oxy
            if (long_deoxy in column_names and short_deoxy in column_names):
                mapping[long_deoxy] = short_deoxy

        logger.info(f"Created short channel mapping for {len(mapping)} columns")
        return mapping

    def _safe_mean(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Calculate mean while handling empty column lists."""
        if not columns:
            return pd.Series(np.nan, index=df.index)
        return df[columns].mean(axis=1)

    def adjust_regions_to_naming(self, column_names: List[str]) -> Dict[str, List[str]]:
        """
        Adjust region channel names to match the current naming convention.
        """
        convention = self.detect_naming_convention(column_names)
        adjusted_regions = {}

        for region_name, region_data in self.regions.items():
            adjusted_channels = []

            for ch_num in region_data["channels"]:
                if convention == '_oxy':
                    oxy_col = f"CH{ch_num}_oxy"
                    deoxy_col = f"CH{ch_num}_deoxy"
                elif convention == 'HbO':
                    oxy_col = f"CH{ch_num} HbO"
                    deoxy_col = f"CH{ch_num} HHb"
                elif convention == 'O2Hb':
                    oxy_col = f"CH{ch_num} O2Hb"
                    deoxy_col = f"CH{ch_num} HHb"
                else:
                    # Try both
                    oxy_col = f"CH{ch_num} HbO"
                    deoxy_col = f"CH{ch_num} HHb"

                if oxy_col in column_names:
                    adjusted_channels.append(oxy_col)
                if deoxy_col in column_names:
                    adjusted_channels.append(deoxy_col)

            adjusted_regions[region_name] = adjusted_channels

        return adjusted_regions

    def get_all_region_columns(self) -> List[str]:
        """Get list of all expected region column names."""
        columns = []

        # Hemisphere-specific regions
        for region in self.regions:
            columns.append(f"{region}_oxy")
            columns.append(f"{region}_deoxy")

        # Combined regions
        for region in self.combined_regions:
            columns.append(f"{region}_oxy")
            columns.append(f"{region}_deoxy")

        return columns