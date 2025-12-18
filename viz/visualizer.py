# visualizer.py - FIXED VERSION (Region plots fixed)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import matplotlib as mpl
import re
from read.channel_utils import CH_REGION_MAP, SHORT_CHANNELS_BY_REGION, build_region_map_od, CH_REGION_MAP_COMBINED
import logging

mpl.rcParams['figure.max_open_warning'] = 0
logger = logging.getLogger(__name__)

# Define colors for different channel types
CHANNEL_TYPE_COLORS = {
    'PFC_LONG': '#1f77b4',
    'PFC_SHORT': '#aec7e8',
    'MOTOR_LONG': '#ff7f0e',
    'MOTOR_SHORT': '#ffbb78',
    'SHORT': '#2ca02c',
    'LONG': '#d62728',
}


class FNIRSVisualizer:
    """fNIRS visualizer with automatic y-axis scaling."""

    def __init__(self, fs: float = 50.0, data_type: str = 'concentration'):
        self.fs = fs
        self.data_type = data_type
        plt.ioff()

        # Update styles based on data type
        if data_type == 'concentration':
            self.styles = {
                'raw': {'HbO': '#E41A1C', 'HHb': '#377EB8'},
                'processed': {'HbO': '#FF6B6B', 'HHb': '#4ECDC4', 'HHb_linestyle': '--'}
            }
            self.y_label = 'μM'
        else:  # 'od'
            self.styles = {
                'raw': {'wl1': '#E41A1C', 'wl2': '#377EB8'},
                'processed': {'wl1': '#FF6B6B', 'wl2': '#4ECDC4', 'wl2_linestyle': '--'}
            }
            self.y_label = 'OD'

    def _get_time(self, length: int) -> np.ndarray:
        """Return a time axis in seconds for data of given length."""
        return np.arange(length, dtype='float64') / self.fs

    def _get_wavelength_pairs(self, column_names: List[str]) -> List[Tuple[str, str]]:
        """Extract wavelength pairs from OD column names."""
        channel_groups = {}

        for col in column_names:
            if col in ["Sample number", "Event"]:
                continue

            m = re.match(r"D(\d+)_R(\d+)_S(\d+)_WL(\d+)", col)
            if m:
                key = f"D{m.group(1)}_R{m.group(2)}_S{m.group(3)}"
                if key not in channel_groups:
                    channel_groups[key] = {}
                wavelength = int(m.group(4))
                channel_groups[key][wavelength] = col

        pairs = []
        for key, wl_dict in channel_groups.items():
            if 760 in wl_dict and 850 in wl_dict:
                pairs.append((wl_dict[760], wl_dict[850]))
            elif len(wl_dict) >= 2:
                wls = sorted(wl_dict.keys())
                if len(wls) >= 2:
                    pairs.append((wl_dict[wls[0]], wl_dict[wls[1]]))

        return pairs

    def plot_raw_all_channels(
            self,
            data: pd.DataFrame,
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None,
            channel_map: Optional[Dict] = None
    ) -> None:
        """Plot every channel in its own subplot."""
        logger.info(f"=== plot_raw_all_channels called ===")
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()[:10]}...")

        time = self._get_time(len(data))

        if self.data_type == 'concentration':
            # Find oxy columns using multiple patterns
            oxy_patterns = [r'\b(HbO|O2Hb)\b', r'_oxy$', r'\soxy$']
            hbo_cols = []
            channel_numbers = []

            for pattern in oxy_patterns:
                pattern_cols = [c for c in data.columns if re.search(pattern, c, re.IGNORECASE)]
                for col in pattern_cols:
                    # Extract channel number for sorting
                    ch_match = re.search(r'CH(\d+)', col)
                    if ch_match:
                        channel_num = int(ch_match.group(1))
                        if col not in hbo_cols:
                            hbo_cols.append(col)
                            channel_numbers.append(channel_num)

            # Sort by channel number
            if hbo_cols and channel_numbers:
                sorted_pairs = sorted(zip(channel_numbers, hbo_cols))
                hbo_cols = [col for _, col in sorted_pairs]
                logger.info(f"Channels sorted numerically: {[p[0] for p in sorted_pairs]}")
            else:
                hbo_cols = sorted(set(hbo_cols))

            if not hbo_cols:
                logger.warning(f"No oxy columns found in concentration data")
                logger.warning(f"Available columns: {data.columns.tolist()}")
                return

            n = len(hbo_cols)
            logger.info(f"Plotting {n} channels")
            fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
            axes = np.atleast_1d(axes)

            for i, (ax, hbo_col) in enumerate(zip(axes, hbo_cols)):
                # Extract channel number
                ch_match = re.search(r'CH(\d+)', hbo_col)
                ch_num = ch_match.group(1) if ch_match else "?"

                # Try to find corresponding deoxy column
                hhb_col = None
                if 'HbO' in hbo_col:
                    hhb_col = hbo_col.replace('HbO', 'HHb')
                elif '_oxy' in hbo_col:
                    hhb_col = hbo_col.replace('_oxy', '_deoxy')
                elif ' oxy' in hbo_col:
                    hhb_col = hbo_col.replace(' oxy', ' deoxy')

                # Plot HbO
                if hbo_col in data.columns:
                    ax.plot(time, pd.to_numeric(data[hbo_col], errors='coerce'),
                            color=self.styles['raw']['HbO'], linewidth=1, label='HbO')

                # Plot HHb if exists
                if hhb_col and hhb_col in data.columns:
                    ax.plot(time, pd.to_numeric(data[hhb_col], errors='coerce'),
                            color=self.styles['raw']['HHb'], linewidth=1, label='HHb')

                ax.set_title(f'CH{ch_num}', loc='left', fontsize=9, fontweight='bold')
                ax.set_ylabel(self.y_label)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.2)

        else:  # OD data
            # Get OD columns (exclude Sample number and Event)
            od_cols = [col for col in data.columns
                       if col not in ["Sample number", "Event"] and 'WL' in col]

            if not od_cols:
                logger.warning(f"No OD columns found in data")
                return

            # Get wavelength pairs for each channel
            wl_pairs = self._get_wavelength_pairs(od_cols)

            if not wl_pairs:
                # If no pairs found, plot all columns individually
                wl_pairs = [(col, None) for col in od_cols]

            # Sort wl_pairs by channel number if possible
            if channel_map:
                # Create a mapping from column to channel number
                col_to_channel = {}
                for ch_num, ch_info in channel_map.items():
                    if 'columns' in ch_info:
                        for wl, cols in ch_info['columns'].items():
                            for col in cols:
                                col_to_channel[col] = ch_num

                # Sort wl_pairs by channel number
                def get_channel_num(pair):
                    col1, _ = pair
                    return col_to_channel.get(col1, 999)  # Put unknown at end

                wl_pairs.sort(key=get_channel_num)
            else:
                # Try to sort by column name
                wl_pairs.sort(key=lambda x: x[0])

            n = len(wl_pairs)
            logger.info(f"Plotting {n} OD channels")
            fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
            axes = np.atleast_1d(axes)

            for ax, (wl1_col, wl2_col) in zip(axes, wl_pairs):
                # Extract channel info from column name
                ch_num = self._extract_channel_number_od(wl1_col)

                # Determine channel type
                is_short = False
                channel_type = None
                if ch_num and channel_map:
                    is_short = self._is_short_channel(ch_num, channel_map)
                    channel_type = self._get_channel_type(ch_num, channel_map)
                elif channel_map and wl1_col in channel_map:
                    # Try to get channel info directly
                    for ch_idx, info in channel_map.items():
                        if 'columns' in info and wl1_col in info['columns']:
                            ch_num = ch_idx
                            is_short = info.get('short_channel', False)
                            break

                # Get color
                color = self._get_color_for_channel(is_short, channel_type)

                # Plot first wavelength
                label = wl1_col
                ax.plot(time, pd.to_numeric(data[wl1_col], errors='coerce'),
                        color=color, linewidth=1, label=label)

                # Plot second wavelength if exists
                if wl2_col and wl2_col in data.columns:
                    label2 = wl2_col
                    ax.plot(time, pd.to_numeric(data[wl2_col], errors='coerce'),
                            color=color, linewidth=0.7, alpha=0.8, label=label2)

                # Create title
                title = f"CH{ch_num}" if ch_num else wl1_col
                if ch_num:
                    if is_short:
                        title += " [SHORT]"
                    if channel_type:
                        title += f" {channel_type}"

                ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
                ax.set_ylabel(self.y_label)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('Raw All Channels', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved plot to {output_path}")

    def plot_raw_regions(
            self,
            regional_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None,
            channel_map: Optional[Dict] = None
    ) -> None:
        """Plot raw regional averages for BRAIN REGIONS ONLY."""
        logger.info(f"=== plot_raw_regions called ===")

        if self.data_type == 'concentration':
            logger.info("Processing concentration data for region plots")

            # Handle both dict and DataFrame input
            if isinstance(regional_data, pd.DataFrame):
                logger.info(f"DataFrame input with shape: {regional_data.shape}")
                logger.info(f"Columns: {regional_data.columns.tolist()}")

                # Try to find region columns
                region_pairs = self._find_region_column_pairs(regional_data)

                if not region_pairs:
                    logger.warning("No region columns found in DataFrame!")
                    # Try to create regions from individual channels
                    region_pairs = self._create_regions_from_channels(regional_data, channel_map)

                if not region_pairs:
                    logger.error("Cannot create region plots - no region data available")
                    return

                logger.info(f"Found {len(region_pairs)} region pairs: {list(region_pairs.keys())}")

                # Plot regions
                self._plot_region_pairs(region_pairs, regional_data, output_path, is_processed=False)

            else:
                # Dict format
                logger.info(f"Dict input with {len(regional_data)} regions")
                valid_regions = [
                    r for r in regional_data
                    if f'{r}_oxy' in regional_data[r].columns
                       and f'{r}_deoxy' in regional_data[r].columns
                ]

                if not valid_regions:
                    logger.warning(f"No valid regions found")
                    return

                fig, axes = plt.subplots(len(valid_regions), 1,
                                         figsize=(12, 3 * len(valid_regions)),
                                         sharex=True)
                axes = np.atleast_1d(axes)

                for ax, region in zip(axes, valid_regions):
                    df = regional_data[region]
                    t = self._get_time(len(df))

                    ax.plot(t, df[f'{region}_oxy'],
                            color=self.styles['raw']['HbO'], label=f'{region} HbO', linewidth=1.5)
                    ax.plot(t, df[f'{region}_deoxy'],
                            color=self.styles['raw']['HHb'], label=f'{region} HHb', linewidth=1.5)

                    # Get CH numbers for this region
                    ch_list = CH_REGION_MAP.get(region, [])
                    ch_numbers = sorted(set(ch_list))
                    ax.set_title(f"{region}  (CH{', CH'.join(map(str, ch_numbers))})", loc='left', fontsize=10)
                    ax.set_ylabel(self.y_label)
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)

                axes[-1].set_xlabel('Time (s)')
                plt.suptitle('Raw Regional Averages', fontsize=12, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close()

        else:  # OD data
            if not isinstance(regional_data, pd.DataFrame):
                raise ValueError("For OD data, regional_data must be a DataFrame")

            if channel_map is None:
                raise ValueError("channel_map required for OD data")

            # Build region map
            region_map = build_region_map_od(regional_data.columns.tolist(), channel_map)
            valid_regions = list(region_map.keys())

            if not valid_regions:
                logger.warning(f"No valid regions found for OD data")
                return

            fig, axes = plt.subplots(len(valid_regions), 1,
                                     figsize=(12, 3 * len(valid_regions)),
                                     sharex=True)
            axes = np.atleast_1d(axes)

            t = self._get_time(len(regional_data))

            for ax, region in zip(axes, valid_regions):
                region_cols = region_map[region]

                # Group columns by wavelength
                wl_groups = {}
                for col in region_cols:
                    if col in regional_data.columns:
                        m = re.search(r"WL(\d+)", col)
                        if m:
                            wl = m.group(1)
                            if wl not in wl_groups:
                                wl_groups[wl] = []
                            wl_groups[wl].append(col)

                # Plot average for each wavelength
                colors = [self.styles['raw']['wl1'], self.styles['raw']['wl2']]
                for i, (wl, cols) in enumerate(sorted(wl_groups.items())):
                    if i >= len(colors):
                        break
                    avg_data = regional_data[cols].mean(axis=1)
                    ax.plot(t, avg_data, color=colors[i], label=f'{region} WL{wl}', linewidth=1.5)

                # Show CH numbers in title
                ch_numbers = sorted(set(CH_REGION_MAP.get(region, [])))
                if not ch_numbers and '_' in region:
                    base_region = region.split('_')[0]
                    ch_numbers = sorted(set(CH_REGION_MAP_COMBINED.get(base_region, [])))

                if ch_numbers:
                    ax.set_title(f"{region}  (CH{', CH'.join(map(str, ch_numbers))})", loc='left', fontsize=10)
                else:
                    ax.set_title(region, loc='left', fontsize=10)

                ax.set_ylabel(self.y_label)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel('Time (s)')
            plt.suptitle('Raw OD Regional Averages', fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

    def plot_processed_regions(
            self,
            data: pd.DataFrame,
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """Plot processed time-courses for each region."""
        logger.info(f"=== plot_processed_regions called ===")
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()}")

        if self.data_type == 'concentration':
            logger.info("Processing concentration data for processed region plots")

            # Try to find region columns
            region_pairs = self._find_region_column_pairs(data)

            if not region_pairs:
                logger.warning("No region columns found in processed data!")
                logger.warning("Looking for region columns that don't start with 'CH'...")

                # Try alternative pattern matching
                for col in data.columns:
                    if not col.startswith('CH') and col not in ["Sample number", "Event", "grand_oxy", "grand_deoxy"]:
                        logger.info(f"Potential region column: {col}")

            if not region_pairs:
                logger.error("Cannot create processed region plots - no region data available")
                return

            logger.info(f"Found {len(region_pairs)} region pairs for processed plot: {list(region_pairs.keys())}")

            # Plot regions
            self._plot_region_pairs(region_pairs, data, output_path, is_processed=True)

        else:  # OD data
            # For OD processed data (if applicable)
            region_pattern = r"([A-Za-z_]+)_wl\d+"
            regions = set()
            for col in data.columns:
                m = re.match(region_pattern, col)
                if m:
                    regions.add(m.group(1))

            regions = sorted(regions)

            if not regions:
                logger.warning(f"No processed OD region columns found")
                return

            t = self._get_time(len(data))
            fig, axs = plt.subplots(len(regions), 1,
                                    figsize=(12, 3 * len(regions)),
                                    sharex=True)
            axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]

            for ax, region in zip(axs, regions):
                # Find wavelength columns for this region
                wl_cols = [c for c in data.columns if c.startswith(f'{region}_')]
                wl_data = {}
                for col in wl_cols:
                    m = re.search(r"_wl(\d+)", col)
                    if m:
                        wl_data[int(m.group(1))] = col

                # Plot sorted by wavelength
                colors = [self.styles['processed']['wl1'], self.styles['processed']['wl2']]
                for i, (wl, col) in enumerate(sorted(wl_data.items())):
                    if i >= len(colors):
                        break
                    linestyle = '-' if i == 0 else self.styles['processed'].get('wl2_linestyle', '--')
                    ax.plot(t, data[col],
                            label=f'{region} WL{wl}',
                            color=colors[i],
                            linestyle=linestyle,
                            linewidth=1.5)

                ax.set_title(region, loc='left', fontsize=10, fontweight='bold')
                ax.set_ylabel(self.y_label)
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)

            axs[-1].set_xlabel('Time (s)')
            plt.suptitle('Processed OD Regional Averages', fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

    # =========================================================================
    # NEW HELPER METHODS FOR REGION DETECTION
    # =========================================================================

    def _find_region_column_pairs(self, data: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
        """Find pairs of oxy/deoxy columns for each region."""
        region_pairs = {}

        # Look for columns ending with _oxy and _deoxy
        oxy_cols = [col for col in data.columns if col.endswith('_oxy')]
        deoxy_cols = [col for col in data.columns if col.endswith('_deoxy')]

        logger.info(f"Found {len(oxy_cols)} oxy columns, {len(deoxy_cols)} deoxy columns")

        # Match oxy and deoxy columns by region name
        for oxy_col in oxy_cols:
            region = oxy_col.replace('_oxy', '')

            # Skip if this is a channel (starts with CH)
            if region.startswith('CH'):
                continue

            # Look for corresponding deoxy column
            deoxy_col = f"{region}_deoxy"
            if deoxy_col in deoxy_cols:
                region_pairs[region] = (oxy_col, deoxy_col)
                logger.info(f"Found region pair: {region} -> {oxy_col}, {deoxy_col}")

        # If no matches, try alternative naming (HbO/HHb)
        if not region_pairs:
            hbo_cols = [col for col in data.columns if ' HbO' in col]
            hhb_cols = [col for col in data.columns if ' HHb' in col]

            for hbo_col in hbo_cols:
                region = hbo_col.replace(' HbO', '')

                # Skip if this is a channel
                if region.startswith('CH'):
                    continue

                hhb_col = f"{region} HHb"
                if hhb_col in hhb_cols:
                    region_pairs[region] = (hbo_col, hhb_col)
                    logger.info(f"Found region pair (alt naming): {region} -> {hbo_col}, {hhb_col}")

        return region_pairs

    def _create_regions_from_channels(self, data: pd.DataFrame, channel_map: Optional[Dict]) -> Dict[
        str, Tuple[List[str], List[str]]]:
        """Create region data by averaging channels in each region."""
        region_pairs = {}

        if not channel_map:
            logger.warning("Cannot create regions from channels: no channel_map provided")
            return region_pairs

        # Get channel to region mapping
        from read.channel_utils import CHANNEL_TO_REGION

        # Group channels by region
        region_channels = {}
        for col in data.columns:
            if col.startswith('CH'):
                # Extract channel number
                ch_match = re.search(r'CH(\d+)', col)
                if ch_match:
                    ch_num = int(ch_match.group(1))
                    region = CHANNEL_TO_REGION.get(ch_num)

                    if region:
                        if region not in region_channels:
                            region_channels[region] = {'oxy': [], 'deoxy': []}

                        if 'HbO' in col or '_oxy' in col or ' oxy' in col:
                            region_channels[region]['oxy'].append(col)
                        elif 'HHb' in col or '_deoxy' in col or ' deoxy' in col:
                            region_channels[region]['deoxy'].append(col)

        # Create region pairs
        for region, channels in region_channels.items():
            if channels['oxy'] and channels['deoxy']:
                region_pairs[region] = (channels['oxy'], channels['deoxy'])
                logger.info(
                    f"Created region {region} from {len(channels['oxy'])} oxy and {len(channels['deoxy'])} deoxy channels")

        return region_pairs

    def _plot_region_pairs(self, region_pairs: Dict, data: pd.DataFrame, output_path: str, is_processed: bool = False):
        """Plot region data from region pairs dictionary."""
        if not region_pairs:
            logger.error("No region pairs to plot")
            return

        regions = sorted(region_pairs.keys())
        logger.info(f"Plotting {len(regions)} regions: {regions}")

        t = self._get_time(len(data))
        fig, axes = plt.subplots(len(regions), 1,
                                 figsize=(12, 3 * len(regions)),
                                 sharex=True)

        if len(regions) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, region in zip(axes, regions):
            oxy_info, deoxy_info = region_pairs[region]

            # Handle different types of oxy/deoxy info
            if isinstance(oxy_info, str) and isinstance(deoxy_info, str):
                # Single column each
                if oxy_info in data.columns:
                    ax.plot(t, data[oxy_info],
                            color=self.styles['processed' if is_processed else 'raw']['HbO'],
                            label=f'{region} HbO',
                            linewidth=1.5)

                if deoxy_info in data.columns:
                    ax.plot(t, data[deoxy_info],
                            color=self.styles['processed' if is_processed else 'raw']['HHb'],
                            label=f'{region} HHb',
                            linestyle=self.styles['processed']['HHb_linestyle'] if is_processed else '-',
                            linewidth=1.5)

            elif isinstance(oxy_info, list) and isinstance(deoxy_info, list):
                # Multiple columns - average them
                oxy_cols = [col for col in oxy_info if col in data.columns]
                deoxy_cols = [col for col in deoxy_info if col in data.columns]

                if oxy_cols:
                    oxy_avg = data[oxy_cols].mean(axis=1)
                    ax.plot(t, oxy_avg,
                            color=self.styles['processed' if is_processed else 'raw']['HbO'],
                            label=f'{region} HbO (avg)',
                            linewidth=1.5)

                if deoxy_cols:
                    deoxy_avg = data[deoxy_cols].mean(axis=1)
                    ax.plot(t, deoxy_avg,
                            color=self.styles['processed' if is_processed else 'raw']['HHb'],
                            label=f'{region} HHb (avg)',
                            linestyle=self.styles['processed']['HHb_linestyle'] if is_processed else '-',
                            linewidth=1.5)

            # Get CH numbers for this region
            ch_list = CH_REGION_MAP.get(region, [])
            if not ch_list and '_' in region:
                base_region = region.split('_')[0]
                ch_list = CH_REGION_MAP_COMBINED.get(base_region, [])

            title = region
            if ch_list:
                ch_numbers = sorted(set(ch_list))
                title = f"{region}  (CH{', CH'.join(map(str, ch_numbers))})"

            ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
            ax.set_ylabel(self.y_label)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')

        plot_type = "Processed" if is_processed else "Raw"
        plt.suptitle(f'{plot_type} Regional Averages', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Saved region plot to {output_path}")

    # =========================================================================
    # EXISTING HELPER METHODS (keep these from previous version)
    # =========================================================================

    def plot_short_channels(
            self,
            data: pd.DataFrame,
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None,
            channel_map: Optional[Dict] = None
    ) -> None:
        """Plot all short channels separately for visual inspection."""
        if self.data_type == 'concentration':
            # For concentration data
            short_cols = []
            for col in data.columns:
                if col not in ["Sample number", "Event"]:
                    ch_num = self._extract_channel_number_conc(col)
                    if ch_num and channel_map and self._is_short_channel(ch_num, channel_map):
                        short_cols.append(col)
        else:
            # For OD data
            short_cols = []
            for col in data.columns:
                if col not in ["Sample number", "Event"] and 'WL' in col:
                    ch_num = self._extract_channel_number_od(col)
                    if ch_num and channel_map and self._is_short_channel(ch_num, channel_map):
                        short_cols.append(col)

        if not short_cols:
            logger.warning(f"No short channels found for {output_path}")
            return

        time = self._get_time(len(data))
        n = len(short_cols)

        fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
        axes = np.atleast_1d(axes) if n > 1 else [axes]

        for ax, col in zip(axes, short_cols):
            ch_num = self._extract_channel_number_conc(
                col) if self.data_type == 'concentration' else self._extract_channel_number_od(col)
            channel_type = self._get_channel_type(ch_num, channel_map) if ch_num and channel_map else None

            # Get color
            color = self._get_color_for_channel(True, channel_type)

            ax.plot(time, pd.to_numeric(data[col], errors='coerce'),
                    color=color, linewidth=1, label=col)

            title = f"CH{ch_num} - {col}" if ch_num else col
            if channel_type:
                title += f" ({channel_type})"

            ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
            ax.set_ylabel(self.y_label)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel('Time (s)')
        plt.suptitle("Short Channel Signals", fontsize=12, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_raw_od(
            self,
            data: pd.DataFrame,
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None,
            channel_map: Optional[Dict] = None
    ) -> None:
        """Plot raw optical density data."""
        time = self._get_time(len(data))

        # Get OD columns
        od_cols = [col for col in data.columns
                   if col not in ["Sample number", "Event"] and 'WL' in col]

        if not od_cols:
            logger.warning(f"No OD columns found for {output_path}")
            return

        n = len(od_cols)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)

        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, col in zip(axes, od_cols):
            ch_num = self._extract_channel_number_od(col)

            # Get color based on channel type
            color = self.styles['raw']['wl1']
            if ch_num and channel_map:
                is_short = self._is_short_channel(ch_num, channel_map)
                channel_type = self._get_channel_type(ch_num, channel_map)
                color = self._get_color_for_channel(is_short, channel_type)

                title = f"CH{ch_num} - {col}"
                if is_short:
                    title += " [SHORT]"
                if channel_type:
                    title += f" ({channel_type})"
            else:
                title = col

            ax.plot(time, pd.to_numeric(data[col], errors='coerce'),
                    color=color, linewidth=1, label=col)

            ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
            ax.set_ylabel(self.y_label)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_events(self, ax, events: pd.DataFrame):
        """Add vertical lines for event markers."""
        for _, row in events.iterrows():
            sample = row['Sample number']
            time = sample / self.fs
            event_name = row.get('Event', '')

            ax.axvline(x=time, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(time, ax.get_ylim()[1], f' {event_name}',
                    rotation=90, verticalalignment='top', fontsize=8, alpha=0.7)

    def plot_raw_overall(
            self,
            data: pd.DataFrame,
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None,
            events: Optional[pd.DataFrame] = None
    ) -> None:
        """Plot grand mean of raw data with optional event markers."""
        t = self._get_time(len(data))
        fig, ax = plt.subplots(figsize=(12, 4))

        if self.data_type == 'concentration':
            # Look for grand mean columns
            grand_oxy = None
            grand_deoxy = None

            for col in data.columns:
                if 'grand_oxy' in col or 'grand HbO' in col:
                    grand_oxy = col
                elif 'grand_deoxy' in col or 'grand HHb' in col:
                    grand_deoxy = col

            if grand_oxy and grand_oxy in data.columns:
                ax.plot(t, data[grand_oxy], label='Raw Mean HbO',
                        color=self.styles['raw']['HbO'], linewidth=1.5)
            if grand_deoxy and grand_deoxy in data.columns:
                ax.plot(t, data[grand_deoxy], label='Raw Mean HHb',
                        color=self.styles['raw']['HHb'], linewidth=1.5)
        else:  # OD data
            # Look for grand mean OD columns
            wl_cols = [c for c in data.columns if 'grand_' in c and 'WL' in c]
            if not wl_cols:
                wl_cols = [c for c in data.columns if c.startswith('grand_')]

            if wl_cols:
                ax.plot(t, data[wl_cols[0]], label='Raw Mean WL1',
                        color=self.styles['raw']['wl1'], linewidth=1.5)
            if len(wl_cols) >= 2:
                ax.plot(t, data[wl_cols[1]], label='Raw Mean WL2',
                        color=self.styles['raw']['wl2'], linewidth=1.5)

        if events is not None and not events.empty and 'Sample number' in events.columns:
            self._plot_events(ax, events)

        ax.set_title('Raw Overall Mean')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(self.y_label)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_processed_overall(
            self,
            data: pd.DataFrame,
            output_path: str,
            y_limits: Optional[Tuple[float, float]] = None,
            events: Optional[pd.DataFrame] = None
    ) -> None:
        """Plot grand-average of processed data with optional event markers."""
        t = self._get_time(len(data))
        fig, ax = plt.subplots(figsize=(12, 4))

        if self.data_type == 'concentration':
            # Look for grand mean columns
            grand_oxy = None
            grand_deoxy = None

            for col in data.columns:
                if 'grand_oxy' in col or 'grand HbO' in col:
                    grand_oxy = col
                elif 'grand_deoxy' in col or 'grand HHb' in col:
                    grand_deoxy = col

            if grand_oxy and grand_oxy in data.columns:
                ax.plot(t, data[grand_oxy],
                        label='Processed Mean HbO',
                        color=self.styles['processed']['HbO'],
                        linewidth=1.5)
            if grand_deoxy and grand_deoxy in data.columns:
                ax.plot(t, data[grand_deoxy],
                        label='Processed Mean HHb',
                        color=self.styles['processed']['HHb'],
                        linestyle=self.styles['processed']['HHb_linestyle'],
                        linewidth=1.5)
        else:  # OD data
            wl_cols = [c for c in data.columns if 'grand_' in c]
            if not wl_cols:
                wl_cols = [c for c in data.columns if c.startswith('grand_')]

            if wl_cols:
                ax.plot(t, data[wl_cols[0]],
                        label='Processed Mean WL1',
                        color=self.styles['processed']['wl1'],
                        linewidth=1.5)
            if len(wl_cols) >= 2:
                ax.plot(t, data[wl_cols[1]],
                        label='Processed Mean WL2',
                        color=self.styles['processed']['wl2'],
                        linestyle=self.styles['processed'].get('wl2_linestyle', '-'),
                        linewidth=1.5)

        if events is not None and not events.empty and 'Sample number' in events.columns:
            self._plot_events(ax, events)

        ax.set_title('Processed Overall Mean')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(self.y_label)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    # -------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------

    def _extract_channel_number_conc(self, column_name: str) -> Optional[int]:
        """Extract channel number from concentration column name."""
        patterns = [
            r'CH(\d+)(?:\s+|_)(?:HbO|HHb|oxy|deoxy)',
            r'CH(\d+)\s+(?:HbO|HHb)',
            r'CH(\d+)_(?:oxy|deoxy)',
        ]

        for pattern in patterns:
            m = re.search(pattern, column_name, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_channel_number_od(self, column_name: str) -> Optional[int]:
        """Extract channel number from OD column name."""
        if 'WL' not in column_name or column_name in ["Sample number", "Event"]:
            return None

        m = re.match(r"D(\d+)_R(\d+)_S(\d+)_WL\d+", column_name)
        if not m:
            return None

        return None

    def _extract_channel_number(self, column_name: str) -> Optional[int]:
        """Generic channel number extractor."""
        if self.data_type == 'concentration':
            return self._extract_channel_number_conc(column_name)
        else:
            return self._extract_channel_number_od(column_name)

    def _is_short_channel(self, ch_num: int, channel_map: Optional[Dict]) -> bool:
        """Check if a channel is a short channel."""
        if channel_map and ch_num in channel_map:
            return channel_map[ch_num].get('short_channel', False)

        from read.channel_utils import SHORT_CHANNEL_LIST
        return ch_num in SHORT_CHANNEL_LIST

    def _get_channel_type(self, ch_num: int, channel_map: Optional[Dict]) -> Optional[str]:
        """Get channel type (PFC_SHORT, MOTOR_LONG, etc.)."""
        if not ch_num:
            return None

        from read.channel_utils import SHORT_CHANNELS_BY_REGION
        for region_type, channels in SHORT_CHANNELS_BY_REGION.items():
            if ch_num in channels:
                return region_type

        for region, channels in CH_REGION_MAP.items():
            if ch_num in channels:
                if 'PFC' in region:
                    return 'PFC_LONG'
                elif any(x in region for x in ['PAR', 'PMC', 'M1', 'SMA']):
                    return 'MOTOR_LONG'

        for region, channels in CH_REGION_MAP_COMBINED.items():
            if ch_num in channels:
                if 'PFC' in region:
                    return 'PFC_LONG'
                elif any(x in region for x in ['M1', 'SMA', 'S1', 'V1']):
                    return 'MOTOR_LONG'

        return None

    def _get_color_for_channel(self, is_short: bool, channel_type: Optional[str]) -> str:
        """Get color based on channel type."""
        if is_short:
            if channel_type and 'PFC' in channel_type:
                return CHANNEL_TYPE_COLORS['PFC_SHORT']
            elif channel_type and 'MOTOR' in channel_type:
                return CHANNEL_TYPE_COLORS['MOTOR_SHORT']
            else:
                return CHANNEL_TYPE_COLORS['SHORT']
        else:
            if channel_type and 'PFC' in channel_type:
                return CHANNEL_TYPE_COLORS['PFC_LONG']
            elif channel_type and 'MOTOR' in channel_type:
                return CHANNEL_TYPE_COLORS['MOTOR_LONG']
            else:
                return CHANNEL_TYPE_COLORS['LONG']