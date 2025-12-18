"""
Implements baseline subtraction for fNIRS data, supporting:
- A custom baseline DataFrame (user-provided), or
- The baseline period marked by specific events, or
- A specified sample range.

Flexible handling of baseline markers (S1, S2, W1, etc.)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def baseline_subtraction(
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        baseline_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Applies baseline subtraction to the given DataFrame of fNIRS signals.

    Flexible baseline detection:
    - Looks for explicit BaselineStart/BaselineEnd markers
    - Uses period BETWEEN first two task markers (e.g., S1 to W1, or S1 to S2)
    - Falls back to first 20 seconds if no markers found

    Typical usage:
    - S1 = baseline/recording start
    - W1/S2 = task start (end of baseline)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fNIRS data (columns for channels), plus any metadata
        columns like 'Sample number', 'Event', 'Time (s)' that should be ignored.
    events_df : pd.DataFrame
        DataFrame specifying events. Must have columns:
          - 'Sample number'
          - 'Event'
    baseline_df : pd.DataFrame, optional
        If provided, each channel's baseline mean is computed from this DataFrame
        instead of from events in `df`. Must have the same column names as `df`.
        Default is None.

    Returns
    -------
    corrected_df : pd.DataFrame
        A new DataFrame with the baseline-subtracted signals.
    """
    corrected_df = df.copy()

    # Identify which columns are channels vs. metadata
    ignore_cols = ['Sample number', 'Event', 'Time (s)', 'Condition', 'Subject']
    data_cols = [col for col in corrected_df.columns if col not in ignore_cols]

    if baseline_df is not None:
        # Use the provided baseline_df
        logger.info("Using provided baseline DataFrame for baseline subtraction")
        for ch in data_cols:
            baseline_mean = baseline_df[ch].mean()
            corrected_df[ch] = corrected_df[ch] - baseline_mean
        return corrected_df

    # ---------------------------------------------------
    # Compute baseline from events_df markers
    # ---------------------------------------------------
    logger.info("Computing baseline from events DataFrame")

    # Clean up events DataFrame
    events_clean = events_df.copy()
    events_clean['Event'] = (
        events_clean['Event']
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)  # Remove internal spaces
    )

    # Filter out empty/nan events
    events_clean = events_clean[
        events_clean['Event'].str.contains(r'[A-Z0-9]', regex=True, na=False)
    ]

    # Sort by sample number
    events_clean = events_clean.sort_values('Sample number').reset_index(drop=True)

    # Log what events we found
    logger.info(f"Found {len(events_clean)} events: {events_clean['Event'].tolist()}")

    # Estimate sampling rate
    if hasattr(df, 'fs'):
        fs = df.fs
    elif 'Time (s)' in df.columns and len(df['Time (s)']) > 1:
        time_diff = df['Time (s)'].iloc[1] - df['Time (s)'].iloc[0]
        fs = 1 / time_diff if time_diff > 0 else 50.0
    else:
        fs = 50.0

    logger.info(f"Using sampling rate: {fs} Hz")

    # Strategy 1: Look for explicit BaselineStart/BaselineEnd
    if ('BASELINESTART' in events_clean['Event'].values and
            'BASELINEEND' in events_clean['Event'].values):
        start_sample = events_clean.loc[
            events_clean['Event'] == 'BASELINESTART', 'Sample number'
        ].values[0]
        end_sample = events_clean.loc[
            events_clean['Event'] == 'BASELINEEND', 'Sample number'
        ].values[0]
        logger.info(f"Found explicit baseline markers: samples {start_sample} to {end_sample}")

    # Strategy 2: Use period between FIRST TWO markers
    # (e.g., S1 to W1, S1 to S2, S1 to S3, etc.)
    elif len(events_clean) >= 2:
        # Get first two markers
        first_marker = events_clean.iloc[0]
        second_marker = events_clean.iloc[1]

        start_sample = first_marker['Sample number']
        end_sample = second_marker['Sample number']

        baseline_duration = (end_sample - start_sample) / fs

        logger.info(
            f"Using period between first two markers as baseline:\n"
            f"  Start: '{first_marker['Event']}' at sample {start_sample}\n"
            f"  End: '{second_marker['Event']}' at sample {end_sample}\n"
            f"  Duration: {baseline_duration:.1f}s"
        )

        # Validate baseline duration is reasonable (between 2s and 60s)
        if baseline_duration < 2.0:
            logger.warning(
                f"Baseline period very short ({baseline_duration:.1f}s). "
                f"Check if markers are correct."
            )
        elif baseline_duration > 60.0:
            logger.warning(
                f"Baseline period very long ({baseline_duration:.1f}s). "
                f"Check if markers are correct. Capping at 30s."
            )
            # Cap at 30 seconds
            end_sample = start_sample + int(30 * fs)

    # Strategy 3: Only ONE marker found - use period from start to that marker
    elif len(events_clean) == 1:
        first_marker = events_clean.iloc[0]
        start_sample = 4  # Skip initial samples
        end_sample = first_marker['Sample number']

        baseline_duration = (end_sample - start_sample) / fs
        logger.info(
            f"Only one marker found ('{first_marker['Event']}'). "
            f"Using {baseline_duration:.1f}s from recording start to marker as baseline."
        )

        # Ensure reasonable duration
        if baseline_duration < 5.0:
            logger.warning(
                f"Baseline too short ({baseline_duration:.1f}s). Using first 20s instead."
            )
            start_sample = 4
            end_sample = int(20 * fs)

    # Strategy 4: No markers found - use first 20 seconds
    else:
        logger.warning("No event markers found. Using first 20 seconds as baseline.")
        start_sample = 4
        end_sample = int(20 * fs)

    # Convert to integers and validate bounds
    start = int(start_sample)
    end = int(end_sample)

    # Ensure indices are within data bounds
    if not (0 <= start < len(corrected_df)):
        logger.warning(f"Start sample {start} out of bounds. Adjusting to 4.")
        start = 4

    if not (start < end <= len(corrected_df)):
        logger.warning(
            f"End sample {end} out of bounds (data length={len(corrected_df)}). "
            f"Adjusting to valid range."
        )
        end = min(end, len(corrected_df))

    if start >= end:
        logger.warning(
            f"Invalid baseline interval: start={start} >= end={end}. "
            "Using first 20 seconds instead."
        )
        start = 4
        end = min(int(20 * fs), len(corrected_df))

    # Apply baseline subtraction
    baseline_duration = (end - start) / fs
    logger.info(
        f"✓ Applying baseline correction: {baseline_duration:.1f}s "
        f"(samples {start} to {end})"
    )

    for ch in data_cols:
        baseline_segment = corrected_df.loc[start:end - 1, ch]
        baseline_mean = baseline_segment.mean()
        corrected_df[ch] = corrected_df[ch] - baseline_mean

    # Add baseline info as attributes
    corrected_df.attrs['baseline_start'] = start
    corrected_df.attrs['baseline_end'] = end
    corrected_df.attrs['baseline_duration_s'] = baseline_duration

    return corrected_df