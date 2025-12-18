"""
tddr.py

Implements the Temporal Derivative Distribution Repair (TDDR) algorithm
for motion artifact correction in fNIRS data.

Reference implementation from:
Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
Temporal Derivative Distribution Repair (TDDR): A motion correction
method for fNIRS. NeuroImage, 184, 171-179.
https://doi.org/10.1016/j.neuroimage.2018.09.025
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def tddr(data: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
    """
    Apply Temporal Derivative Distribution Repair (TDDR) to correct motion artifacts
    in fNIRS data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing fNIRS data. Each float-type column is treated as a
        channel (e.g., 'CH1 HbO', 'CH1 HbR'), while non-float columns (e.g.,
        'Sample number', 'Event') are skipped.
    sample_rate : float
        Sampling rate of the data in Hz.

    Returns
    -------
    corrected_df : pd.DataFrame
        DataFrame with TDDR-corrected data for each float-type channel.
    """
    corrected_df = data.copy()

    # Apply TDDR to each float column (e.g., O2Hb and HHb channels)
    for col in corrected_df.columns:
        # Only process float64 columns
        if corrected_df[col].dtype == np.float64:
            corrected_df[col] = TDDR(
                np.array(corrected_df[col], dtype='float64'),
                sample_rate
            )

    return corrected_df


def TDDR(signal, sample_rate):
    """
    Reference implementation for the TDDR algorithm.

    This function is the reference implementation for the TDDR algorithm for
    motion correction of fNIRS data, as described in:

    Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
    Temporal Derivative Distribution Repair (TDDR): A motion correction
    method for fNIRS. NeuroImage, 184, 171-179.
    https://doi.org/10.1016/j.neuroimage.2018.09.025

    Usage:
        signals_corrected = TDDR(signals, sample_rate)

    Inputs:
        signal: A [sample x channel] matrix of uncorrected optical density data
        sample_rate: A scalar reflecting the rate of acquisition in Hz

    Outputs:
        signal_corrected: A [sample x channel] matrix of corrected optical density data
    """
    signal = np.array(signal)

    if len(signal.shape) != 1:
        # Multi-channel: recursively process each channel
        for ch in range(signal.shape[1]):
            signal[:, ch] = TDDR(signal[:, ch], sample_rate)
        return signal

    # Check for NaN or Inf in input
    if not np.all(np.isfinite(signal)):
        # Return original signal if it contains invalid values
        return signal

    # Preprocess: Separate high and low frequencies
    filter_cutoff = 0.5
    filter_order = 3
    Fc = filter_cutoff * 2 / sample_rate
    signal_mean = np.mean(signal)
    signal = signal - signal_mean

    if Fc < 1:
        fb, fa = butter(filter_order, Fc)
        signal_low = filtfilt(fb, fa, signal, padlen=0)
    else:
        signal_low = signal

    signal_high = signal - signal_low

    # Initialize
    tune = 4.685
    D = np.sqrt(np.finfo(signal.dtype).eps)
    mu = np.inf
    iter = 0

    # Step 1. Compute temporal derivative of the signal
    deriv = np.diff(signal_low)

    # Step 2. Initialize observation weights
    w = np.ones(deriv.shape)

    # Step 3. Iterative estimation of robust weights
    while iter < 50:
        iter = iter + 1
        mu0 = mu

        # Step 3a. Estimate weighted mean
        mu = np.sum(w * deriv) / np.sum(w)

        # Step 3b. Calculate absolute residuals of estimate
        dev = np.abs(deriv - mu)

        # Step 3c. Robust estimate of standard deviation of the residuals
        sigma = 1.4826 * np.median(dev)

        # Safety check: if sigma is zero, signal has no variation
        if sigma < 1e-10:
            # No motion artifacts to correct, return original
            return signal + signal_mean

        # Step 3d. Scale deviations by standard deviation and tuning parameter
        r = dev / (sigma * tune)

        # Step 3e. Calculate new weights according to Tukey's biweight function
        w = ((1 - r ** 2) * (r < 1)) ** 2

        # Step 3f. Terminate if new estimate is within machine-precision of old estimate
        if abs(mu - mu0) < D * max(abs(mu), abs(mu0)):
            break

    # Step 4. Apply robust weights to centered derivative
    new_deriv = w * (deriv - mu)

    # Step 5. Integrate corrected derivative
    signal_low_corrected = np.cumsum(np.insert(new_deriv, 0, 0.0))

    # Postprocess: Center the corrected signal
    signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)

    # Postprocess: Merge back with uncorrected high frequency component
    signal_corrected = signal_low_corrected + signal_high + signal_mean

    # Final safety check
    if not np.all(np.isfinite(signal_corrected)):
        # If correction failed, return original signal
        return signal + signal_mean

    return signal_corrected