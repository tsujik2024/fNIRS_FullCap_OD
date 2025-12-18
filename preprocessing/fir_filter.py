import pandas as pd
import numpy as np
from scipy.signal import firwin, filtfilt


def fir_filter(df: pd.DataFrame, order: int, Wn: list, fs: int):
    """
    FIR filter for fNIRS data that:
    - Automatically skips 'Event' and 'Sample number' columns
    - Only processes numeric data
    - Silently handles problematic values
    - Returns filtered data in original structure
    """
    # 1. Identify columns to filter (all numeric except excluded)
    exclude = {'Event', 'Sample number'}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    filter_cols = [col for col in numeric_cols if col not in exclude]

    # 2. Early return if nothing to filter
    if not filter_cols:
        return df.copy()

    # 3. Design filter once
    b = firwin(order + 1, Wn, pass_zero=False, fs=fs)

    # 4. Process columns in bulk (faster than looping)
    filtered = df.copy()
    for col in filter_cols:
        # Convert to numpy array safely
        data = pd.to_numeric(df[col], errors='coerce').to_numpy()
        # Replace NaNs with zeros (silently)
        data = np.nan_to_num(data)
        # Apply filter
        filtered[col] = filtfilt(b, [1.0], data)

    return filtered