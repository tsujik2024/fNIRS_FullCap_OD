"""
Convert full-cap fNIRS OD data to HbO/HbR concentration using MBLL.

Output format:
    CH0 HbO, CH0 HHb, CH1 HbO, CH1 HHb, ... CH25 HbO, CH25 HHb

UPDATED: Now uses Prahl extinction coefficients from:
    https://omlc.org/spectra/hemoglobin/summary.html
    (Scott Prahl, compiled from Gratzer and Kollias data)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1. Prahl Extinction Coefficients
#
#    Source: https://omlc.org/spectra/hemoglobin/summary.html
#    Original units: cm^-1 / M (molar, i.e., per mole/liter)
#
#    For MBLL we need: cm^-1 / mM (millimolar)
#    Conversion: divide by 1000 (since 1 M = 1000 mM)
#
#    The table provides HbO2 (oxygenated) and Hb (deoxygenated)
# ---------------------------------------------------------------------

# Raw Prahl coefficients in cm^-1 / M
_PRAHL_RAW = {
    # lambda (nm): (HbO2, Hb)
    750: (518.0, 1405.24),
    752: (533.2, 1515.32),
    754: (548.4, 1541.76),
    756: (562.0, 1560.48),
    757: (568.0, 1560.48),  # Interpolated between 756 and 758
    758: (574.0, 1560.48),
    759: (580.0, 1554.50),  # Interpolated between 758 and 760
    760: (586.0, 1548.52),
    762: (598.0, 1508.44),
    764: (610.0, 1459.56),
    836: (1001.2, 692.64),
    838: (1011.6, 692.48),
    839: (1016.8, 692.42),  # Interpolated between 838 and 840
    840: (1022.0, 692.36),
    842: (1032.4, 692.20),
    844: (1042.8, 691.96),
    846: (1050.0, 691.76),
    848: (1054.0, 691.52),
    850: (1058.0, 691.32),
    852: (1062.0, 691.08),
}

# Convert to cm^-1 / mM for MBLL
# Note: HbO = HbO2 (oxygenated hemoglobin)
#       HbR = Hb (deoxygenated hemoglobin, also called HHb)
EXT_COEFFS = {}
for wl, (hbo2, hb) in _PRAHL_RAW.items():
    EXT_COEFFS[wl] = {
        "HbO": hbo2 / 1000.0,  # Convert M^-1 to mM^-1
        "HbR": hb / 1000.0,
    }

# Log the coefficients being used
logger.info("Using Prahl extinction coefficients (cm^-1 / mM):")
for wl in sorted(EXT_COEFFS.keys()):
    logger.debug(f"  {wl} nm: HbO={EXT_COEFFS[wl]['HbO']:.4f}, HbR={EXT_COEFFS[wl]['HbR']:.4f}")


def get_extinction_coefficient(wavelength: int) -> dict:
    """
    Get extinction coefficients for a given wavelength.

    If the exact wavelength is not in the table, finds the nearest available.

    Parameters
    ----------
    wavelength : int
        Wavelength in nm

    Returns
    -------
    dict : {"HbO": float, "HbR": float} in cm^-1 / mM
    """
    if wavelength in EXT_COEFFS:
        return EXT_COEFFS[wavelength]

    # Find nearest wavelength
    available = sorted(EXT_COEFFS.keys())
    nearest = min(available, key=lambda x: abs(x - wavelength))

    logger.warning(f"Wavelength {wavelength} nm not in table, using nearest: {nearest} nm")
    return EXT_COEFFS[nearest]


# ---------------------------------------------------------------------
# 2. MBLL calculation for one channel
# ---------------------------------------------------------------------

def mbll_dual_wavelength(od1, od2, wl1, wl2, dpf, distance_cm):
    """
    Compute HbO/HbR concentration from two OD signals using MBLL.

    The Modified Beer-Lambert Law:
        ΔOD = ε * Δc * d * DPF

    Where:
        ΔOD = change in optical density (unitless)
        ε = extinction coefficient (cm^-1 / mM)
        Δc = change in concentration (mM)
        d = source-detector distance (cm)
        DPF = differential pathlength factor (unitless)

    For two wavelengths and two chromophores (HbO, HbR):
        [ΔOD1]   [ε_HbO(λ1)  ε_HbR(λ1)] [ΔHbO]
        [ΔOD2] = [ε_HbO(λ2)  ε_HbR(λ2)] [ΔHbR] * d * DPF

    Solving by matrix inversion gives HbO and HbR.

    Parameters
    ----------
    od1, od2 : array-like
        Optical density signals at wavelengths 1 and 2
    wl1, wl2 : int
        Wavelengths in nm
    dpf : float
        Differential pathlength factor (typically 5.0-6.0 for adult head)
    distance_cm : float
        Source-detector distance in cm

    Returns
    -------
    HbO, HbR : arrays
        Concentration changes in µM (micromolar)
    """
    # Get extinction coefficients (cm^-1 / mM)
    e1 = get_extinction_coefficient(wl1)
    e2 = get_extinction_coefficient(wl2)

    # Build extinction matrix
    # [ε_HbO(λ1)  ε_HbR(λ1)]
    # [ε_HbO(λ2)  ε_HbR(λ2)]
    eps_hbo_1 = e1["HbO"]
    eps_hbr_1 = e1["HbR"]
    eps_hbo_2 = e2["HbO"]
    eps_hbr_2 = e2["HbR"]

    # Determinant for matrix inversion
    denom = (eps_hbo_1 * eps_hbr_2 - eps_hbo_2 * eps_hbr_1)

    if abs(denom) < 1e-12:
        raise ValueError(f"Singular extinction matrix for λ={wl1},{wl2} nm")

    # Effective pathlength
    pathlength = dpf * distance_cm  # cm

    # MBLL inversion to get concentration in mM
    # Using Cramer's rule / matrix inversion:
    # [HbO]   1/det * [ ε_HbR(λ2)  -ε_HbR(λ1)] [OD1/L]
    # [HbR] =         [-ε_HbO(λ2)   ε_HbO(λ1)] [OD2/L]

    HbO = (eps_hbr_2 * od1 - eps_hbr_1 * od2) / (denom * pathlength)
    HbR = (-eps_hbo_2 * od1 + eps_hbo_1 * od2) / (denom * pathlength)

    # Convert mM → µM (multiply by 1000)
    HbO_uM = HbO * 1000.0
    HbR_uM = HbR * 1000.0

    return HbO_uM, HbR_uM


# ---------------------------------------------------------------------
# 3. Main conversion function
# ---------------------------------------------------------------------

def convert_od_to_concentration(df_od, channel_map, metadata=None):
    """
    Convert OD → concentration for all channels.

    Parameters
    ----------
    df_od : DataFrame
        Columns = OD signals (D1_R1_S3_WL850)
    channel_map : dict
        Built by loaders.py, gives wavelength grouping per CH#
        Each channel has: distance_mm, dpf, wavelength_pairs, columns
    metadata : dict, optional
        Legacy parameter, not needed (distance and DPF are in channel_map)

    Returns
    -------
    df_conc : DataFrame
        Columns = CH# HbO, CH# HHb
        Values in µM (micromolar)
    """

    df_conc = pd.DataFrame(index=df_od.index)

    successful_channels = 0
    skipped_channels = 0

    for ch_idx, info in channel_map.items():
        # Get distance from channel_map (already in mm)
        distance_mm = info.get("distance_mm")
        if distance_mm is None:
            logger.warning(f"Channel {ch_idx} missing distance_mm, skipping")
            skipped_channels += 1
            continue

        distance_cm = distance_mm / 10.0

        # Get DPF from channel_map (default to 6.0 if not present)
        # Note: DPF varies by tissue type and wavelength
        # Typical values for adult head: 5.0-6.0
        dpf_val = info.get("dpf", 6.0)

        # Get wavelength columns
        wl_dict = info.get("columns")  # e.g., {760: [...cols], 850: [...cols]}
        if wl_dict is None:
            logger.warning(f"Channel {ch_idx} missing columns dict, skipping")
            skipped_channels += 1
            continue

        wl_list = sorted(wl_dict.keys())
        if len(wl_list) != 2:
            logger.warning(f"Channel {ch_idx} has {len(wl_list)} wavelengths, need exactly 2")
            skipped_channels += 1
            continue

        wl1, wl2 = wl_list

        # Get column names for each wavelength
        cols_wl1 = wl_dict.get(wl1, [])
        cols_wl2 = wl_dict.get(wl2, [])

        if not cols_wl1 or not cols_wl2:
            logger.warning(f"Channel {ch_idx} missing column names for wavelengths")
            skipped_channels += 1
            continue

        col1 = cols_wl1[0]
        col2 = cols_wl2[0]

        # Check if columns exist in dataframe
        if col1 not in df_od.columns or col2 not in df_od.columns:
            logger.warning(f"Channel {ch_idx}: columns {col1} or {col2} not found in data")
            skipped_channels += 1
            continue

        od1 = df_od[col1].values
        od2 = df_od[col2].values

        try:
            HbO, HbR = mbll_dual_wavelength(
                od1, od2, wl1, wl2,
                dpf=dpf_val,
                distance_cm=distance_cm
            )

            df_conc[f"CH{ch_idx} HbO"] = HbO
            df_conc[f"CH{ch_idx} HHb"] = HbR
            successful_channels += 1

        except Exception as e:
            logger.error(f"MBLL conversion failed for channel {ch_idx}: {e}")
            skipped_channels += 1
            continue

    if df_conc.empty:
        logger.error("No channels were successfully converted!")
    else:
        logger.info(f"Successfully converted {successful_channels} channels "
                    f"({skipped_channels} skipped)")

        # Log some statistics about the output
        hbo_cols = [c for c in df_conc.columns if 'HbO' in c]
        if hbo_cols:
            sample_col = hbo_cols[0]
            logger.debug(f"Sample output range ({sample_col}): "
                         f"min={df_conc[sample_col].min():.2f}, "
                         f"max={df_conc[sample_col].max():.2f} µM")

    return df_conc


# ---------------------------------------------------------------------
# 4. Utility functions
# ---------------------------------------------------------------------

def print_extinction_coefficients():
    """Print all available extinction coefficients."""
    print("\nPrahl Extinction Coefficients (cm^-1 / mM)")
    print("=" * 50)
    print(f"{'Wavelength (nm)':<20} {'HbO':<15} {'HbR':<15}")
    print("-" * 50)
    for wl in sorted(EXT_COEFFS.keys()):
        print(f"{wl:<20} {EXT_COEFFS[wl]['HbO']:<15.4f} {EXT_COEFFS[wl]['HbR']:<15.4f}")
    print("=" * 50)
    print("\nSource: https://omlc.org/spectra/hemoglobin/summary.html")
    print("(Scott Prahl, compiled from Gratzer and Kollias data)")


if __name__ == "__main__":
    print_extinction_coefficients()