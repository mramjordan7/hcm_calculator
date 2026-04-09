"""
Hot Cracking Susceptibility Index Mathematical Models.

Contains the core mathematical functions for calculating hot cracking
susceptibility indexes from Scheil solidification data. Implements Kou's model
(CSI), Clyne-Davies model (CSC), and Berglund model (HCS) along with Mode 2
cooling time calculations. These functions are called by index_calculations.py
to process individual alloy results.
"""
from typing import Optional
import numpy as np
from scipy.interpolate import interp1d


def kous_model(fraction_solid: np.ndarray, temperature_celcius: np.ndarray,
               fs_min: float = 0.64, fs_max: float = 0.9604) -> float:
    """
    Calculate Kou's hot cracking susceptibility index (KHCS).
 
    Maximum absolute steepness of T vs sqrt(fs) in the vulnerable solidification
    range. Upper limit at fs=0.9604 (sqrt(fs)=0.98) based on Xia & Kou (2020) —
    extensive bonding occurs at sqrt(fs)=0.99, making cracking unlikely beyond
    this point. Uses np.gradient for derivative calculation.
 
    Parameters
    ----------
    fraction_solid : np.ndarray
        Fraction solid array.
    temperature_celcius : np.ndarray
        Temperature array (°C).
    fs_min : float
        Lower fraction solid bound (default = 0.64).
    fs_max : float
        Upper fraction solid bound (default = 0.9604, i.e. sqrt(fs) = 0.98).
 
    Returns
    -------
        KHCS value: float
    """
    fs_arr = np.asarray(fraction_solid)
    temp_arr = np.asarray(temperature_celcius)
 
    # Filter to vulnerable range
    mask = (fs_arr >= fs_min) & (fs_arr <= fs_max)
    fs_filtered = fs_arr[mask]
    temp_filtered = temp_arr[mask]
 
    # Fallback to last two points if fewer than 2 points in range
    if len(fs_filtered) < 2:
        if len(fs_arr) < 2:
            return np.nan
        fs_filtered = fs_arr[-2:]
        temp_filtered = temp_arr[-2:]
 
    sqrt_fs = np.sqrt(fs_filtered)
    steepness = np.abs(np.gradient(temp_filtered, sqrt_fs))
 
    return float(np.max(steepness))


def mode2_cooling(temperature_celcius: np.ndarray, fraction_solid: np.ndarray,
                  latent_heat: np.ndarray, heat_capacity: np.ndarray
                  ) -> np.ndarray:
    """
    Calculate cooling time using Mode 2 (constant heat flow) model.

    Parameters
    ----------
    temperature_celcius : np.ndarray
        Temperature array (°C).
    fraction_solid : np.ndarray
        Fraction solid array.
    latent_heat : np.ndarray
        Latent heat array (J/g).
    heat_capacity : np.ndarray
        Heat capacity array (J/gK).

    Returns
    -------
    time_values : np.ndarray
        Solidification time array (seconds).

    """
    time_values = np.zeros(len(temperature_celcius))
    total_steps = len(temperature_celcius) - 1
    constant = -1e6

    for i in range(total_steps):
        temp_i = temperature_celcius[i]
        temp_i_next = temperature_celcius[i+1]
        fi = fraction_solid[i]
        fi_plus1 = fraction_solid[i+1]
        cp = heat_capacity[i+1]
        lh = latent_heat[i+1]

        # Calculate time step
        term1 = cp * (temp_i_next - temp_i)
        term2 = lh * (fi_plus1 - fi)
        t_prev = time_values[i]
        t_next = ((term1 + term2) / constant) + t_prev
        time_values[i+1] = t_next

    return time_values


def csc_model(fraction_solid: np.ndarray, temperature_celcius: np.ndarray,
              latent_heat: np.ndarray, heat_capacity: np.ndarray,
              fs_targets: Optional[list] = [0.4, 0.9, 0.99]) -> float:
    """
    Calculate Clyne-Davies solidification cracking susceptibility (CSC) index.

    Parameters
    ----------
    fraction_solid : np.ndarray
        Fraction solid array.
    temperature_celcius : np.ndarray
        Temperature array (°C).
    latent_heat : np.ndarray
        Latent heat array (J/g).
    heat_capacity : np.ndarray
        Heat capacity array (J/gK).
    fs_targets : Optional[list], optional
        Fraction of solid targets for time interpolation.
        The defaults are:
            - 0.4 and 0.9 for the available stress relief time
            - 0.9 and 0.99 for the vulnerable crack time

    Returns
    -------
    csc: float
        CSC value.
    """
    time_values = mode2_cooling(temperature_celcius, fraction_solid,
                                latent_heat, heat_capacity)

    # Interpolate times at fraction solid targets
    time_at_fs = {}
    fs_values = np.asarray(fraction_solid)

    for fs_target in fs_targets:
        time_interp = interp1d(fs_values, time_values, kind='linear',
                               fill_value='extrapolate')
        time_at_fs[fs_target] = float(time_interp(fs_target))

    # Extract specific time targets
    time_fs_40 = time_at_fs.get(0.4, np.nan)
    time_fs_90 = time_at_fs.get(0.9, np.nan)
    time_fs_99 = time_at_fs.get(0.99, np.nan)

    # print(f"Cooling times - fs=0.4: {time_fs_40:.4f}s, fs=0.9:\
    #       {time_fs_90:.4f}s, fs=0.99: {time_fs_99:.4f}s")

    # # Calculate CSC
    # if not (np.isfinite([time_fs_40, time_fs_90, time_fs_99]).all()):
    #     print("Warning: Invalid times for CSC calculation")
    #     return np.nan

    denominator = time_fs_90 - time_fs_40
    # if abs(denominator) < 1e-10:
    #     print("Warning: Near-zero denominator in CSC calculation")
    #     return np.nan

    csc = (time_fs_99 - time_fs_90) / denominator

    return float(csc)


def hcs_model(csc_value: float, solidification_range: float,
              grain_size: float = 0.01) -> float:    # 0.01mm = 10µm
    """
    Calculate Hot Cracking Susceptibility (HCS) index from Berglund et.al.

    Parameters
    ----------
    csc_value : float
        CSC index value.
    solidification_range : float
        Solidification termperature range (°C).
    grain_size : float
        Asssumed grain size (default is 10.0 micrometers).

    Returns
    -------
    hcs: float
        HCS index value.
    """
    if not np.isfinite([csc_value, solidification_range]).all():
        return np.nan

    hcs = csc_value * grain_size * solidification_range

    return float(hcs)
