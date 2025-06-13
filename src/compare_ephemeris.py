from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from icecream import ic
import matplotlib.pyplot as plt
import cftime

from ngdc_utils import find_pop_files, load_pop_gse_coords, convert_cftime_to_datetime_utc
from sscweb_utils import fetch_sscweb_coordinates
from jpl_utils import fetch_jpl_coordinates

def plot_ephemeris_components(t_pop, x_pop, y_pop, z_pop,
                              t_ssc, x_ssc, y_ssc, z_ssc):
    plt.figure(figsize=(12, 9))

    for i, (label, pop, ssc) in enumerate(zip(
        ['X', 'Y', 'Z'],
        [x_pop, y_pop, z_pop],
        [x_ssc, y_ssc, z_ssc]
    )):
        plt.subplot(3, 1, i+1)
        plt.plot(t_pop, pop, label=f'POP {label}', alpha=0.6)
        plt.plot(t_ssc, ssc, 'o-', label=f'SSC {label}', markersize=3)
        plt.ylabel(f"{label} (km)")
        plt.grid(True)
        plt.legend()

    plt.suptitle("DSCOVR GSE Position Components: ngdc vs. SSCWeb")
    plt.xlabel("Time (UTC)")
    plt.tight_layout()
    plt.show()


def compare_ephemeris(start_time, end_time):
    # --- Load POP data from ngdc---
    pop_paths = find_pop_files(start_time, end_time)
    print("Found POP files:")
    for path in pop_paths:
        print(f"  - {path}")

    all_t_pop, all_x_pop, all_y_pop, all_z_pop = [], [], [], []
    for pop_file in pop_paths:
        data = load_pop_gse_coords(pop_file)

        if "default" in data:
            t_pop, x_pop, y_pop, z_pop = data["default"]
        elif "even" in data:  # if interleaved, default to even indices
            t_pop, x_pop, y_pop, z_pop = data["even"]
        else:
            raise ValueError(f"Unrecognized POP file structure for {pop_file}")

        all_t_pop.append(t_pop)
        all_x_pop.append(x_pop)
        all_y_pop.append(y_pop)
        all_z_pop.append(z_pop)

    # Concatenate data across days
    t_pop = np.concatenate(all_t_pop)
    x_pop = np.concatenate(all_x_pop)
    y_pop = np.concatenate(all_y_pop)
    z_pop = np.concatenate(all_z_pop)
    # ic(t_pop)

    # Only convert if necessary. don't know if this is needed at all
    # if isinstance(t_pop[0], cftime.DatetimeBase):
    #     t_pop = convert_cftime_to_datetime_utc(t_pop)

    # --- Load SSCWeb Data ---
    t_ssc, x_ssc, y_ssc, z_ssc = fetch_sscweb_coordinates(start_time, end_time)

    # --- Interpolation ---
    t_ssc_ts = np.array([t.timestamp() for t in t_ssc])
    t_pop_ts = np.array([t.timestamp() for t in t_pop])

    fx = interp1d(t_pop_ts, x_pop, bounds_error=False, fill_value=np.nan)
    fy = interp1d(t_pop_ts, y_pop, bounds_error=False, fill_value=np.nan)
    fz = interp1d(t_pop_ts, z_pop, bounds_error=False, fill_value=np.nan)

    dx = x_ssc - fx(t_ssc_ts)
    dy = y_ssc - fy(t_ssc_ts)
    dz = z_ssc - fz(t_ssc_ts)
    dr = np.sqrt(dx**2 + dy**2 + dz**2)

    # --- Load JPL Horizons Data ---
    t_jpl, x_jpl, y_jpl, z_jpl = fetch_jpl_coordinates(start_time, end_time, step="10m", target="DSCOVR")

    t_jpl_ts = np.array([t.timestamp() for t in t_jpl])
    dx_jpl = x_jpl - fx(t_jpl_ts)
    dy_jpl = y_jpl - fy(t_jpl_ts)
    dz_jpl = z_jpl - fz(t_jpl_ts)
    dr_jpl = np.sqrt(dx_jpl**2 + dy_jpl**2 + dz_jpl**2)


    # --- Summary Stats ---
    print(f"\nCompared with {len(pop_paths)} ngdc file(s):")
    print(f"ΔR Max:  {np.nanmax(dr):.2f} km")
    print(f"ΔR Mean: {np.nanmean(dr):.2f} km")
    print(f"ΔR RMS:  {np.sqrt(np.nanmean(dr**2)):.2f} km")

    print(f"\nJPL vs ngdc:")
    print(f"ΔR Max:  {np.nanmax(dr_jpl):.2f} km")
    print(f"ΔR Mean: {np.nanmean(dr_jpl):.2f} km")
    print(f"ΔR RMS:  {np.sqrt(np.nanmean(dr_jpl**2)):.2f} km")

    # --- Preview Data ---
    print("\nngdc data:")
    for i in range(3):
        print(f"{i}: t = {t_pop[i]}, x = {x_pop[i]:.2f} km, y = {y_pop[i]:.2f} km, z = {z_pop[i]:.2f} km")

    print("\nSSCWeb data:")
    for i in range(3):
        print(f"{i}: t = {t_ssc[i]}, x = {x_ssc[i]:.2f} km, y = {y_ssc[i]:.2f} km, z = {z_ssc[i]:.2f} km")

    print("\njpl horizons data:")
    for i in range(3):
        print(f"{i}: t = {t_jpl[i]}, x = {x_jpl[i]:.2f} km, y = {y_jpl[i]:.2f} km, z = {z_jpl[i]:.2f} km")

    return dr, t_pop, x_pop, y_pop, z_pop, t_ssc, x_ssc, y_ssc, z_ssc, t_jpl, x_jpl, y_jpl, z_jpl


def plot_position_error(t_ssc, dr_ssc, t_jpl, dr_jpl):
    plt.figure(figsize=(10, 5))
    plt.plot(t_ssc, dr_ssc, label="SSC vs POP", alpha=0.8)
    plt.plot(t_jpl, dr_jpl, label="JPL vs POP", alpha=0.8)
    plt.title("DSCOVR Position Error vs. POP")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Distance Error (km)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = datetime(2021, 11, 25, 0, 0, 0)
    end_time = datetime(2021, 11, 25, 23, 59, 59)

    dr, t_pop, x_pop, y_pop, z_pop, t_ssc, x_ssc, y_ssc, z_ssc, t_jpl, x_jpl, y_jpl, z_jpl = compare_ephemeris(start_time, end_time)

    # plot_ephemeris_components(t_pop, x_pop, y_pop, z_pop, t_ssc, x_ssc, y_ssc, z_ssc)
