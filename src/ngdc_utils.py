"""
Downloads attitude or orbit data from NGDC using bs4, saves to data dir.
"""
from fontTools.varLib.models import nonNone
from icecream import ic
import glob
from netCDF4 import Dataset, num2date
import os
import numpy as np
from datetime import datetime, timezone, timedelta
import requests
import gzip
import shutil
from tqdm import tqdm
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)


def download_file(url, out_path):
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception(f"Failed to download: {url}")
    total = int(r.headers.get('content-length', 0))
    with open(out_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(out_path)) as bar:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def get_ngdc_file(date_str: str, file_type: str = "att") -> str:
    """
    Downloads a single .nc file (either attitude or orbit) for the specified date.

    Args:
        date_str: Date in 'YYYY-MM-DD' format
        file_type: 'att' or 'pop'

    Returns:
        Full path to the decompressed .nc file
    """
    if file_type not in ["att", "pop"]:
        raise ValueError("file_type must be either 'att' or 'pop'")

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    ymd = dt.strftime("%Y/%m")
    base_url = f"https://www.ngdc.noaa.gov/dscovr/data/{ymd}/"
    r = requests.get(base_url)
    if r.status_code != 200:
        raise Exception(f"Failed to access directory listing: {base_url}")

    soup = BeautifulSoup(r.text, "html.parser")
    date_start = dt.strftime("%Y%m%d000000")
    date_end = dt.strftime("%Y%m%d235959")
    file_prefix = f"oe_{file_type}_dscovr_s{date_start}_e{date_end}"
    matching_files = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(file_prefix)]
    if not matching_files:
        raise Exception(f"No matching {file_type} file found for date {date_str}")
    filename = matching_files[0]
    url = base_url + filename
    gz_path = os.path.join(OUT_DIR, filename)
    nc_path = gz_path.replace(".gz", "")

    if not os.path.exists(gz_path) and not os.path.exists(nc_path):
        print(f"Downloading: {url}")
        download_file(url, gz_path)
    else:
        print(f"Already downloaded: {gz_path if os.path.exists(gz_path) else nc_path}")

    if os.path.exists(gz_path) and not os.path.exists(nc_path):
        print(f"Decompressing: {gz_path}")
        with gzip.open(gz_path, 'rb') as f_in, open(nc_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"✅ Decompressed to: {nc_path}")
        os.remove(gz_path)

    return nc_path


def get_files_for_range(start_date: str, end_date: str, file_type: str = "att") -> list:
    """
    Downloads multiple files between two dates (inclusive).

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        file_type: 'att' or 'pop'

    Returns:
        List of decompressed file paths
    """
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    all_paths = []

    while dt_start <= dt_end:
        date_str = dt_start.strftime("%Y-%m-%d")
        try:
            path = get_ngdc_file(date_str, file_type=file_type)
            all_paths.append(path)
        except Exception as e:
            print(f"⚠️ {e}")
        dt_start += delta

    return all_paths

def find_pop_files(start_time, end_time, base_path="../data/raw"):
    """
    Find one POP file per day between start_time and end_time (inclusive).

    Args:
        start_time (datetime): Start of the range.
        end_time (datetime): End of the range.
        base_path (str): Directory where POP files are stored.

    Returns:
        List of POP NetCDF file paths.
    """
    all_files = []
    current_day = start_time

    while current_day <= end_time:
        sstr = current_day.strftime("%Y%m%d")
        pattern = os.path.join(base_path, f"oe_pop_dscovr_s{sstr}*.nc")
        matches = sorted(glob.glob(pattern))
        ic(f"Searching: {pattern}")
        ic(f"Matches: {matches}")
        if matches:
            all_files.append(matches[0])  # Grab the first match for that day
        else:
            print(f"⚠️ No file found for {sstr}")
        current_day += timedelta(days=1)

    if not all_files:
        raise FileNotFoundError(f"No POP files found between {start_time} and {end_time}")

    return all_files

def load_pop_gse_coords(nc_path):
    """
    Loads GSE coordinates and timestamps from a DSCOVR Predictive Orbit Product (POP) NetCDF file.

    NOTE ON 48-ENTRY FILES:
    Some POP files contain 48 entries instead of the expected 24. These appear to include
    *two interleaved datasets*, where:
        - Even indices (0, 2, 4, ...) represent the standard hourly cadence
        - Odd indices (1, 3, 5, ...) seem to correspond to secondary points (e.g., 1 second after hour)

    This behavior is inconsistent across dates and may relate to maneuver activity or something else.
    For now it is advised to choose either the even or odd dataset.

    Any questions please contact s-auriemma via github or sarah.auriemma@noaa.gov

    This function returns both even- and odd-indexed datasets if 48 entries are present, along with the raw full set.
    """
    with Dataset(nc_path, "r") as nc:
        # ic(nc.variables.keys())
        # ic(nc.__dict__)

        x = nc.variables["sat_x_gse"][:]
        y = nc.variables["sat_y_gse"][:]
        z = nc.variables["sat_z_gse"][:]

        time_var = nc.variables["time"]
        units = nc.variables["time"].units # should be ms since 1970-01-01T0Z

        lenoftime = time_var.shape[0]
        if lenoftime == 48:
            print('⚠️ Interleaved 48-entry file detected')
            print("See note in `load_pop_gse_coords` function header about this known issue.")

            idx_even = np.arange(0, 48, 2)
            idx_odd = np.arange(1, 48, 2)

            time_even = num2date(time_var[idx_even], units=units, only_use_cftime_datetimes=False)
            time_odd = num2date(time_var[idx_odd], units=units, only_use_cftime_datetimes=False)

            return {
                "even": (np.array(time_even), x[idx_even], y[idx_even], z[idx_even]),
                "odd": (np.array(time_odd), x[idx_odd], y[idx_odd], z[idx_odd]),
                "raw": (np.array(num2date(time_var[:], units=units)), x, y, z)
            }

        elif lenoftime == 24:
            time = num2date(time_var[:], units=units)
            return {"default": (np.array(time), x, y, z)}

        else:
            raise ValueError(f"Unexpected POP record length: {lenoftime} (expected 24 or 48)")


def convert_cftime_to_datetime_utc(cftime_array):
    # Cast to float64 if it's not already
    times_ms = np.array(cftime_array, dtype='float64')
    times_ms = np.round(times_ms)

    # Convert ms since epoch to datetime
    times = [
        datetime.utcfromtimestamp(ms / 1000.0).replace(tzinfo=timezone.utc)
        for ms in times_ms
    ]
    return np.array(times)


import matplotlib.pyplot as plt


def plot_interleaved_pop_datasets(time_even, x_even, y_even, z_even,
                                  time_odd, x_odd, y_odd, z_odd,
                                  label="POP File Comparison"):
    """
    Plot even vs odd entries in a 48-entry interleaved POP file.

    Parameters:
    - time_even, x_even, y_even, z_even: Data from even indices (0, 2, ..., 46)
    - time_odd,  x_odd,  y_odd,  z_odd:  Data from odd indices (1, 3, ..., 47)
    - label (str): Optional title for the figure
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(time_even, x_even, label="Even X", marker='o', linestyle='-')
    axs[0].plot(time_odd, x_odd, label="Odd X", marker='x', linestyle='--')
    axs[0].set_ylabel("X (km)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_even, y_even, label="Even Y", marker='o', linestyle='-')
    axs[1].plot(time_odd, y_odd, label="Odd Y", marker='x', linestyle='--')
    axs[1].set_ylabel("Y (km)")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time_even, z_even, label="Even Z", marker='o', linestyle='-')
    axs[2].plot(time_odd, z_odd, label="Odd Z", marker='x', linestyle='--')
    axs[2].set_ylabel("Z (km)")
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle(label)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    att_paths, pop_paths = None, None
    # att_paths = get_files_for_range("2021-11-25", "2021-11-26", file_type="att")
    # pop_paths = get_files_for_range("2019-05-01", "2019-05-01", file_type="pop")

    if att_paths:
        print("\nDownloaded ATT files:")
        for path in att_paths:
            print("  -", path)
    if pop_paths:
        print("\nDownloaded POP files:")
        for path in pop_paths:
            print("  -", path)

    # load_pop_gse_coords('../data/raw/oe_pop_dscovr_s20211125000000_e20211125235959_p20211126022149_pub.nc')
    data = load_pop_gse_coords('../data/raw/oe_pop_dscovr_s20211125000000_e20211125235959_p20211126022149_pub.nc')

    if "default" in data: # 24 timestamps
        t, x, y, z = data["default"]
        print("Default cadence (hourly):")
        for i in range(6):
            print(f"{i}: t = {t[i]}, x = {x[i]:.2f}, y = {y[i]:.2f}, z = {z[i]:.2f}")

    elif "even" in data and "odd" in data: # 48 timestamps
        t_even, x_even, y_even, z_even = data["even"]
        t_odd, x_odd, y_odd, z_odd = data["odd"]
        print("Interleaved cadence:")
        for i in range(6):
            print(f"Even {i}: t = {t_even[i]}, x = {x_even[i]:.2f}, y = {y_even[i]:.2f}, z = {z_even[i]:.2f}")
            print(f"Odd  {i}: t = {t_odd[i]},  x = {x_odd[i]:.2f},  y = {y_odd[i]:.2f},  z = {z_odd[i]:.2f}")
        plot_interleaved_pop_datasets(t_even, x_even, y_even, z_even, t_odd, x_odd, y_odd, z_odd, label="POP Ephemeris Interleaving Check")
