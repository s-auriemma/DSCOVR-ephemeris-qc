"""
Fetches DSCOVR ephemeris data from SSCWeb in various coordinate systems.
"""

from sscws.sscws import SscWs
import numpy as np
from datetime import datetime, timezone
from icecream import ic

def fetch_sscweb_coordinates(start_time, end_time, coord_system="GSE"):
    """
    Fetch position data for DSCOVR from SSCWeb in the specified coordinate system.

    Args:
        start_time (datetime): Start time (UTC).
        end_time (datetime): End time (UTC).
        coord_system (str): Coordinate system to use (e.g., "GSE", "GSM").

    Returns:
        time (np.ndarray): Array of datetime objects.
        x, y, z (np.ndarray): Arrays of coordinates in kilometers.
    """
    ssc = SscWs()
    result = ssc.get_locations(
        ['dscovr'],
        [start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
         end_time.strftime('%Y-%m-%dT%H:%M:%SZ')]
    )

    data = result['Data'][0]
    times = [t.replace(tzinfo=timezone.utc) for t in data['Time']]

    coords_list = data['Coordinates']
    coord_entry = next((c for c in coords_list if c['CoordinateSystem'].name.upper() == coord_system.upper()), None)

    if coord_entry is None:
        raise ValueError(f"No coordinates found for system '{coord_system}'.")

    x = np.array(coord_entry['X'])
    y = np.array(coord_entry['Y'])
    z = np.array(coord_entry['Z'])

    return np.array(times), x, y, z


def list_available_observatories():
    """
    Get all observatory IDs from SSCWeb.

    Returns:
        List of observatory ID strings.
    """
    ssc = SscWs()
    obs = ssc.get_observatories()
    return [o['Id'] for o in obs['Observatories']]

if __name__ == "__main__":
    from datetime import timedelta

    start = datetime(2021, 11, 25, 0, 0, 0)
    end = start + timedelta(hours=1)

    t, x, y, z = fetch_sscweb_coordinates(start, end, coord_system="GSE")

    print(f"First timestamp: {t[0]}")
    print(f"First position (GSE): x = {x[0]:.2f} km, y = {y[0]:.2f} km, z = {z[0]:.2f} km")
