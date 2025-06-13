"""
Fetches DSCOVR ephemeris data from JPL Horizons using SunPy.
"""

from datetime import datetime, timezone
import numpy as np
from icecream import ic
from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates.frames import GeocentricSolarEcliptic
import astropy.units as u


def fetch_jpl_coordinates(start_time, end_time, step='1m', target='DSCOVR'):
    """
    Fetch GSE-equivalent coordinates of a spacecraft from JPL Horizons using SunPy.
    """
    # Construct time range dictionary for Horizons query
    time_range = {
        'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'stop': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'step': step
    }

    # Fetch coordinate object with velocity disabled
    coords = get_horizons_coord(target, time=time_range, include_velocity=False)

    # # Convert observation times to UTC datetime
    # times = coords.obstime.to_datetime(timezone.utc)
    #
    # # Extract ICRS Cartesian coordinates in kilometers
    # x = coords.cartesian.x.to('km').value
    # y = coords.cartesian.y.to('km').value
    # z = coords.cartesian.z.to('km').value

    # Convert to GSE coordinates
    coords_gse = coords.transform_to(GeocentricSolarEcliptic(obstime=coords.obstime))

    # Extract time and Cartesian coordinates in km
    times = coords.obstime.datetime
    x = coords_gse.cartesian.x.to(u.km).value
    y = coords_gse.cartesian.y.to(u.km).value
    z = coords_gse.cartesian.z.to(u.km).value

    return np.array(times), x, y, z


if __name__ == "__main__":
    start = datetime(2021, 11, 25, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2021, 11, 25, 1, 0, 0, tzinfo=timezone.utc)
    times, x, y, z = fetch_jpl_coordinates(start, end, step='10m', target='DSCOVR')

    print(f"First timestamp: {times[0]}")
    print(f"First position: x = {x[0]:.2f} km, y = {y[0]:.2f} km, z = {z[0]:.2f} km")

