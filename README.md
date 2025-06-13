# DSCOVR Ephemeris Validation

This project compares predictive orbit product (POP) ephemeris data for the DSCOVR spacecraft—sourced from NGDC NetCDF files—with external sources including SSCWeb and JPL Horizons. The main goal is to identify discrepancies, especially those related to coordinate transformations and unexpected time cadence anomalies in NGDC POP files.

- Validate NGDC predictive orbit products
- Cross-check GSE coordinate consistency against SSCWeb and JPL Horizons
- Identify anomalies in satellite position (ΔR), velocity jumps, or coordinate offsets
- Investigate time cadence irregularities and possible data formatting bugs

## Features 
- Fetches, downloads, and extracts NGDC predictive orbit (pop) NetCDF files via `ngdc_utils.py`
- Compares with SSCWeb and JPL Horizons data
- Visualizes differences in GSE coordinates

[//]: # (## Known Issues)


## Todo/future works
- auto-download pop files from ngdc
- attitude/quaternion based coordinate transforms

## Author 
Sarah Auriemma (s-auriemma @ github)

Work done for: CIRES / NOAA NCEI / STP - Space weather group