"""Tools to extract and analyze data from GOES-R data."""

import datetime as dt
import netCDF4 as nc
import numpy as np

def get_total_fire_power(nc_dataset, southwest_corner, northeast_corner):
    """Extract the total fire power in the area of interest.

    Arguments:
    nc_dataset is a dataset returned by nc.Dataset(filename). It is
        assumed this the fire files. Usually they have ABI-L2-FDCC
        in the file name.
    southwest_corner is a (lat,lon) tuple of the southwest corner of 
        area of interest.
    northeast_corner is a (lat,lon) tuple of the northeast corner of 
        area of interest.

    Returns: a tuple with the valid time and the fire power in
    gigawatts, (valid_time, fire_power_gw).
    """
    time = nc_dataset.variables['time_bounds'][:]
    time = sum(time) / len(time)
    time = _SATELLITE_EPOCH + dt.timedelta(seconds=time)

    idxs = _get_grid_cell_indexes(
        nc_dataset.variables['goes_imager_projection'],
        nc_dataset.variables['x'], 
        nc_dataset.variables['y'], 
        southwest_corner, 
        northeast_corner)

    powers = list(nc_dataset.variables['Power'][:].flatten()[idxs])
    powers = (x for x in powers if x != 'masked')
    total_power = sum(powers) / 1000  # This makes it Gigawatts

    return (time, total_power)


# EPOCH - satellite data stored in NetCDF files uses this datetime as
# the epoch. Time values in the files are in seconds since this time.
_SATELLITE_EPOCH = dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)


def _get_grid_cell_indexes(proj, xs, ys, southwest_corner, northeast_corner):
    """Get the indexes of the desired pixels in a satellite image.

    I found this algorithm at 
    https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm

    After implementing it I checked the ranges of lat-lon coords 
    produced by using it against the values reported in the 
    NetCDF4 files.

    Arguments:
    proj is the projection from a GOES-R NetCDF4 file.
    xs is a 1D array of from the NetCDF4 file with the x-coordinates.
    ys is a 1D array of from the NetCDF4 file with the y-coordinates.
    southwest_corner is a (lat,lon) tuple of the southwest corner of 
        area of interest.
    northeast_corner is a (lat,lon) tuple of the northeast corner of 
        area of interest.

    Returns: A list of indexes into a flattened array of the values
    from a satellite image.
    """
    # Unpack values from the projection
    eq_rad = proj.semi_major_axis
    polar_rad = proj.semi_minor_axis
    h = proj.perspective_point_height + eq_rad
    lon0 = proj.longitude_of_projection_origin

    # Unpack values from the area we want to grab the data
    min_lat, min_lon = southwest_corner
    max_lat, max_lon = northeast_corner

    # Calculate the lat and lon grids
    xs, ys = np.meshgrid(xs, ys)
    a_vals = np.power(np.sin(xs), 2.0) + \
            np.power(np.cos(xs), 2.0) * (np.power(np.cos(ys), 2.0) + \
                eq_rad * eq_rad / polar_rad / polar_rad * np.power(np.sin(ys), 2.0))
    b_vals = -2 * h * np.cos(xs) * np.cos(ys)
    c_val = h * h - eq_rad * eq_rad

    rs = (-b_vals -
          np.sqrt(np.power(b_vals, 2.0) - 4 * a_vals * c_val)) / (2 * a_vals)

    sx = rs * np.cos(xs) * np.cos(ys)
    sy = -rs * np.sin(xs)
    sz = rs * np.cos(xs) * np.sin(ys)

    lats = np.arctan((eq_rad *eq_rad * sz) \
            / (polar_rad * polar_rad * np.sqrt(np.power(h - sx, 2.0) + np.power(sy, 2.0))))
    lats = np.degrees(lats)

    lons = np.radians(lon0) - np.arctan(sy / (h - sx))
    lons = np.degrees(lons)

    # Flatten the arrays so we get a 1D list of indexes
    lats = lats.flatten()
    lons = lons.flatten()

    # Filter out values not in our bounding box
    lats = np.where(np.logical_and(lats >= min_lat, lats <= max_lat))[0]
    lons = np.where(np.logical_and(lons >= min_lon, lons <= max_lon))[0]
    idxs = list(set(lons).intersection(set(lats)))

    return idxs


