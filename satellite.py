"""Tools to extract and analyze data from GOES-R."""

import datetime as dt
import netCDF4 as nc
import numpy as np
import os.path as path
import s3fs


def download_goes_hotspot_characterization(folder, start, end, satellite="G17"):
    """Download hotspot characterization data for Amazon AWS S3.

    Queries the appropriate S3 folders for file names. If they 
    already exist in the specified folder, they are not 
    downloaded again.

    Arguments:
    folder is the location to store the files
    start is the start time
    end is the end time and must be after start
    satellite must be "G17" or "G16" to choose which satellite to use.

    Returns: A list of file names on the local machine for your processing.
    """
    
    assert isinstance(start, dt.datetime)
    assert isinstance(end, dt.datetime)
    assert end > start
    assert satellite == "G17" or satellite == "G16"
    
    if satellite == "G17":
        bucket = 's3://noaa-goes17'
    elif satellite == "G16":
        bucket = 's3://noaa-goes16'
    
    product = 'ABI-L2-FDCC'
    
    # Use the anonymous credentials to access public data
    fs = s3fs.S3FileSystem(anon=True)
    
    current_time = start
    result_list = []
    while current_time < end:
        
        time_path = current_time.strftime("%Y/%j/%H")
        remote_dir = "{}/{}/{}".format(bucket, product, time_path)
        
        remote_files = np.array(fs.ls(remote_dir))
        local_files = (f.split('/')[-1] for f in remote_files)
        local_files = (path.join(folder, f) for f in local_files)
        
        files = tuple(zip(remote_files, local_files))
        
        for remote, local in files:
            result_list.append(local)
            
            if not path.exists(local):
                print("Downloading", local)
                fs.get(remote, local)
        
        # Move ahead an hour
        current_time += dt.timedelta(hours=1)
    
    return result_list


class BoundingBox:
    """Simple spatial AND temporal boundaries for satellite data."""
    
    def __init__(self, southwest_corner, northeast_corner, start, end, name):
        """Create a simple bounding box.

        southwest_corner is a (lat,lon) tuple of the southwest corner of 
            area of interest.
        northeast_corner is a (lat,lon) tuple of the northeast corner of 
            area of interest.
        """
        assert isinstance(start, dt.datetime)
        assert isinstance(end, dt.datetime)
        assert start < end
        
        self.min_lat, self.min_lon = southwest_corner
        self.max_lat, self.max_lon = northeast_corner
        self.start, self.end = start, end
        self.name = name
        
        return


def total_fire_power_time_series(files, bounding_boxes):
    """Create time series of total fire power.

    Arguments:
    files is a list of NetCDF4 files with fire power data.
    bounding_boxes is a collection bounding boxes to gather
        data for.

    Returns: A dictionary where the bounding box names are 
    the keys and the values numpy arrays with valid time
    and fire power. 

    {"name":(ndarray of valid times, ndarray of total fire power), ..}
    """
    if not isinstance(bounding_boxes, (list, tuple)):
        bbs = (bounding_boxes,)
    else:
        bbs = bounding_boxes
    
    result_times = {}
    result_powers = {}
    for bb in bbs:
        result_times[bb.name] = []
        result_powers[bb.name] = []
    
    for f in files:
        if isinstance(f, nc.Dataset):
            nc_data = f
        else:
            nc_data = nc.Dataset(f)
        
        for bb in bbs:
            time = get_valid_time(nc_data)
            
            if time >= bb.start and time <= bb.end:
                total_power = get_total_fire_power(nc_data, bb)
                
                result_times[bb.name].append(time)
                result_powers[bb.name].append(total_power)
    
    results = {}
    for area in result_times.keys():
        results[area] = (np.array(result_times[area]), np.array(result_powers[area]))
    
    return results


def get_valid_time(nc_dataset):
    """Extract the valid time.

    This is the average of the starting and ending times of
    the scan.

    Arguments:
    nc_dataset is a dataset returned by nc.Dataset(filename).
        It is assumed that these are fire files. Usually 
        they have ABI-L2-FDCC in the file name.

    Returns: the valid time as a datetime.datetime object.
    """
    time = nc_dataset.variables['time_bounds'][:]
    time = sum(time) / len(time)
    time = _SATELLITE_EPOCH + dt.timedelta(seconds=time)
    
    return time


# EPOCH - satellite data stored in NetCDF files uses this datetime as
# the epoch. Time values in the files are in seconds since this time.
_SATELLITE_EPOCH = dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)


def get_total_fire_power(nc_dataset, bounding_box):
    """Extract the total fire power in the area of interest.

    Arguments:
    nc_dataset is a dataset returned by nc.Dataset(filename).
        It is assumed that these are fire files. Usually 
        they have ABI-L2-FDCC in the file name.
    bounding_box is the area from which to extract data.

    Returns: The fire power in gigawatts.
    """
    idxs = _get_grid_cell_indexes(
        nc_dataset.variables['goes_imager_projection'], nc_dataset.variables['x'],
        nc_dataset.variables['y'], bounding_box
    )
    
    powers = list(nc_dataset.variables['Power'][:].flatten()[idxs])
    powers = (x for x in powers if x != 'masked')
    total_power = sum(powers) / 1000   # This makes it Gigawatts
    
    return total_power


def _get_grid_cell_indexes(proj, xs, ys, bounding_box):
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
    bounding_box is the area we need to get the indexes for.

    Returns: A list of indexes into a flattened array of the values
    from a satellite image.
    """
    # Unpack values from the projection
    eq_rad = proj.semi_major_axis
    polar_rad = proj.semi_minor_axis
    h = proj.perspective_point_height + eq_rad
    lon0 = proj.longitude_of_projection_origin
    
    # Unpack values from the area we want to grab the data
    min_lat, min_lon = bounding_box.min_lat, bounding_box.min_lon
    max_lat, max_lon = bounding_box.max_lat, bounding_box.max_lon
    
    # Calculate the lat and lon grids
    xs, ys = np.meshgrid(xs, ys)
    a_vals = np.power(np.sin(xs), 2.0) + \
            np.power(np.cos(xs), 2.0) * (np.power(np.cos(ys), 2.0) + \
                eq_rad * eq_rad / polar_rad / polar_rad * np.power(np.sin(ys), 2.0))
    b_vals = -2 * h * np.cos(xs) * np.cos(ys)
    c_val = h * h - eq_rad * eq_rad
    
    rs = (-b_vals - np.sqrt(np.power(b_vals, 2.0) - 4 * a_vals * c_val)) / (2 * a_vals)
    
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
