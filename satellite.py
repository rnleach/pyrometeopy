"""Tools to extract and analyze data from GOES-R."""

import datetime as dt
import netCDF4 as nc
import numpy as np
from pathlib import Path
import s3fs


def download_goes_hotspot_characterization(folder, start, end, satellite="G17", full_disk=False):
    """Download hotspot characterization data from Amazon AWS S3.

    Queries the appropriate S3 folders for file names. If they 
    already exist in the specified folder, they are not 
    downloaded again.

    Arguments:
    folder is the location to store the files
    start is the start time
    end is the end time and must be after start
    satellite must be "G17" or "G16" to choose which satellite to use.
    full_disk means to use the full disk instead of the conus imagery.

    Returns: A list of file names on the local machine for your processing.
    """
    if full_disk:
        product = 'ABI-L2-FDCF'
    else:
        product = 'ABI-L2-FDCC'
    
    return download_goes_data(folder, start, end, product, satellite)


def download_goes_data(folder, start, end, product, satellite="G17"):
    """Download GOES data from Amazon AWS S3.

    First checks the local archive in 'folder' and checks for at least
    11 files for a date and hour. If there are not enough files, then 
    it queries the appropriate S3 folders for file names. If they 
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
    
    if not isinstance(folder, Path):
        folder = Path(folder)
    
    assert folder.is_dir()
    
    start, bucket = _validate_satellite_dates(satellite, start, end)
    if start is None:
        return []
    
    # Get a list of files we already have downloaded.
    current_files = tuple(f for f in folder.iterdir() if "ABI-L2" in f.name and f.suffix == ".nc")
    
    # Files older than this are too old to be missing, and must be
    # permanently missing. So we shouldn't check for them again, just
    # remember that htey are missing so we can skip them.
    too_old_to_be_missing = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    
    # The list of hours with missing data.
    missing_data_path = folder / "missing_data.txt"
    if missing_data_path.exists():
        with open(missing_data_path, "r") as mdf:
            missing_data = list(l.strip() for l in mdf if l.strip() != "")
    else:
        missing_data = []
    
    current_time = start
    result_list = []
    while current_time < end:
        
        # Check to see how many matching files we have
        time_prefix = current_time.strftime("_s%Y%j%H")
        missing_key = "{}{}".format(satellite, time_prefix)
        local_files_this_hour = (f for f in current_files if satellite in f.name)
        local_files_this_hour = tuple(f for f in local_files_this_hour if time_prefix in f.name)
        
        # Should be 12 per hour for CONUS 
        if len(local_files_this_hour) >= 11:                                                   
            result_list.extend(local_files_this_hour)
        
        elif missing_key not in missing_data:
            
            result_list.extend(
                _download_files(
                    current_time, bucket, product, folder, too_old_to_be_missing, missing_data,
                    missing_key
                )
            )
        
        # Move ahead an hour
        current_time += dt.timedelta(hours=1)
    
    # Remember the missing!
    with open(missing_data_path, "w") as mdf:
        for line in missing_data:
            mdf.write(line)
            mdf.write("\n")
    
    return result_list


def _download_files(
    current_time, s3_bucket, product, target_dir, too_old_to_be_missing, missing_data, missing_key
):
    """Download the files for the hour given by current_time.

    The remote directory is built from current_time, s3_bucket, and product.
    target_dir is the directory on the local file system to store downloaded
        data.
    too_old_to_be_missing and missing data keep track of files that are 
        missing and very unlikely to ever be updated.

    Returns a generator that yields the file name on the local file system of 
    any downloaded files. If the target file was already downloaded, it just 
    yields the local file name without redownloading it.
    """
    time_path = current_time.strftime("%Y/%j/%H")
    remote_dir = "{}/{}/{}".format(s3_bucket, product, time_path)
    
    # Use the anonymous credentials to access public data
    fs = s3fs.S3FileSystem(anon=True)
    
    remote_files = list(fs.ls(remote_dir))
    local_files = (f.split('/')[-1] for f in remote_files)
    local_files = (target_dir / f for f in local_files)
    
    files = tuple(zip(remote_files, local_files))
    
    # If there's some missing data, remember!
    if len(files) < 11 and current_time < too_old_to_be_missing:
        missing_data.append(missing_key)
    
    for remote, local in files:
        
        if not local.exists() or not local.is_file():
            print("Downloading", local)
            fs.get(remote, str(local))
        
        yield local
    
    return None


def _validate_satellite_dates(satellite, start, end):
    """Validate the start and end times for the satellite.

    Uses the known operational dates of the satellites to
    adjust the start date if needed. It also selects the
    Amazon S3 bucket to use.

    Returns: a tuple of (start, S3 bucket). If the start
    and end times are invalid, it returns (None, None).
    """
    GOES_16_OPERATIONAL = dt.datetime(2017, 12, 18, 17, 30, tzinfo=dt.timezone.utc)
    GOES_17_OPERATIONAL = dt.datetime(2019, 2, 12, 18, tzinfo=dt.timezone.utc)
    
    # Satellite specific checks and setup.
    if satellite == "G17":
        
        if end < GOES_17_OPERATIONAL:
            return (None, None)
        if start < GOES_17_OPERATIONAL:
            start = GOES_17_OPERATIONAL
        
        bucket = 's3://noaa-goes17'
    
    elif satellite == "G16":
        
        if end < GOES_16_OPERATIONAL:
            return (None, None)
        if start < GOES_16_OPERATIONAL:
            start = GOES_16_OPERATIONAL
        
        bucket = 's3://noaa-goes16'
    
    return (start, bucket)


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
    
    def sw_corner(self):
        """Get the southwest corner as a tuple (lat, lon)."""
        return (self.min_lat, self.min_lon)
    
    def ne_corner(self):
        """Get the northeast corner as a tuple (lat, lon)."""
        return (self.max_lat, self.max_lon)
    
    def corners(self):
        """Get a tuple of the corners, each themselves a tuple."""
        return (self.sw_corner(), self.ne_corner())


def total_fire_power_time_series(files, bounding_box):
    """Create time series of total fire power.

    Arguments:
    files is a list of NetCDF4 files with fire power data.
        Either the paths or opened nc.Dataset's can be
        passed in.
    bounding_box is the bounding boxe to gather data for.

    Returns: A dictionary where valid time is the key and
    the value is the fire power.
    """
    
    assert isinstance(bounding_box, BoundingBox)
    bb = bounding_box
    
    results = {}
    for f in files:
        if isinstance(f, nc.Dataset):
            nc_data = f
            # Ownder opened, they take responsibility for closing.
            needs_close = False
        else:
            nc_data = nc.Dataset(f)
            needs_close = True

        try:
            time = get_valid_time(nc_data)
            
            if time >= bb.start and time <= bb.end:
                total_power = get_total_fire_power(nc_data, bb)
                
                results[time] = total_power
        
        except Exception as e:
            if isinstance(f, nc.Dataset):
                msg = f.filepath()
            else:
                msg = f
            print("Error, skipping {} for error {}".format(msg, e))
            continue

        if needs_close:
            nc_data.close()
    
    return results

def is_valid_netcdf_file(nc_data):
    """Various QC checks on the data in the file."""
    fname = Path(nc_data.filepath()).name

    start_str = fname.split("_")[3][1:-1]
    start_fname = dt.datetime.strptime(start_str + " UTC", "%Y%j%H%M%S %Z", )
    start_fname = start_fname.replace(tzinfo=dt.timezone.utc)
    end_str = fname.split("_")[4][1:-1]
    end_fname = dt.datetime.strptime(end_str + " UTC", "%Y%j%H%M%S %Z")
    end_fname = end_fname.replace(tzinfo=dt.timezone.utc)

    avg_fname = start_fname + (end_fname - start_fname) / 2

    vtime = get_valid_time(nc_data)
    if vtime is None:
        return False

    diff = (avg_fname - vtime).total_seconds()

    if diff > 60:
        return False

    return True


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
    try:
        time = nc_dataset.variables['time_bounds'][:]
        time = sum(time) / len(time)
        time = _SATELLITE_EPOCH + dt.timedelta(seconds=time)
        
        return time
    except Exception:
        return None


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
    min_lat, min_lon = bounding_box.sw_corner()
    max_lat, max_lon = bounding_box.ne_corner()
    
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
