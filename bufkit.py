"""Module for loading Bufkit files."""
from __future__ import print_function

from collections import namedtuple
from datetime import datetime as dt
from itertools import zip_longest
from os import listdir
import os.path as path
import re

from math import sin, cos, pi

import numpy as np


class Profile(
    namedtuple(
        'Profile',
        [
                                                      # Location Data
            'stid',                                   # station id, e.g. KMSO
            'stnm',                                   # station number, usually USAF id number
            'lat',                                    # Latitude
            'lon',                                    # Longitude
            'elevation',                              # Station elevation in meters

                                                      # Time Data
            'time',                                   # valid time of the sounding
            'leadTime',                               # lead time in hours from model initialization

                                                      # Indexes
            'show',                                   # Showalter index
            'li',                                     # Lifted Index
            'swet',                                   # SWEAT Index
            'kinx',                                   # K index
            'lcl',                                    # LCL in mb
            'pwat',                                   # precipitable water in inches
            'totl',                                   # Total Totals index
            'cape',                                   # CAPE J/Kg
            'lclt',                                   # Potential Temperature at LCL, kelvin
            'cin',                                    # CIN J/Kg
            'eql',                                    # Equilibrium level, mb
            'lfc',                                    # Level of free convection, mb
            'brch',                                   # Bulk Richardson Number

                                                      # Profiles
            'pressure',                               # Pressure in mb
            'temp',                                   # Temp in deg C
            'wbt',                                    # Wet bulb temp in deg C
            'dewpoint',                               # Dew point in deg C
            'thetaE',                                 # equivalent potential temp in K
            'windDir',                                # wind direction
            'windSpd',                                # wind speed in knots
            'uWind',                                  # west to east wind in m/s
            'vWind',                                  # south to north wind in m/s
            'omega',                                  # vertical velocity in Pa/sec
            'cloud',                                  # Cloud fraction in percent
            'hgt',                                    # Height in meters
        ]
    )
):
    """A profile from a Bufkit data file.

    This object is a single profile parsed from a Bufkit data file and
    it contains no surface data.
    """
    
    __slots__ = ()
    
    def __str__(self):
        return "%s valid = %s lead time = %3d" % \
            (self.stid, self.time.strftime("%m/%d/%Y %H%MZ"), self.leadTime)


class Surface(
    namedtuple(
        'Surface',
        [
            'station',       # 6 digit station number
            'time',          # valid time
            'pmsl',          # mean sea level pressure hPa
            'pres',          # station pressure hPa
            'skin_temp',     # skin temperature C
            'soil_temp1',    # layer 1 soil temperature(K)
            'soil_temp2',    # layer 2 soil temperature (k)
            'snow',          # 1-hour snowfall (kg/m^2)
            'soil_moist',    # percent soil moisture availability
            'precip',        # 1-hour total precipitation (mm)
            'conv_precip',   # 1 hour convective precipitation (mm)
            'lcld',          # low cloud coverage (%)
            'mcld',          # mid cloud coverage (%)
            'hcld',          # high cloud coverage (%)
            'snow_ratio',    # snow ratio from explicit cloud scheme (%)
            'uWind',         # 10 meter u-wind (m/s)
            'vWind',         # 10 meter v-wind (m/s)
            'runoff',        # 1-hour accumulated runoff (mm)
                             # 1-hour accumulated baseflow ground water runoff (mm)
            'baseflow',
            'temp',          # 2-meter temperature (C)
            'q_2',           # 2-meter specific humidity
            'snow_pres',     # True/False
            'fzra_pres',     # True/False
            'ip_pres',       # True/False
            'rain_pres',     # True/False
            'u_storm',       # U-component storm motion (m/s)
            'v_storm',       # V-component storm motion (m/s)
            'helicity',      # storm relative helicity (m^2/s^2)
            'evap',          # 1-hour surface evaporation
            'cloud_base_p',  # Cloud base pressure (hPa)
            'visibility',    # Visibility (km)
            'dewpoint',      # 2-meter dew point (C)
        ]
    )
):
    """A surface section from a Bufkit data file.

    This object is a single set of surface data parsed from a Bufkit
    data file and it contains no upper air data.
    """
    
    __slots__ = ()
    
    def __str__(self):
        return "%s valid = %s" % (self.station, self.time.strftime("%m/%d/%Y %H%MZ"))


class Sounding(namedtuple('Sounding', ['profile', 'surface'])):
    """A matching profile and surface data from a Bufkit data file.
    
    The profile and surface data should be for the same valid time from
    the same file.
    """
    __slots__ = ()
    
    def __str__(self):
        return "Sounding " + str(self.profile)


def parse_str(text):
    """Parse a string as if it had been read directly from a Bufkit file.

    text is the string, or text data, that will be parsed according to
    Bufkit file format.

    Returns a Tuple with all the parsed Sounding objects sorted in order
    of increasing lead time. If there are any errors parsing a sounding
    it's silently skipped. If the whole file is invalid, then an empty
    Tuple is returned.
    """
    profiles = re.finditer(r'(^STID(.|\n)*?)(?=^(STID|STN YYMMDD/HHMM))', text, re.MULTILINE)
    
    profiles = map(lambda p: p.group(1), profiles)
    profiles = map(__parse_profile, profiles)
    
    surface_data = re.search(r'(^STN YYMMDD/HHMM)(.|\n)*', text, re.MULTILINE)
    
    if surface_data is not None:
        surface_data = __parse_surface_data(surface_data.group(0))
    
    # Merge profile and surface data
    sndings = []
    for p in profiles:
        sfc = next((sfc for sfc in surface_data if sfc.time == p.time), [None])
        if sfc is None:
            print("SUCKS")
        sndings.append(Sounding(p, sfc))
    
    return tuple(sorted(sndings, key=lambda x: x.profile.leadTime))


def parse_file(file_name):
    """Parse a Bufkit file to provide a Tuple of Soundings.

    file_name is a path that will be opened, read, and parsed.

    Returns a Tuple with all the parsed Sounding objects sorted in order
    of increasing lead time. If there are any errors parsing a sounding
    it's silently skipped. If the whole file is invalid, then an empty
    Tuple is returned.
    """
    
    with open(file_name, 'r') as f:
        text = f.read()
        return parse_str(text)


def get_profile_from_str(string_data, vt):
    """Get the lowest lead time profile that matches the valid time.

    string_data - either a single string representing a file's worth of
        data or a list of strings, each representing a file's worth of
        data.
    vt - the valid time we want the lowest lead time sounding for.

    Returns a single BufkitData.Sounding valid at vt
    """
    if isinstance(string_data, str):
        data = (string_data,)
    else:
        data = string_data
    
    snds = get_profiles_from_strs(data)
    
    for snd in snds:
        if snd.profile.time == vt:
            return snd
    return None


def get_profile(target_dir, vt):
    """Get the lowest lead time profile that matches the valid time.
    
    target_dir - the directory holding all the bufkit files
    vt - the valid time as a datetime.datetime object. UTC always 
        assumed.

    Returns a single BufkitData.Sounding valid at vt
    """
    snds = get_profiles(target_dir)
    
    for snd in snds:
        if snd.profile.time == vt:
            return snd
    return None


def get_profiles_from_strs(list_of_strings):
    """Parse all of the strings in the list into a merged time series.

       The single time series is composed of soundings taken from all 
       the strings. If there were mulitple strings that had a sounding 
       for a specific valid time, then the one with the shortest lead
       time will be selected. In this way several model runs can be 
       mosaiced together into a single time series.
    """
    time_series_list = map(parse_str, list_of_strings)
    return __merge_soundings_into_single_series(time_series_list)


def get_profiles(target_dir):
    """Load all of the profiles in a directory merged into a time series.

       The single time series is composed of soundings taken from all 
       the files. If there were mulitple files that had a sounding for a
       specific valid time, then the one with the shortest lead time
       will be selected. In this way several files/model runs can be 
       mosaiced together into a single time series.
    """
    # Remember what directory we loaded last, and cache it.
    if "loadedDirectory" not in get_profiles.__dict__ or \
            get_profiles.loadedDirectory != target_dir:
        get_profiles.loadedDirectory = target_dir
        
        # Load the files
        paths = (path.join(target_dir, f) for f in listdir(target_dir))
        time_series_list = map(parse_file, paths)
        
        profiles_list = __merge_soundings_into_single_series(time_series_list)
        
        get_profiles.profiles = profiles_list
    
    return get_profiles.profiles


def __merge_soundings_into_single_series(time_series_list):
    p_dict = {}
    for ts in time_series_list:
        for sounding in ts:
            key = sounding.profile.time.strftime("%Y%m%d%H%M")
            if key not in p_dict.keys():
                p_dict[key] = sounding
            elif sounding.profile.leadTime < p_dict[key].profile.leadTime:
                p_dict[key] = sounding
    
    profiles_list = list(p_dict.values())
    profiles_list.sort(key=lambda x: x.profile.time)
    
    return tuple(profiles_list)


def __parse_columns(text):
    tokens = text.split()
    numCols = sum(1 for token in tokens if token.strip() and token[0].isalpha())
    headers = tokens[0:numCols]
    data = tokens[numCols:]
    
    headerMapping = {}
    for i in range(numCols):
        headerMapping[headers[i]] = i
    
    data_dict = {}
    for header in headers:
        if r'/' in data[headerMapping[header]]:
            map_func = __parse_date_time
        else:
            map_func = float
        vals = map(map_func, data[headerMapping[header]::numCols])
        vals = map(lambda x: x if x != 9999.0 and x != -9999.0 else None, vals)
        data_dict[header] = tuple(vals)
    
    return data_dict


def __parse_date_time(text):
    text = text.strip()
    year = int(text[0:2]) + 2000
    month = int(text[2:4])
    day = int(text[4:6])
    hour = int(text[7:9])
    return dt(year, month, day, hour)


def __parse_profile(rawText):
    
    # Break the sounding into 3 sections
    regEx = re.compile(r"^STID(.|\n)*?^\s*$", re.MULTILINE)
    match = regEx.search(rawText)
    statInfo = ""
    if match is not None:
        statInfo = match.group(0)
    
    regEx = re.compile(r"^STID(.|\n)*?^\s*$((.|\n)*?)^PRES", re.MULTILINE)
    match = regEx.search(rawText)
    sndParams = ""
    if match is not None:
        sndParams = match.group(2)
    
    regEx = re.compile(r"^PRES(.|\n)*?^\s*$", re.MULTILINE)
    match = regEx.search(rawText)
    sndData = ""
    if match is not None:
        sndData = match.group(0)
    
    #
    # Get the station information
    #
    regEx = re.compile(r"STID = ([A-Z0-9]{3,4})?\s*(?=STNM)")
    stid = regEx.search(statInfo)
    if stid is not None:
        stid = stid.group(1)
    
    regEx = re.compile(r"STNM = ([0-9]+)\s+(?=TIME)")
    stnm = regEx.search(statInfo)
    if stnm is not None:
        stnm = stnm.group(1)
    
    if stid is None:
        stid = stnm
    
    regEx = re.compile(r"SLAT = (.*)\s+(?=SLON)")
    lat = regEx.search(statInfo)
    if lat is not None:
        lat = float(lat.group(1))
    
    regEx = re.compile(r"SLON = (.*)\s+(?=SELV)")
    lon = regEx.search(statInfo)
    if lon is not None:
        lon = float(lon.group(1))
    
    regEx = re.compile(r"SELV = (.*)\s+(?=STIM)")
    elevation = regEx.search(statInfo)
    if elevation is not None:
        elevation = float(elevation.group(1))
    
    #
    # Parse the time data
    #
    regEx = re.compile(r'TIME = ([0-9]{6}/[0-9]{2})00')
    time = regEx.search(statInfo)
    if time is not None:
        time = __parse_date_time(time.group(1))
    
    regEx = re.compile(r"STIM = ([0-9]+)\s+")
    leadTime = regEx.search(statInfo)
    if leadTime is not None:
        leadTime = int(leadTime.group(1))
    
    #
    # Helper function for parsing indexes
    #
    def parse_index(key):
        regEx = re.compile(key + r" = ([-+]?([0-9]*\.[0-9]+|[0-9]+))")
        val = regEx.search(sndParams)
        if val is not None:
            val = float(val.group(1))
            if val == -9999.0 or val == 9999.0:
                val = None
        return val
    
    #
    # Get the sounding indexes
    #
    show = parse_index("SHOW")
    li = parse_index("LIFT")
    swet = parse_index("SWET")
    kinx = parse_index("KINX")
    lcl = parse_index("LCLP")
    pwat = parse_index("PWAT")
    totl = parse_index("TOTL")
    cape = parse_index("CAPE")
    lclt = parse_index("LCLT")
    cin = parse_index("CINS")
    eql = parse_index("EQLV")
    lfc = parse_index("LFCT")
    brch = parse_index("BRCH")
    
    #
    # Parse the data section
    #
    data = __parse_columns(sndData)
    
    # Make list for the present elements
    pressure = data.get('PRES')
    temp = data.get('TMPC')
    wbt = data.get('TMWC')
    dewpoint = data.get('DWPC')
    thetaE = data.get('THTE')
    windDir = data.get('DRCT')
    windSpd = data.get('SKNT')
    omega = data.get('OMEG')
    cloud = data.get('CFRL')
    hgt = data.get('HGHT')
    
    # Function to translate (spd,dir) into (u,v)
    def spd_dir_to_uv(pair):
        spd, direct = pair
        direct_rad = direct / 180.0 * pi
        spd_ms = 0.514444 * spd
        return (-spd_ms * sin(direct_rad), -spd_ms * cos(direct_rad))
    
    uWind = None
    vWind = None
    if windSpd is not None and windDir is not None:
        sd_pairs = zip(windSpd, windDir)
        uv_pairs = map(spd_dir_to_uv, sd_pairs)
        uWind, vWind = zip(*list(uv_pairs))
        uWind = tuple(uWind)
        vWind = tuple(vWind)
    
    return Profile(
        stid, stnm, lat, lon, elevation, time, leadTime, show, li, swet, kinx, lcl, pwat, totl,
        cape, lclt, cin, eql, lfc, brch, pressure, temp, wbt, dewpoint, thetaE, windDir, windSpd,
        uWind, vWind, omega, cloud, hgt
    )


def __parse_surface_data(rawText):
    sfc_data = __parse_columns(rawText)
    station = sfc_data.get('STN')
    if station is None:
        return None
    
    time = sfc_data.get('YYMMDD/HHMM')
    if time is None:
        return None
    
    pmsl = sfc_data.get('PMSL', [])
    pres = sfc_data.get('PRES', [])
    skin_temp = sfc_data.get('SKTC', [])
    soil_temp1 = sfc_data.get('STC1', [])
    soil_temp2 = sfc_data.get('STC2', [])
    snow = sfc_data.get('SNFL', [])
    soil_moist = sfc_data.get('WTNS', [])
    precip = sfc_data.get('P01M', [])
    conv_precip = sfc_data.get('C01M', [])
    lcld = sfc_data.get('LCLD', [])
    mcld = sfc_data.get('MCLD', [])
    hcld = sfc_data.get('HCLD', [])
    snow_ratio = sfc_data.get('SNRA', [])
    uWind = sfc_data.get('UWND', [])
    vWind = sfc_data.get('VWND', [])
    runoff = sfc_data.get('R01M', [])
    baseflow = sfc_data.get('BFGR', [])
    temp = sfc_data.get('T2MS', [])
    q_2 = sfc_data.get('Q2MS', [])
    snow_pres = sfc_data.get('WXTS', [])
    fzra_pres = sfc_data.get('WXTP', [])
    ip_pres = sfc_data.get('WXTZ', [])
    rain_pres = sfc_data.get('WXTR', [])
    u_storm = sfc_data.get('USTM', [])
    v_storm = sfc_data.get('VSTM', [])
    helicity = sfc_data.get('HLCY', [])
    evap = sfc_data.get('SLLH', [])
    cloud_base_p = sfc_data.get('CDBP', [])
    visibility = sfc_data.get('VSBK', [])
    dewpoint = sfc_data.get('TD2M', [])
    
    def make_surface(big_tuple):
        return Surface(*big_tuple)
    
    return tuple(
        map(
            make_surface,
            zip_longest(
                station, time, pmsl, pres, skin_temp, soil_temp1, soil_temp2, snow, soil_moist,
                precip, conv_precip, lcld, mcld, hcld, snow_ratio, uWind, vWind, runoff, baseflow,
                temp, q_2, snow_pres, fzra_pres, ip_pres, rain_pres, u_storm, v_storm, helicity,
                evap, cloud_base_p, visibility, dewpoint
            )
        )
    )


if __name__ == "__main__":
    import os.path as path
    
    print("Testing bufkit: ")
    test_path = path.join("test_data", "17090212.nam4km_kmso.buf")
    ts = parse_file(test_path)
    
    print("Number of items: %d" % sum(1 for _ in iter(ts)))
    print("Number 3 is %s" % str(ts[2]))
    
    for snd in iter(ts):
        print("T=%sC, DP=%sC" % (snd.surface.temp, snd.surface.dewpoint))
