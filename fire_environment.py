"""Analysis of Bufkit soundings in the context of the wildfire environment.

This module is concerned with aspects of the environment that are not
directly related to the fire plume or the atmosphere's interactions 
with it. These are parameters that may influence the fire at the 
surface independent of the plume.

dcape considers potential thunderstorm outflow winds.
HDW is the hot-dry-wind index.

References:

Severe Prediction Center website, https://www.spc.noaa.gov/exper/soundings/help/params2.html. 
    Accessed May 26th, 2021.

Srock AF, Charney JJ, Potter BE, Goodrick SL. The Hot-Dry-Windy Index: A New Fire Weather Index. 
    Atmosphere. 2018; 9(7):279. https://doi.org/10.3390/atmos9070279
"""

from collections import namedtuple
from itertools import dropwhile, takewhile

try:
    import pyrometeopy.formulas as wxf
except Exception:
    import formulas as wxf


def dcape(sounding):
    """Calculate the DCAPE of the provided sounding.

    From https://www.spc.noaa.gov/exper/soundings/help/params2.html

    Downdraft CAPE (DCAPE)
    Follow a moist adiabat down to the ground from the minimum theta-e value in the lowest 400 mb 
    of the sounding, and then the calculate area. DCAPE values greater than 1000 J/kg have been
    associated with increasing potential for strong downdrafts and damaging outflow winds.

    Returns the DCAPE in J/kg.
    """
    pcl = _dcape_parcel(sounding)
    descent_iter = _moist_adiabatic_descend_parcel(pcl, sounding)
    
    def convert_to_buoyancy_like_value(x):
        et, pt, h, p = x
        et = wxf.theta_kelvin(p, et)
        pt = wxf.theta_kelvin(p, pt)
        buoyancy_val = (pt - et) / et
        return (buoyancy_val, h)
    
    descent_vals = tuple(convert_to_buoyancy_like_value(x) for x in descent_iter)
    
    # Build iterator to iterate over two values at a time so we can integrate
    # with the trapezoid rule
    steps = zip(descent_vals, descent_vals[1:])
    
    int_buoyancy = 0.0
    
    for v0, v1 in steps:
        b0, h0 = v0
        b1, h1 = v1
        
        dz = h1 - h0
        # dz < 0.0 for descent
        assert dz <= 0.0, "%.2f,%.2f" % (h0, h1)
        
        int_buoyancy += (b0 + b1) * dz
    
    dcape_val = int_buoyancy * -wxf.g / 2.0
    
    return dcape_val


def _dcape_parcel(sounding):
    """Find the starting parcel for anaylyzing dcape."""
    from fire_plumes import Parcel
    
    # Find the top pressure which is 400hPa above the surface.
    sfc_pressure = sounding.surface.pres
    if sfc_pressure is None:
        pprofile = iter(sounding.profile.pressure)
        while sfc_pressure is None:
            sfc_pressure = next(pprofile)
    top_pressure = sfc_pressure - 400.0
    
    # Get all the ingredients needed to make a parcel from the profile.
    levels = zip(sounding.profile.temp, sounding.profile.dewpoint, sounding.profile.pressure)
    # Wrap them in a parcel namedtuple.
    levels = (Parcel(*x) for x in levels)
    # Only look at levels with all the values we need.
    levels = (x for x in levels if x.is_complete())
    # Only take values up to the pressure of the top level
    levels = takewhile(lambda x: x.pressure >= top_pressure, levels)
    # Calculate the theta-e value and pair it with the parcel.
    level_vals = ((x, wxf.theta_e_kelvin(*x)) for x in levels)
    # Filter out None theta-e values
    level_vals = (x for x in level_vals if x[1] is not None)
    # Find the level with the lowest theta e
    pcl, _ = min(level_vals, key=lambda x: x[1])
    
    assert pcl is not None, "None parcel."
    
    return pcl


def _moist_adiabatic_descend_parcel(parcel, sounding):
    """Descend a parcel to the surface moist adiabatically.

    Returns an iterator with elements of 
    (env_virt_t, pcl_virt_t, hgt, pres). The iterator moves in a top down
    order through the sounding.
    """
    # Work from the top down.
    env_profile = zip(
        reversed(sounding.profile.pressure),
        reversed(sounding.profile.temp),
        reversed(sounding.profile.dewpoint),
        reversed(sounding.profile.hgt),
    )
    # Remove levels missing values.
    env_profile = (x for x in env_profile if all(x))
    # Skip levels above the level of our starting parcel
    env_profile = dropwhile(lambda x: x[0] < parcel.pressure, env_profile)
    # Calculate the environmental virtual temperature
    env_profile = ((x[0], wxf.virtual_temperature_c(x[1], x[2], x[0]), x[3]) for x in env_profile)
    # Remove any levels with None
    env_profile = (x for x in env_profile if all(x))
    
    pcl_theta_e = wxf.theta_e_kelvin(parcel.temperature, parcel.dew_point, parcel.pressure)
    
    def calc_parcel_vt(press):
        parcel_t = wxf.temperature_c_from_theta_e_saturated_and_pressure(press, pcl_theta_e)
        return wxf.virtual_temperature_c(parcel_t, parcel_t, press)
    
    full_profile = ((x[1], calc_parcel_vt(x[0]), x[2], x[0]) for x in env_profile)
    # Remove levels that we could not calculate a value.
    full_profile = (x for x in full_profile if all(x))
    
    return full_profile


def hdw(sounding, elevation=None):
    """Calculate the Hot-Dry-Windy index for a sounding."""
    
    bottom = sounding.profile.elevation
    if elevation is not None and elevation > bottom:
        bottom = elevation
    top = bottom + 500.0
    
    # Find the station pressure for the surface adjusted temperature and dew point.
    bottom_p = sounding.surface.pres
    i = 0
    while bottom_p is None or sounding.profile.hgt[i] < bottom:
        bottom_p = sounding.profile.pressure[i]
        i += 1
    
    vals = zip(
        sounding.profile.hgt, sounding.profile.temp, sounding.profile.dewpoint,
        sounding.profile.windSpd, sounding.profile.pressure
    )

    vals = filter(lambda x_: x_[0] >= bottom, vals)
    vals = tuple(takewhile(lambda x: x[0] <= top, vals))
    
    # Filter out None values
    vpds = (
        (x[1], x[2], x[4])
        for x in vals
        if x[1] is not None and x[2] is not None and x[4] is not None
    )
    # Convert to potential temperature and specific humidity for reducing to the surface.
    vpds = ((wxf.theta_kelvin(x[2], x[0]), wxf.specific_humidity(x[1], x[2])) for x in vpds)
    # Finish surface adjustment.
    vpds = (
        (
            wxf.temperature_c_from_theta(x[0], bottom_p),
            wxf.dew_point_from_p_and_specific_humidity(bottom_p, x[1])
        ) for x in vpds
    )
    
    vpds = ((wxf.vapor_pressure_liquid_water(x[0]) - \
            wxf.vapor_pressure_liquid_water(x[1])) for x in vpds)
    max_vpd = max(vpds)
    
    max_wspd = max(x[3] for x in vals if x[3] is not None)
    max_wspd = wxf.knots_to_mps(max_wspd)
    
    return max_vpd * max_wspd


if __name__ == "__main__":
    import os.path as path
    from bufkit import parse_file
    
    file_names = (
        "17082812.nam4km_kmso.buf", "17082818.nam4km_kmso.buf", "17082900.nam4km_kmso.buf",
        "17082906.nam4km_kmso.buf", "17082912.nam4km_kmso.buf", "17082918.nam4km_kmso.buf",
        "17083000.nam4km_kmso.buf", "17083006.nam4km_kmso.buf", "17083012.nam4km_kmso.buf",
        "17083018.nam4km_kmso.buf", "17083100.nam4km_kmso.buf", "17083106.nam4km_kmso.buf",
        "17083112.nam4km_kmso.buf", "17083118.nam4km_kmso.buf", "17090100.nam4km_kmso.buf",
        "17090106.nam4km_kmso.buf", "17090112.nam4km_kmso.buf", "17090118.nam4km_kmso.buf",
        "17090200.nam4km_kmso.buf", "17090206.nam4km_kmso.buf", "17090212.nam4km_kmso.buf",
        "17090218.nam4km_kmso.buf", "17090300.nam4km_kmso.buf", "17090306.nam4km_kmso.buf",
        "17090312.nam4km_kmso.buf"
    )
    
    paths = (path.join("test_data", fname) for fname in file_names)
    test_datas = (parse_file(p) for p in paths)
    
    for test_data in test_datas:
        print("\n\nWorking on model initial time %s" % test_data[0].profile.time)
        
        for snd in test_data:
            print("DCAPE: {:4.0f} J/kg  -  HDW {:4.0f}, {:4.0f}".format(dcape(snd), hdw(snd), 
                hdw(snd, 1520)))
