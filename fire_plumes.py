"""Analysis of Bufkit soundings in the context of wildfire plumes."""

from collections import namedtuple
from itertools import dropwhile, takewhile
import math

import bufkit
import formulas as wxf
import numpy as np


class Parcel(namedtuple("Parcel", ("temperature", "dew_point", "pressure"))):
    """Represents a parcel of air and its properties.

    temperature and dew_point are in °C and pressure is in hPa.
    """
    
    __slots__ = ()
    
    def __str__(self):
        return "Parcel: %.2f °C / %.2f °C %.0f hPa" % \
            (self.temperature, self.dew_point, self.pressure)


class ParcelProfile(
    namedtuple(
        "ParcelProfile",
        ("env_virt_temp", "parcel_virt_temp", "pres", "hgt", "lcl_height", "parcel")
    )
):
    """The profile of a lifted parcel.

    env_virt_temp is the environmental virtual temperature in °C,
    parcel_virt_temp is the parcel's virtual temperature in °C,
    pres is the pressure level in hPa,
    hgt is the geopotential height in meters.

    The above are all lists or tuples of the same length. An index
    into those arrays represents the values at a given level of the
    parcel's ascent.

    lcl_height is the height in meters of the lifting condensation
        level.
    parcel is the original parcel that was lifted.

    This structure can be analyzed for things like CAPE, or the
    equilibrium level.
    """
    
    __slots__ = ()
    
    def __str__(self):
        return "Parcel_Profile: starting parcel - %s, LCL - %.0f m" % \
            (self.lcl_height, self.parcel)


class ParcelAscentAnalysis(
    namedtuple(
        "ParcelAscentAnalysis", (
            "lcl", "el", "top", "max_int_buoyancy", "level_max_int_buoyancy", "dry_int_buoyancy",
            "cloud_depth"
        )
    )
):
    """Results from analyzing a ParcelProfile.

    lcl is the lifting condnesation level in meters,
    el is the equilibrium level in meters. In situations where there is
        multiple equilibrium levels, this is the highest one.
    top is the level, in meters, where the integrated buoyancy becomes
        zero. So all the CAPE has been realized and it keeps rising past
        the equilibrium level and turns it's kinetic energy back to
        potential energy. In many cases we can't analyze the top because
        of insufficient data in the upper atmosphere (moisture), so this
        value may be None.
    max_int_buoyancy is the maximum amount of CAPE. Usually this occurs
        at the equilibrium level. However, sometimes there are stable
        layers with more than one equilibrium level in the profile. In
        this case we want to know how strong the updraft is at its
        strongest.
    level_max_int_buoyancy is the level where the max_int_buoyancy 
        occurs. This is usually the equilibrium level, but in cases where
        there are multiple equilibrium levels, this is the one with the
        strongest updraft potential.
    dry_int_buoyancy is the amount of the integrated buoyancy that comes
        from just the dry-adiabatic portion of the lifting. This can be
        considered the amount of CAPE that results from the heating by
        the fire.
    cloud_depth is the lcl to top distance. If top is None, we just use
        the highest possible level in the sounding. If this value is
        negative, that means that the parcel never reached the LCL, and
        it fell short of creating a cloud by this distance.
    """
    
    __slots__ = ()
    
    def __str__(self):
        if self.lcl is None:
            lcl = "None "
        else:
            lcl = "%5.0fm" % self.lcl
        
        if self.el is None:
            el = " None "
        else:
            el = "%5.0fm" % self.el
        
        if self.level_max_int_buoyancy is None:
            lmib = " None "
        else:
            lmib = "%5.0fm" % self.level_max_int_buoyancy
        
        if self.top is None:
            top = " None "
        else:
            top = "%5.0fm" % self.top
        
        return ("ParcelAscentAnalysis: LCL - %s, EL - %s, Top - %s, " +
                "Max Int. Bouyancy - %4.0f J/Kg, Level MIB - %s," +
                "Dry Int. Bouyancy - %4.0f J/Kg, Cloud depth - %+5.0fm") % \
            (
                lcl,
                el,
                top,
                self.max_int_buoyancy,
                lmib,
                self.dry_int_buoyancy,
                self.cloud_depth,
        )
    
    def percent_wet_cape(self):
        """The percentage of the buoyancy that is from latent heat release.

        This is the (max_integrated_buoyancy - dry_integrated_buoyancy) /
        max_integrated_buoyancy.
        """
        if self.max_int_buoyancy is None or self.dry_int_buoyancy is None:
            return None
        if self.max_int_buoyancy == 0.0:
            return 0.0
        
        return (self.max_int_buoyancy - self.dry_int_buoyancy) / \
            self.max_int_buoyancy


class BlowUpAnalysis(
    namedtuple("BlowUpAnalysis", ("dt_cloud", "dt_lmib_blow_up", "dz_lmib_blow_up"))
):
    """An analysis of how much heating it will take to cause a blow up.

    dt_cloud is the amount of heating required to create a pyrocumulus.
    dt_lmib_blow_up is the amount of heating required to cause the
        level of maximum integrated buoyancy to blow up, or quickly jump.
    dz_lmib_blow_up is the change in height (meters) of the level of
        maximum integrated buoyancy between dt_lmib_blow_up - 0.5°C of
        heating and dt_lmib_blow_up + 0.5°C of heating. This is 
        basically the magnitude of the blowup.
    """
    
    __slots__ = ()
    
    def __str__(self):
        if self.dt_cloud is None:
            cld = "None"
        else:
            cld = "%4.1f" % self.dt_cloud
        
        if self.dt_lmib_blow_up is None:
            lmib = "None"
        else:
            lmib = "%4.1f" % self.dt_lmib_blow_up
        
        if self.dz_lmib_blow_up is None:
            dz = " None"
        else:
            dz = "%5.0f" % self.dz_lmib_blow_up
        
        return ("BlowUpAnalysis: ΔT Cld - %s," + " ΔT LMIB - %s, ΔZ LMIB - %s") % (cld, lmib, dz)


def lift_parcel(parcel, sounding):
    """Simulate lifting a parcel through a sounding focusing on buoyancy.

    Since buoyancy is the focus of lifting, virtual temperatures for
    both the environment and the parcel are given.

    parcel is a Parcel as defined above.
    sounding is a Sounding from the BufkitData module.

    Returns a ParcelProfile.
    """
    assert (len(sounding.profile.pressure) == len(sounding.profile.temp)), "different length lists"
    assert (
        len(sounding.profile.pressure) == len(sounding.profile.dewpoint)
    ), "different length lists"
    assert (len(sounding.profile.pressure) == len(sounding.profile.hgt)), "different length lists"
    
    env_profile = zip(
        sounding.profile.pressure, sounding.profile.temp, sounding.profile.dewpoint,
        sounding.profile.hgt
    )
    env_profile = (x for x in env_profile if not any(y is None for y in x))
    env_profile = map(
        lambda x: (x[0], wxf.virtual_temperature_c(x[1], x[2], x[0]), x[3]), env_profile
    )
    env_profile = tuple(env_profile)
    
    p_lcl, _ = wxf.press_and_temp_k_at_lcl(parcel.temperature, parcel.dew_point, parcel.pressure)
    
    parcel_theta = wxf.theta_kelvin(parcel.pressure, parcel.temperature)
    parcel_theta_e = wxf.theta_e_kelvin(parcel.temperature, parcel.dew_point, parcel.pressure)
    dry_parcel_sh = wxf.specific_humidity(parcel.dew_point, parcel.pressure)
    
    pres, env_virt_temp, hgt = zip(*env_profile)
    # Reverse for numpy interpolation
    pres = tuple(reversed(pres))
    env_virt_temp = tuple(reversed(env_virt_temp))
    hgt = tuple(reversed(hgt))
    
    # Assert parcel is within range of the sounding.
    assert parcel.pressure <= pres[-1], "%.1f <= %.1f" % (parcel.pressure, pres[-1])
    
    def calc_parcel_vt(press):
        if press is None:
            return None
        try:
            if press > p_lcl:
                parcel_t = wxf.temperature_c_from_theta(parcel_theta, press)
                parcel_dp = wxf.dew_point_from_p_and_specific_humidity(press, dry_parcel_sh)
            else:
                parcel_t = wxf.temperature_c_from_theta_e_saturated_and_pressure(
                    press, parcel_theta_e
                )
                parcel_dp = parcel_t
        except Exception:
            return None
        return wxf.virtual_temperature_c(parcel_t, parcel_dp, press)
    
    to_ret_env_virt_t = []
    to_ret_pcl_virt_t = []
    to_ret_hgt = []
    to_ret_pres = []
    
    def add_to_arrays(env, pcl, height, p):
        to_ret_env_virt_t.append(env)
        to_ret_pcl_virt_t.append(pcl)
        to_ret_hgt.append(height)
        to_ret_pres.append(p)
    
    # Add the first level as the level of the parcel
    env_vt = np.interp(parcel.pressure, pres, env_virt_temp)
    pcl_vt = calc_parcel_vt(parcel.pressure)
    height = np.interp(parcel.pressure, pres, hgt)
    
    add_to_arrays(env_vt, pcl_vt, height, parcel.pressure)
    
    old_p = parcel.pressure
    old_h = height
    old_pcl_t = pcl_vt
    old_env_t = env_vt
    lcl_height = None
    
    snding_iter = dropwhile(lambda x: x[0] >= parcel.pressure, env_profile)
    
    for p, env_vt, height in snding_iter:
        
        assert p < old_p
        assert height > old_h, "%.2f > %.2f (%.1f -> %.1f)" % (height, old_h, old_p, p)
        
        pcl_vt = calc_parcel_vt(p)
        if pcl_vt is None:
            break
        
        old_buoyancy = old_pcl_t - old_env_t
        buoyancy = pcl_vt - env_vt
        # Check to see if the parcel and environment parcels cross.
        crossing_vals = None
        lcl_vals = None
        if old_buoyancy <= 0.0 and buoyancy > 0.0:
            p_cross = np.interp(0.0, (old_buoyancy, buoyancy), (old_p, p))
            cross_env_t = np.interp(p_cross, pres, env_virt_temp)
            cross_height = np.interp(p_cross, pres, hgt)
            crossing_vals = (cross_env_t, cross_env_t, cross_height, p_cross)
            
            assert p_cross <= old_p and p_cross >= p
            assert cross_height >= old_h and cross_height <= height
        
        if old_buoyancy > 0.0 and buoyancy <= 0.0:
            p_cross = np.interp(0.0, (buoyancy, old_buoyancy), (p, old_p))
            cross_env_t = np.interp(p_cross, pres, env_virt_temp)
            cross_height = np.interp(p_cross, pres, hgt)
            crossing_vals = (cross_env_t, cross_env_t, cross_height, p_cross)
            
            assert p_cross <= old_p and p_cross >= p
            assert cross_height >= old_h and cross_height <= height
        
        if old_p > p_lcl and p <= p_lcl:
            lcl_height = np.interp(
                p_lcl,
                pres,
                hgt,
            )
            lcl_env_t = np.interp(p_lcl, pres, env_virt_temp)
            lcl_pcl_t = np.interp(p_lcl, (p, old_p), (pcl_vt, old_pcl_t))
            lcl_vals = (lcl_env_t, lcl_pcl_t, lcl_height, p_lcl)
            
            assert p_lcl <= old_p and p_lcl >= p
            assert lcl_height >= old_h and lcl_height <= height
        
        # Ensure proper insertion order
        if crossing_vals is not None and lcl_vals is not None:
            if crossing_vals[3] > lcl_vals[3]:
                add_to_arrays(*crossing_vals)
                add_to_arrays(*lcl_vals)
            else:
                add_to_arrays(*lcl_vals)
                add_to_arrays(*crossing_vals)
        elif crossing_vals is not None:
            add_to_arrays(*crossing_vals)
        elif lcl_vals is not None:
            add_to_arrays(*lcl_vals)
        
        add_to_arrays(env_vt, pcl_vt, height, p)
        
        old_p = p
        old_pcl_t = pcl_vt
        old_env_t = env_vt
        old_h = height
    
    to_ret_hgt = tuple(to_ret_hgt)
    to_ret_pres = tuple(to_ret_pres)
    to_ret_pcl_virt_t = tuple(to_ret_pcl_virt_t)
    to_ret_env_virt_t = tuple(to_ret_env_virt_t)
    
    # Quality control, did we do this correctly
    assert len(to_ret_hgt) == len(to_ret_pres)
    
    for i in range(1, len(to_ret_pres)):
        assert to_ret_pres[i - 1] > to_ret_pres[i] and \
            to_ret_hgt[i - 1] < to_ret_hgt[i],         \
            "order error: %.1f > %.1f and %.2f < %.2f" % \
            (to_ret_pres[i - 1],
             to_ret_pres[i], to_ret_hgt[i - 1], to_ret_hgt[i])
    
    return ParcelProfile(
        to_ret_env_virt_t, to_ret_pcl_virt_t, to_ret_pres, to_ret_hgt, lcl_height, parcel
    )


def analyze_parcel_ascent(parcel_profile):
    """Analyze a parcel profile for important values.

    Returns a ParcelAscentAnalysis
    """
    el_hgt, top_hgt, max_int_buoyancy, lmib, cloud = _analyze_parcel_ascent_inner(parcel_profile)
    
    dry_profile = _dry_parcel_profile(parcel_profile)
    dry_parcel_profile = parcel_profile._replace(parcel_virt_temp=dry_profile)
    
    _, _, dry_max_int_buoyancy, *_ = _analyze_parcel_ascent_inner(
        dry_parcel_profile, stop_at_el=True
    )
    
    # Rounding and small errors sometimes cause the dry integrated
    # buoyancy to be slightly higher than the maximum integrated buoyancy,
    # so we just take the minimum.
    return ParcelAscentAnalysis(
        parcel_profile.lcl_height, el_hgt, top_hgt, max_int_buoyancy, lmib,
        min(dry_max_int_buoyancy, max_int_buoyancy), cloud
    )


def _analyze_parcel_ascent_inner(parcel_profile, stop_at_el=False):
    """Helper function

    Returns (el_hgt, top_hgt, max_int_buoyancy, level_max_int_buoyancy,
        cloud_depth)
    """
    
    def convert_to_buoyancy_like_value(x):
        p, h, pt, et = x
        et = wxf.theta_kelvin(p, et)
        pt = wxf.theta_kelvin(p, pt)
        buoyancy_val = (pt - et) / et
        return (buoyancy_val, h)
    
    ascent_iter = zip(
        parcel_profile.pres, parcel_profile.hgt, parcel_profile.parcel_virt_temp,
        parcel_profile.env_virt_temp
    )
    
    ascent_vals = tuple(convert_to_buoyancy_like_value(x) for x in ascent_iter)
    
    # Build iterator to iterate over two values at a time sow we can integrate
    # with the trapezoid rule
    steps = zip(ascent_vals, ascent_vals[1:])
    
    el_hgt = None
    top_hgt = None
    int_buoyancy0 = 0.0
    int_buoyancy = 0.0
    max_int_buoyancy = 0.0
    level_max_int_buoyancy = 0.0
    cloud_depth = None
    
    for v0, v1 in steps:
        b0, h0 = v0
        b1, h1 = v1
        
        dz = h1 - h0
        assert dz >= 0.0, "%.2f,%.2f" % (h0, h1)
        
        int_buoyancy += (b0 + b1) * dz
        if int_buoyancy >= max_int_buoyancy:
            level_max_int_buoyancy = h1
            max_int_buoyancy = int_buoyancy
        
        if b0 > 0.0 and b1 <= 0.0:
            el_hgt = np.interp(0.0, (b1, b0), (h1, h0))
            if stop_at_el:
                break
        
        if int_buoyancy0 >= 0.0 and int_buoyancy < 0.0:
            top_hgt = np.interp(0.0, (int_buoyancy, int_buoyancy0), (h1, h0))
            break
        
        int_buoyancy0 = int_buoyancy
    
    if parcel_profile.lcl_height is not None:
        if top_hgt is None:
            cloud_depth = h0 - parcel_profile.lcl_height
        else:
            cloud_depth = top_hgt - parcel_profile.lcl_height
    
    max_int_buoyancy *= wxf.g / 2.0
    
    return (el_hgt, top_hgt, max_int_buoyancy, level_max_int_buoyancy, cloud_depth)


def _dry_parcel_profile(parcel_profile):
    """A dry adiabat starting at the parcel profiles surface.

    Given a parcel profile, generate a parallel array that has a version
    of that profile where condensation never occurred, so it was lifted
    dry adiabatically the whole way up. This isn't exactly a dry
    adiabat since we are interested in virtual temperatures for
    buoyancy calculations, we also have to take the remaining moisture
    in the parcel into consideration when calculating the virtual
    temperature.

    Returns a tuple parallel to the levels in the parcel_profile with
    virtual temperatures.
    """
    
    parcel = parcel_profile.parcel
    
    p_lcl, _ = wxf.press_and_temp_k_at_lcl(parcel.temperature, parcel.dew_point, parcel.pressure)
    
    parcel_theta = wxf.theta_kelvin(parcel.pressure, parcel.temperature)
    dry_parcel_sh = wxf.specific_humidity(parcel.dew_point, parcel.pressure)
    
    def calc_parcel_vt(press):
        if press is None:
            return None
        try:
            parcel_t = wxf.temperature_c_from_theta(parcel_theta, press)
            if press > p_lcl:
                parcel_dp = wxf.dew_point_from_p_and_specific_humidity(press, dry_parcel_sh)
            else:
                parcel_dp = parcel_t
        except Exception:
            return None
        return wxf.virtual_temperature_c(parcel_t, parcel_dp, press)
    
    profile_iter = zip(parcel_profile.pres, parcel_profile.env_virt_temp, parcel_profile.hgt)
    
    to_ret_pcl_virt_t = []
    
    for p, env_vt, height in profile_iter:
        pcl_vt = calc_parcel_vt(p)
        if pcl_vt is None:
            break
        
        to_ret_pcl_virt_t.append(pcl_vt)
    
    to_ret_pcl_virt_t = tuple(to_ret_pcl_virt_t)
    
    return to_ret_pcl_virt_t


def mixed_layer_parcel(sounding):
    """Get a 100 hPa surface based mixed layer parcel."""
    sfc_press = sounding.profile.pressure[0]
    if sfc_press is None:
        return None
    
    top_press = sfc_press - 100.0
    
    sum_theta = 0.0
    sum_sh = 0.0
    count = 0
    
    dp = sounding.surface.dewpoint
    t = sounding.surface.temp
    if dp is not None and t is not None:
        count += 1
        sum_sh += wxf.specific_humidity(dp, sfc_press)
        sum_theta += wxf.theta_kelvin(sfc_press, t)
    
    iter = zip(sounding.profile.pressure, sounding.profile.temp, sounding.profile.dewpoint)
    iter = filter(lambda x: x[0] is not None and x[1] is not None and x[2] is not None, iter)
    iter = takewhile(lambda x: x[0] >= top_press, iter)
    
    for p, t, dp in iter:
        count += 1
        sum_sh += wxf.specific_humidity(dp, p)
        sum_theta += wxf.theta_kelvin(p, t)
    
    ave_sh = sum_sh / count
    ave_theta = sum_theta / count
    
    t = wxf.temperature_c_from_theta(ave_theta, sfc_press)
    dp = wxf.dew_point_from_p_and_specific_humidity(sfc_press, ave_sh)
    
    return Parcel(t, dp, sfc_press)


def heated_parcel(starting_parcel, heating, moisture_ratio):
    """Simulate a fire heated (and potentially moistened) parcel.

    starting_parcel is the parcel we want to heat up.
    heating is the change in temperature of the parcel in K or C (they're
        the same).
    moisture_ratio how much fire moisture to add, which depends on the
        heating. A value of 10.0 means add 1 g/kg of moisture for every
        10K of heating. None implies adding no moisture at all.

    Returns a Parcel.
    """
    assert isinstance(starting_parcel, Parcel)
    assert heating is not None
    
    new_t = starting_parcel.temperature + heating
    
    if moisture_ratio is None:
        new_dp = starting_parcel.dew_point
    else:
        sh = wxf.specific_humidity(starting_parcel.dew_point, starting_parcel.pressure)
        dsh = heating / moisture_ratio / 1000
        new_dp = wxf.dew_point_from_p_and_specific_humidity(starting_parcel.pressure, sh + dsh)
    
    return starting_parcel._replace(temperature=new_t, dew_point=new_dp)


def blow_up_analysis(sounding, moisture_ratio):
    """Perform a fire plume blow up analysis on a sounding.

    moisture_ratio is the moisture_ratio to apply when creating a
    heated_parcel from the above so named function. If it is None that
    implies no fire moisture should be added to the parcels.

    Start with a mixed layer parcel and apply heating to it in 0.1C
    increments up until a maximum heating of 20°C has been applied.

    The blow up temperatures are calculated by using numerical
    derivatives to find where the rate of change is at a maximum.

    Returns BlowUpAnalysis.
    """
    MIN_DT = 0.0
    MAX_DT = 20.0
    DT_STEP = 0.1
    
    pcl0 = mixed_layer_parcel(sounding)
    dts = np.arange(MIN_DT, MAX_DT, DT_STEP)
    
    pcls = (heated_parcel(pcl0, dt, moisture_ratio) for dt in dts)
    profiles = (lift_parcel(p, sounding) for p in pcls)
    anals = tuple(analyze_parcel_ascent(pp) for pp in profiles)
    
    _lcls, _els, _tops, _byncy, lmibs, _dry_byncy, cld_dpts = zip(*anals)
    
    def clean_up_pairs(dt_vals, tgt_vals):
        """Match dts with their vals, removing any None values."""
        pairs = tuple(x for x in zip(dt_vals, tgt_vals) if all(x))
        if len(pairs) == 0:
            return None
        
        return zip(*pairs)
    
    def find_blow_up_dt(dt_vals, tgt_vals):
        data = clean_up_pairs(dt_vals, tgt_vals)
        if data is None:
            return MAX_DT
        
        tgt_dts, tgt_vals = data
        tgt_grads = np.gradient(tgt_vals, tgt_dts)
        max_idx = np.argmax(tgt_grads)
        low_idx = max(0, max_idx - int(0.5 / DT_STEP))
        high_idx = min(len(tgt_dts) - 1, max_idx + int(0.5 / DT_STEP))
        
        return (tgt_dts[max_idx], tgt_vals[high_idx] - tgt_vals[low_idx])
    
    dt_lmib_blow_up, dz_lmib_blow_up = find_blow_up_dt(dts, lmibs)
    
    data = clean_up_pairs(dts, cld_dpts)
    if data is None:
        dt_cloud = MAX_DT
    else:
        cld_dts, cld_vals = data
        cld_idx = min(range(len(cld_dts)), key=lambda i: abs(cld_vals[i]))
        dt_cloud = cld_dts[cld_idx]
    
    return BlowUpAnalysis(dt_cloud, dt_lmib_blow_up, dz_lmib_blow_up)


def hdw(sounding):
    bottom = sounding.profile.elevation
    top = bottom + 500.0
    
    vals = zip(
        sounding.profile.hgt, sounding.profile.temp, sounding.profile.dewpoint,
        sounding.profile.windSpd
    )
    
    vals = tuple(takewhile(lambda x: x[0] <= top, vals))
    
    vpds = ((x[1], x[2]) for x in vals if x[1] is not None and x[2] is not None)
    vpds = ((vapor_pressure_liquid_water(x[0]) - vapor_pressure_liquid_water(x[1])) for x in vpds)
    max_vpd = max(vpds)
    
    max_wspd = max(x[3] for x in vals if x[3] is not None)
    max_wspd = knots_to_mps(max_wspd)
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
        print("\n\n\nWorking on model initial time %s" % test_data[0].profile.time)
        
        for snd in test_data:
            vt = snd.profile.time
            
            # Create a mixed layer parcel
            pcl = mixed_layer_parcel(snd)
            pcl = pcl._replace(temperature=pcl.temperature + 10)
            parcel_profile = lift_parcel(pcl, snd)
            res = analyze_parcel_ascent(parcel_profile)
            
            if res.percent_wet_cape() is None:
                wet_pct = "None"
            else:
                wet_pct = "%3.0f%%" % (res.percent_wet_cape() * 100,)
            
            print("%s, Pct Wet CAPE - %s" % (res, wet_pct))
        
        for snd in test_data:
            print(blow_up_analysis(snd, None))