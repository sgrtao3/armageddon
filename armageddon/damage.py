import numpy as np
import pandas as pd
import armageddon
import math

def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees) 
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------    

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii for the input damage levels

    Examples
    --------

    >>> import armageddon
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
                   'outcome': 'Airburst'}
    >>> armageddon.damage_zones(outcome, 52.79, -2.95, 135, pressures=[1e3, 3.5e3, 27e3, 43e3])
    """

    k1 = 3.14e11  # define two parameters to simply the formula
    k2 = 1.8e7
    Rp = 6371e3 
    r = outcome.get('burst_distance')
    zb = outcome.get('burst_altitude')
    ratio = r / Rp  # r/Rp
    t = 180 / np.pi  # the parameter to exchange Degree and Radian
    
    blat = float(np.arcsin(np.sin(lat / t) * np.cos(ratio) + np.cos(lat / t) * np.sin(ratio) * np.cos(bearing / t)) * t)
    delta_lon = np.arctan((np.sin(bearing / t) * np.sin(ratio) * np.cos(lat / t))
                          / (np.cos(ratio) - np.sin(lat / t) * np.sin(blat / t))) * t
    blon = float(lon + delta_lon)

    Ek = outcome.get('burst_energy') # The value of Ek has been calculated in calculate_energy function

    # Use Newton's down-hill method to solve nonlinear equation p(r)
    # Get the damrad value corresponding to each p in the pressures list
    def funx(x, p):
        return (k1 * ((x ** 2 + zb ** 2) / (Ek ** (2 / 3))) ** -1.3 +
                k2 * ((x ** 2 + zb ** 2) / (Ek ** (2 / 3))) ** (-0.565) - p)

    def dfunx(x):
        f = (-1.3 * k1 * ((x * 2) / (Ek ** (2 / 3))) * ((x ** 2 + zb ** 2) / (Ek ** (2 / 3))) ** -2.3 + 
                k2 * (-0.565) * ((x * 2) / (Ek ** (2 / 3))) * ((x ** 2 + zb ** 2) / (Ek ** (2 / 3))) ** (-1.565))
        return f

    #  the iterative formula of downhill_method is r(k+1)(lambda) = r(k) - lambda(p(rk)/dp(rk))
    def downhill_method(x, p, ep=1e-05, max_iter=100):
        for iter in range(max_iter):
            if (dfunx(x) == 0):
                return 0
            r = 1
            x1 = x - r * funx(x, p) / dfunx(x) 
            while (math.fabs(funx(x1, p)) > math.fabs(funx(x, p))):
                r = r / 2
                x1 = x - r * funx(x,p) / dfunx(x)
            x1 = x - r * funx(x, p) / dfunx(x)
            if(math.fabs(x1 - x) <= ep):
                return x1
            else:
                x = x1
        return 0

    damrad = []
    for p in pressures:
        result = downhill_method(1,p)
        damrad.append(float(abs(result)))
    return blat, blon, damrad


fiducial_means = {'radius': 10, 'angle': 20, 'strength': 1e6,
                  'density': 3000, 'velocity': 19e3,
                  'lat': 51.5, 'lon': 1.5, 'bearing': -45.}
fiducial_stdevs = {'radius': 1, 'angle': 1, 'strength': 5e5,
                   'density': 500, 'velocity': 1e3,
                   'lat': 0.025, 'lon': 0.025, 'bearing': 0.5}


def impact_risk(planet, means=fiducial_means, stdevs=fiducial_stdevs,
                pressure=27.e3, nsamples=10, sector=True):
    """
    Perform an uncertainty analysis to calculate the risk for each affected
    UK postcode or postcode sector

    Parameters
    ----------

    planet: armageddon.Planet instance
        The Planet instance from which to solve the atmospheric entry
    means: dict
        A dictionary of mean input values for the uncertainty analysis. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``
    stdevs: dict
        A dictionary of standard deviations for each input value. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``
    pressure: float
        The pressure at which to calculate the damage zone for each impact
    nsamples: int
        The number of iterations to perform in the uncertainty analysis
    sector: logical, optional
        If True (default) calculate the risk for postcode sectors, otherwise
        calculate the risk for postcodes

    Returns
    -------
    
    risk: DataFrame
        A pandas DataFrame with columns for postcode (or postcode sector) and
        the associated risk. These should be called ``postcode`` or ``sector``,
        and ``risk``.
    """

    parameters = np.zeros((nsamples, 8))
    index = 0
    # generator nsample parameters
    for key, value in fiducial_means.items():
        parameters[:, index] = np.random.normal(value, stdevs.get(key), nsamples)
        index += 1

    # if planet.solve_atmospheric_entry support numpy operation
    # we can eliminate this loop i.e.radius=parameter[:,0] to be more efficient
    result_dict = {}
    for parameter in parameters:
        # this part of code is same as example.py
        result = planet.solve_atmospheric_entry(radius=parameter[0], angle=parameter[1],
                                                strength=parameter[2], density=parameter[3],
                                                velocity=parameter[4])
        result = planet.calculate_energy(result)
        outcome = planet.analyse_outcome(result)
        blast_lat, blast_lon, damage_rad = armageddon.damage_zones(outcome,
                                                                   lat=parameter[5], lon=parameter[6],
                                                                   bearing=parameter[7],
                                                                   pressures=[1e3, 3.5e3, 27e3, 43e3])
        locator = armageddon.PostcodeLocator()
        postcodes = locator.get_postcodes_by_radius((blast_lat, blast_lon), radii=damage_rad, sector=sector)
        populations = locator.get_population_of_postcode(postcodes, sector=sector)

        # calculate risk
        damage_zones_level = 1
        for radi in damage_rad:
            cur_postcodes = postcodes[damage_zones_level - 1]
            cur_population = populations[damage_zones_level - 1]
            for i in range(len(cur_postcodes)):
                result_dict[cur_postcodes[i]] = (result_dict[cur_postcodes[i]] if cur_postcodes[i] in result_dict else 0) + \
                                                (damage_zones_level * cur_population[i])
            damage_zones_level += 1

    # transfer result to pandas DataFrame and divide risk by nsamples
    result_df = pd.DataFrame.from_dict(result_dict, orient='index')
    result_df.columns = ['risk']
    result_df['risk'] = result_df['risk'].apply(lambda x: x / nsamples)
    return result_df.sort_values('risk')

if __name__ == '__main__':
    earth = armageddon.Planet(atmos_func='exponential', atmos_filename=None,
                              Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                              g=9.81, H=8000., rho0=1.2)
    print(impact_risk(earth))