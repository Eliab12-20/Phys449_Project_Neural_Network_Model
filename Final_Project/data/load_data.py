import numpy as np
import pandas as pd
import os
from pprint import pprint
from astropy.units import earthMass, jupiterMass, earthRad, jupiterRad, solMass, \
    solRad, AU
import astro_eq as ae
from uncertainties import ufloat


# TODO: Potential bug
def jupiter_to_earth(dataset, feature):
    assert feature in ['mass', 'mass_error_max', 'mass_error_min', 'radius',
                       'radius_error_max', 'radius_error_min']
    if 'radius' in feature:
        df = dataset[feature].apply(
            lambda x: (x * jupiterRad).to('earthRad').value)
    elif 'mass' in feature:
        df = dataset[feature].apply(
            lambda x: (x * jupiterMass).to('earthMass').value)
    new_df = pd.DataFrame({feature: df})
    dataset.update(new_df)
    return dataset


# TODO: Potential bugs
def add_temp_eq_dataset(dataset):
    semi_major_axis = dataset.semi_major_axis * AU.to('solRad')
    teq_planet = [ae.plant_tem(teff, a / rad, ecc)
                  for teff, a, rad, ecc,
                  in zip(dataset.star_teff, semi_major_axis,
                         dataset.star_radius, dataset.eccentricity)]
    dataset.insert(2, 'temp_eq', teq_planet)
    return dataset


def add_star_luminosity_dataset(dataset):
    """Compute the stellar luminosity
    L_star/L_sun = (R_star/R_sun)**2 * (Teff_star / Teff_sun)**4
    Radius star is already expressed in Sun radii in the dataset
    lum_sun    = 3.828 * 10**26   # Watt
    radius_sun = 6.95508 * 10**8  # meters"""
    Teff_sun = 5777.0  # Kelvin
    L_star = [R_star ** 2 * (Teff_star / Teff_sun) ** 4
              for R_star, Teff_star
              in zip(dataset.star_radius, dataset.star_teff)]
    dataset.insert(2, 'star_luminosity', L_star)
    return dataset


# TODO: PHYS
def add_temp_eq_error_dataset(dataset):
    semi_major_axis = dataset.semi_major_axis * AU.to('solRad')
    semi_major_axis_error = dataset.semi_major_axis_error * AU.to('solRad')
    teq_planet = [ae.getTeqpl_error(ufloat(teff, abs(teff_e)),
                                 ufloat(a, abs(a_e))/ufloat(rad, abs(rad_e)),
                                 ufloat(ecc, abs(ecc_e)))
                  for teff, teff_e, a, a_e, rad, rad_e, ecc, ecc_e
                  in zip(dataset.star_teff, dataset.star_teff_error,
                         semi_major_axis, semi_major_axis_error,
                         dataset.star_radius, dataset.star_radius_error,
                         dataset.eccentricity, dataset.eccentricity_error)]
    teq_planet_value = [teq.nominal_value for teq in teq_planet]
    teq_planet_error = [teq.s for teq in teq_planet]
    dataset.insert(2, 'temp_eq_error', teq_planet_error)
    dataset.insert(2, 'temp_eq', teq_planet_value)
    return dataset


# TODO: PHYS
def add_star_luminosity_error_dataset(dataset):
    """Compute the stellar luminosity
    L_star/L_sun = (R_star/R_sun)**2 * (Teff_star / Teff_sun)**4
    Radius star is already expressed in Sun radii in the dataset
    lum_sun    = 3.828 * 10**26   # Watt
    radius_sun = 6.95508 * 10**8  # meters"""
    Teff_sun = 5778                 # Kelvin
    L_star = [ufloat(R_star, abs(R_star_error))**2 *
              (ufloat(Teff_star, abs(Teff_star_error)) / Teff_sun)**4
              for R_star, R_star_error, Teff_star, Teff_star_error
              in zip(dataset.star_radius, dataset.star_radius_error,
                     dataset.star_teff, dataset.star_teff_error)]
    L_star_value = [ls.nominal_value for ls in L_star]
    L_star_error = [ls.s for ls in L_star]
    dataset.insert(2, 'star_luminosity_error', L_star_error)
    dataset.insert(2, 'star_luminosity', L_star_value)
    return dataset


def load_dataset(exo_path, solar_path, features, solar=True, add_feature=True):
    """
    param: exo_path: exoplanet data file path
    param: solar_path: solar planet data file path
    param: features: ['mass', 'semi_major_axis','eccentricity', 'star_metallicity',
                        'star_radius', 'star_teff','star_mass', 'radius']
    """
    # load exoplanets
    dataset_exo = pd.read_csv(exo_path, index_col=0)
    dataset_exo = dataset_exo[features]

    # Removes the planets with NaN values,
    dataset_exo = dataset_exo.dropna(axis=0, how='any')

    # load solar planets
    dataset_solar = pd.read_csv(solar_path, index_col=0)
    dataset_solar = dataset_solar[features]
    # Removes the planets with NaN values,
    dataset_solar = dataset_solar.dropna(axis=0, how='any')

    # convert jupiter to earth (mass and radii)
    dataset_exo = jupiter_to_earth(dataset_exo, 'mass')
    dataset_exo = jupiter_to_earth(dataset_exo, 'radius')

    # data correction
    dataset_exo.loc['Kepler-10 c'].mass = 17.2

    # add solar system planet into exoplanet dataset
    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar], axis=0)
    else:
        dataset = dataset_exo

    if add_feature:
        dataset = add_temp_eq_dataset(dataset)
        dataset = add_star_luminosity_dataset(dataset)

    return dataset


def load_dataset_error(exo_path, solar_path, features,  solar=True):
    """
    param: exo_path: exoplanet data file path
    param: solar_path: solar planet data file path
    param: features: ['mass', 'semi_major_axis','eccentricity',
                        'star_radius', 'star_teff','star_mass', 'radius'],
                        Note: here is diff with load_dataset, no star_metallicity
    """
    mass_and_radii_features = ['mass', 'mass_error_max', 'mass_error_min',
                               'radius', 'radius_error_max', 'radius_error_min']
    features_error_range_list = [f for feature in features for f in (feature,
                                                                     f"{feature}_error_min",
                                                                     f"{feature}_error_max")]
    features_error_range_dir = {feature: [f"{feature}_error_min",
                                          f"{feature}_error_max"] for feature in
                                features}
    features_error_list = [f for feature in features for
                           f in (feature, f"{feature}_error")]
    features_error_dir = {feature: [f"{feature}_error"] for
                          feature in features}

    # load exoplanets
    dataset_exo = pd.read_csv(exo_path, index_col=0)
    dataset_exo = dataset_exo[features_error_range_list]
    dataset_exo = dataset_exo.dropna(subset=features)
    # replace inf
    dataset_exo = dataset_exo.replace([np.inf, -np.inf], np.nan)

    # TODO: PHYS, replace Nan by a value???
    # TODO: PHYS, Read the explain in paper
    for key in features_error_range_dir:
        for feature in features_error_range_dir[key]:
            max_error = dataset_exo[feature].quantile(0.9)
            dataset_exo[feature] = dataset_exo[feature].replace(np.nan,
                                                                max_error)

    # convert jupiter to earth
    for mr_feature in mass_and_radii_features:
        dataset_exo = jupiter_to_earth(dataset_exo, mr_feature)

    for key in features_error_range_dir:
        dataset_exo[features_error_dir[key][0]] = dataset_exo[
            features_error_range_dir[key]].mean(axis=1)

    dataset_exo = dataset_exo[features_error_list]

    # TODO: PHYS. change mass of Kepler 10c
    dataset_exo.loc['Kepler-10 c'].mass = 17.2
    dataset_exo.loc['Kepler-10 c'].mass_error = 1.9

    # Solar system data
    dataset_solar = pd.read_csv(solar_path, index_col=0)
    dataset_solar = dataset_solar[features_error_list]
    dataset_solar_system = dataset_solar.dropna(subset=features)

    if solar:
        dataset = pd.concat([dataset_exo, dataset_solar], axis=0)
    else:
        dataset = dataset_exo

    dataset = add_temp_eq_error_dataset(dataset)
    dataset = add_star_luminosity_error_dataset(dataset)

    return dataset


def load_data_rv():
    # TODO: Implement
    pass