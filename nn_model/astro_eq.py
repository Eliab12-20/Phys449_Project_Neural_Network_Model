import numpy as np
import pandas as pd
from uncertainties import umath as um


def plant_tem(Teffst, aR, ecc, f=1 / 4., A=0):
    """
    Parameters
    ----------
    Teffst : float
        The effective temperature of the star (T_eff_star), usually in Kelvin (K).

    f : float
        The heat redistribution factor for the planet's atmosphere. This factor accounts for
        how the heat is distributed across the planet's surface. A value of 1/4 suggests uniform
        redistribution (f).

    A : float
        The Bond albedo (A) representing the fraction of the total energy reflected by the planet.
        It ranges from 0 (no reflection) to 1 (total reflection).

    aR : float
        The ratio of the semi-major axis of the planet's orbit to the stellar radius (a/R_star),
        a dimensionless quantity that determines the distance factor for received stellar radiation.

    ecc : float
        The eccentricity of the planet's orbit (e). Values range from 0 (circular orbit) to
        nearly 1 (highly elliptical orbit), affecting the average distance to the star and
        the resulting stellar radiation received by the planet.
    """
    return Teffst * (f * (1 - A)) ** (1 / 4.) * np.sqrt(1 / aR) / (
                1 - ecc ** 2) ** (1 / 8.)


# TODO: don't understand
def getTeqpl_error(Teffst, aR, ecc, A=0, f=1 / 4.):
    """Return the planet equilibrium temperature.

    Relation adapted from equation 4 page 4 in http://www.mpia.de/homes/ppvi/chapter/madhusudhan.pdf
    and https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law
    and later updated to include the effect of excentricity on the average stellar planet distance
    according to equation 5 p 25 of Laughlin & Lissauer 2015arXiv150105685L (1501.05685)
    Plus Exoplanet atmospheres, physical processes, Sara Seager, p30 eq 3.9 for f contribution.

    :param float/np.ndarray Teffst: Effective temperature of the star
    :param float/np.ndarray aR: Ration of the planetary orbital semi-major axis over the stellar
        radius (without unit)
    :param float/np.ndarray A: Bond albedo (should be between 0 and 1)
    :param float/np.ndarray f: Redistribution factor. If 1/4 the energy is uniformly redistributed
        over the planetary surface. If f = 2/3, no redistribution at all, the atmosphere immediately
        reradiate whithout advection.
    :return float/np.ndarray Teqpl: Equilibrium temperature of the planet
    """
    return Teffst * (f * (1 - A)) ** (1 / 4.) * um.sqrt(1 / aR) / (
                1 - ecc ** 2) ** (1 / 8.)