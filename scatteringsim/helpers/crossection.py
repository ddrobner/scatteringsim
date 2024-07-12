import numpy as np
import pandas as pd

from random import uniform, random
from scipy.constants import Avogadro
from scipy.interpolate import bisplrep, bisplev

# the idea here is that we compute the interpolation initially
def compute_interp_cx(fname : str) -> np.float64:
    cx = pd.read_csv(fname)
    energy = np.array(cx['energy'])
    theta = np.array(cx['theta'])
    diffcx = np.array( cx['energy'] )
    return bisplrep(energy, theta, diffcx)

def diff_cx(theta, ke, tck) -> np.float64:
    return bisplev(ke, theta, tck)

def diffcx_riemann_sum(ke, tck, meshsize=0.01, theta_min=0.1, theta_max=1) -> np.float64:
    x_points = np.arange(theta_min, theta_max, meshsize)
    y_points = [diff_cx(i, ke, tck) for i in x_points]

    return np.trapz(y_points, x_points, meshsize)

def scattering_probability(ke, dx, r_sum, density=0.8562, mol_wt=246.43) -> np.float64:
    sample_dim = 1
    sigma = r_sum*1E-24

    # now let's do the same stuff as before to determine the probability
    rho = density # g/cm^3, see above
    n = Avogadro/(mol_wt) * dx * rho

    eff_a = sigma*n
    total_a = sample_dim**2
    return eff_a/total_a
    
def scaled_diff_cx(theta, ke, tck) -> np.float64:
    # luckily, our differential crossection s a decreasing function on the interval we care about
    # so, we just take the left endpoint as our x-value
    theta_min = 0.1
    scale = 1/(diff_cx(theta_min, ke, tck))
    return diff_cx(theta,ke, tck)*scale

def scattering_angle(ke, tck) -> np.float64:
    theta_min = 0.1
    while True:
        # first we sample from a uniform distribution of valid x-values
        xsample = uniform(theta_min, 1)
        # then find the scaled differential crosssection at the x-sample
        scx = scaled_diff_cx(xsample, ke, tck)
        # and then return the x-sample if a random number is less than that value
        if random() < scx:
            return xsample