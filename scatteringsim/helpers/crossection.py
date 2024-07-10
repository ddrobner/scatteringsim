import numpy as np
import pandas as pd

from random import uniform, random
from scipy.constants import Avogadro

def diff_cx(theta, ke, fname) -> np.float64:
    exp_cx = pd.read_csv(fname)
    theta = np.rad2deg(theta)
    for k in exp_cx.index:
        if theta <= exp_cx['theta'][0]:
            # for low scattering angles which we don't have better data for 
            # (ie. below 12 degrees )
            # nvm 1/x makes a lot more sense here lol
            return 2*np.pi*((1/theta)) + 58.6308 
        elif theta <= exp_cx["theta"][k+1] and theta >= exp_cx['theta'][k]:
            return 2*np.pi*np.interp(theta, list(exp_cx['theta']), list(exp_cx['sigma']))
        elif theta >= list(exp_cx['theta'])[-1]:
            return 2*np.pi*((list(exp_cx['sigma'])[-1] - list(exp_cx['sigma'])[-2])/(list(exp_cx['theta'])[-1] - list(exp_cx['theta'])[-2]))*theta

def diffcx_riemann_sum(fname : str, meshsize=0.01, theta_min=0.1, ke=8) -> np.float64:
    x_points = np.arange(theta_min, 1, meshsize)
    y_points = [diff_cx(i, ke, fname) for i in x_points]
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
    
def scaled_diff_cx(theta, ke, fname) -> np.float64:
    # luckily, our differential crossection s a decreasing function on the interval we care about
    # so, we just take the left endpoint as our x-value
    theta_min = 0.1
    scale = 1/(diff_cx(theta_min, ke, fname))
    return diff_cx(theta,ke)*scale

def scattering_angle(ke, fname) -> np.float64:
    theta_min = 0.1
    while True:
        # first we sample from a uniform distribution of valid x-values
        xsample = uniform(theta_min, 1)
        # then find the scaled differential crosssection at the x-sample
        scx = scaled_diff_cx(xsample, ke, fname)
        # and then return the x-sample if a random number is less than that value
        if random() < scx:
            return xsample