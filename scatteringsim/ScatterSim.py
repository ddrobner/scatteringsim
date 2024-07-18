from scatteringsim.helpers.stoppingpower import read_stopping_power
from scatteringsim.helpers.alphapath import gen_alpha_path
from scatteringsim.helpers.energytransfer import energy_transfer
from scatteringsim.structures import *

from numpy import pi
from scipy.constants import Avogadro
from scipy.interpolate import LinearNDInterpolator
from random import uniform, random

import copy
import numpy as np
import pandas as pd

class ScatterSim:
    def __init__(self, e_0: float, stepsize: float, stp_fname: str, cx_fname: str):
        self.stp = read_stopping_power(stp_fname)
        self.e_0 = e_0
        self.stepsize = stepsize
        self.alpha_path = gen_alpha_path(self.e_0, self.stepsize)
        self.density = 0.8562 
        self.mol_wt = 246.43

        self.cx = pd.read_csv(stp_fname)
        # converting to radians
        self.cx['theta'] = np.deg2rad(self.cx['theta'])
        # scaling the cross section accordingly
        self.cx['cx'] = 180.0/np.pi * self.cx['cx']
        xy = self.cx[['energy', 'theta']].to_numpy()
        z = self.cx['cx'].to_numpy()
        self.cx_interpolator = LinearNDInterpolator(xy, z)
        # set up a lookup table for the riemann sums
        temp_cx = {}
        for e in self.cx['energy'].unique():
            angles = self.cx[self.cx['energy'] == e]['theta']
            dcx = self.cx[self.cx['energy'] == e]['cx']
            itg = np.trapz(dcx, angles)
            temp_cx[e] = itg
        
        self.total_cx = pd.DataFrame(temp_cx, columns=['Energy', 'Total'])

        self.epsilon = 0.01


    def differential_cx(self, theta, ke, scaled=False):
        cx_pt = float(self.cx_interpolator((ke, theta)))
        if scaled == True:
            scale = self.cx_interpolator([ke, i] for i in np.linspace(0, np.pi, 10)).max()
            return (1/scale)*cx_pt 
        return cx_pt

    def total_crossection(self, ke):
        return float(np.interp(ke, self.total_cx['Energy'].to_numpy(), self.total_cx['Total'].to_numpy()))

    # moving this to a class method to avoid all of this passing variables
    # around nonsense
    def scattering_angle(self, ke) -> np.float64:
        theta_min = 0.1
        theta_max = pi
        while True:
            # first we sample from a uniform distribution of valid x-values
            xsample = uniform(theta_min, theta_max)
            # then find the scaled differential crosssection at the x-sample
            scx = self.differential_cx(xsample, ke, scaled=True)
            # and then return the x-sample if a random number is less than that value
            if random() < scx:
                return xsample 


    def scattering_probability(self, ke) -> np.float64:
        sample_dim = 1
        sigma = self.total_crossection(ke)*1E-24

        # now let's do the same stuff as before to determine the probability
        rho = self.density # g/cm^3, see above
        n = Avogadro/(self.mol_wt) * self.stepsize * rho

        eff_a = sigma*n
        total_a = sample_dim**2
        return eff_a/total_a

    def scatter_sim(self, e_0: float, alpha_path : list) -> AlphaEvent:
        # TODO add ability to get scattering angles out
        # we can do the binning/etc later

        # let's set both quenching factors to 1 in here so we can tune them later
        a_path = copy.deepcopy(alpha_path)
        proton_event_path = []
        e_i = e_0
        # The alpha energy path is completely deterministic, so to speed things up
        # let's pre-bake it. Then, for each step we can do the scattering stuff

        # great, so now we have our pre-baked alpha energy
        # now we can iterate over the list and do the proton scattering
        scattered = False
        scatter_e = []
        alpha_out = []
        for s in range(len(a_path)):
            #rsum = diffcx_riemann_sum(a_path[s], tck, theta_max=pi/2)
            if self.scattering_probability(a_path[s]) > random.random():
                scattered = True
                scatter_angle = self.scattering_angle(a_path[s])
                transfer_e = energy_transfer(a_path[s], scatter_angle)
                a_path[s] = transfer_e.e_alpha
                #a_path = a_path[0:s-1]
                alpha_out.append(a_path[0:s-1])
                proton_event_path.append(transfer_e.e_proton)
                scatter_e.append(ScatterFrame(scatter_angle, transfer_e.e_proton, scatter_angle))
                alpha_out.append(gen_alpha_path(transfer_e.e_alpha, self.stp, stepsize=self.stepsize, epsilon=self.epsilon))
                break
        if not scattered:
            alpha_out.append(a_path)

        return AlphaEvent(alpha_out, proton_event_path, scatter_e)