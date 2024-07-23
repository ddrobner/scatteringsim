from scatteringsim.helpers.stoppingpower import read_stopping_power
from scatteringsim.helpers.alphapath import gen_alpha_path
from scatteringsim.helpers.energytransfer import energy_transfer
from scatteringsim.structures import *

from numpy import pi
from scipy.constants import Avogadro
from scipy.interpolate import LinearNDInterpolator
import random
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory

from typing import Type

import copy
import numpy as np
import pandas as pd

class ScatterSim:
    def __init__(self, e_0: float, num_alphas: int, stepsize: float, nhit: int, stp_fname: str, cx_fname: str, proton_factor: float = 0.5):
        self.stp = read_stopping_power(stp_fname)
        self.e_0 = e_0
        self.num_alphas = num_alphas
        self.proton_factor = proton_factor
        self.stepsize = stepsize
        self.nhit = nhit

        #picking a min theta so we neglect the small amounts of transferred energy
        self.theta_min = np.pi/4
        self.theta_max = 2*np.pi

        # leaving these as constants in here since we are unlikely to change them
        self.density = 0.8562 
        self.mol_wt = 246.43

        self.cx = pd.read_csv(cx_fname)
        # converting to radians
        self.cx['theta'] = np.deg2rad(self.cx['theta'])
        # scaling the cross section accordingly
        """
        xy = self.cx[['energy', 'theta']].to_numpy()
        z = self.cx['cx'].to_numpy()
        self.cx_interpolator = LinearNDInterpolator(xy, z)
        # set up a lookup table for the riemann sums
        temp_es = []
        temp_cx = []
        for e in self.cx['energy'].unique():
            angles = self.cx[self.cx['energy'] == e]['theta']
            dcx = self.cx[self.cx['energy'] == e]['cx']
            itg = np.trapz(dcx, angles)
            if(itg != 0):
                temp_es.append(e)
                temp_cx.append(itg)
        self.total_cx = pd.DataFrame(zip(temp_es, temp_cx), columns=['Energy',
        'Total'])
        del temp_es
        del temp_cx
        """
        
        self.epsilon = 0.1

        self._alpha_sim = None
        self._quenched_spec = None
        self._result = None

    @property
    def alpha_sim(self):
        return self._alpha_sim
    
    @property
    def quenched_spectrum(self):
        return self._quenched_spec

    @property
    def result(self):
        return self._result

    """
    def differential_cx(self, theta, ke, scaled=False):
        cx_pt = float(self.cx_interpolator((ke, theta)))
        if scaled == True:
            scale = self.cx_interpolator([(ke, i) for i in np.linspace(self.theta_min, self.theta_max, 10)]).max()
            return (1/scale)*cx_pt 
        return cx_pt
    """

    def differential_cx(self, theta, ke, scaled=False):
        theta_s = self.cx['theta'].to_numpy()
        cx_s = self.cx['cx'].to_numpy()
        cx_pt = np.interp(theta, theta_s, cx_s)
        if scaled:
            scale = np.interp(self.theta_min, theta_s, cx_s)
            return cx_pt/scale
        return cx_pt

    def total_crossection(self, ke):
        #return np.interp(ke, self.total_cx['Energy'].to_numpy(),
        #self.total_cx['Total'].to_numpy())
        return np.trapz([i*180/np.pi for i in self.cx['cx'].to_numpy()], self.cx['theta'].to_numpy())

    # moving this to a class method to avoid all of this passing variables
    # around nonsense
    def scattering_angle(self, ke) -> np.float64:
        theta_min = self.theta_min 
        theta_max = self.theta_max 
        while True:
            # first we sample from a uniform distribution of valid x-values
            xsample = random.uniform(theta_min, theta_max)
            # then find the scaled differential crosssection at the x-sample
            scx = self.differential_cx(xsample, ke, scaled=True)
            # and then return the x-sample if a random number is less than that value
            if random.random() < scx:
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

    def scatter_sim(self, alpha_path : list) -> AlphaEvent:
        #a_path = copy.deepcopy(alpha_path)
        proton_event_path = []
        scattered = False
        scatter_e = []
        alpha_out = []
        for s in range(len(alpha_path)):
            #rsum = diffcx_riemann_sum(alpha_path[s], tck, theta_max=pi/2)
            if self.scattering_probability(alpha_path[s]) > random.random():
                scattered = True
                scatter_angle = self.scattering_angle(alpha_path[s])
                transfer_e = energy_transfer(alpha_path[s], scatter_angle)
                print(f"Scattered: {round(scatter_angle, 4)}rad {transfer_e.e_proton}MeV p+")
                #alpha_path[s] = transfer_e.e_alpha
                #alpha_path = a_path[0:s-1]
                alpha_out.append(alpha_path[0:s-1])
                proton_event_path.append(transfer_e.e_proton)
                scatter_e.append(ScatterFrame(transfer_e.e_alpha, transfer_e.e_proton, scatter_angle))
                alpha_out.append(gen_alpha_path(transfer_e.e_alpha, self.stp, stepsize=self.stepsize, epsilon=self.epsilon))
                break
        if not scattered:
            alpha_out.append(alpha_path)

        return AlphaEvent(alpha_out, proton_event_path, scatter_e)

    def quenched_spectrum(self, sim_data: AlphaEvent,  proton_factor: float, alpha_factor: float=0.1) -> list[np.float64]:
        q_spec = []
        a_diffs = []
        n_boundaries = 0
        i = 0
        for ap in sim_data.alpha_path:
            a = 1
            if(len(sim_data.proton_scatters) > n_boundaries and i != 0):
                a_diffs.append(abs((sim_data.alpha_path[i-1][-1] + sim_data.proton_scatters[n_boundaries]) - ap[0]))
                n_boundaries += 1
            while a < len(ap):
                a_diffs.append(abs(ap[a] - ap[a-1]))
                a += 1
            i += 1
        q_spec.append( sum( [alpha_factor*j for j in a_diffs] + [proton_factor*k for k in sim_data.proton_scatters] ) )
        return q_spec

    def particle_scint_sim(self, scatter_args: tuple, scatter_kwargs: dict, quench_args: tuple, quench_kwargs: dict) -> tuple[AlphaEvent, list[np.float64]]:
        """A function to run both particle and quenching sim in one shot

        Args:
            scatter_args (tuple): tuple of arguments to be passed to ScatterSim.scatter_sim 
            scatter_kwargs (dict): dict of kwargs for ScatterSim.scatter_sim 
            quench_args (tuple): tuple of args for ScatterSim.quenched_spectrum
            quench_kwargs (dict): dict of kwargs for ScatterSim.quenched_spectrum 

        Returns:
            tuple[AlphaEvent, list[np.float64]]: tuple of the list of alpha
            deposits, as well as the quenched result 
        """
        a_part = self.scatter_sim(*scatter_args, **scatter_kwargs)
        q_part = self.quenched_spectrum(a_part, *quench_args, **quench_kwargs)
        q_part = [l for ls in q_part for l in ls]
        return (a_part, q_part)
    
    def compute_smearing(self, e_i, nhit):
        return random.gauss(e_i*nhit, np.sqrt(e_i*nhit))/nhit

    def start(self):
        alpha_path = gen_alpha_path(self.e_0, self.stp, epsilon=self.epsilon, stepsize=self.stepsize)
        with Pool(cpu_count()) as p:
            # store the output of both in a temporary list of tuples 
            tmp_res = p.starmap(self.particle_scint_sim, [((alpha_path,), dict(), (self.proton_factor,), {"alpha_factor": 0.1}) for i in range(self.num_alphas)])
            # and unpack into two lists
            self._alpha_sim, self._quenched_spec = zip(*tmp_res)
            smeared_spectrum = p.starmap(self.compute_smearing, [(i, self.nhit) for i in self._quenched_spectrum])
            p.close()
            p.join()
        self._result = smeared_spectrum