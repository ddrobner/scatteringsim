import cupy as cp
import numpy as np
import pandas as pd

import cupy.random as crandom
import multiprocessing as mp
import pickle

import random

from scatteringsim.utils import read_stopping_power, gen_alpha_path, energy_transfer
from scatteringsim.structures import ScatterFrame, AlphaEvent
from scipy.interpolate import LinearNDInterpolator, interp1d
from os.path import isfile

from scipy.constants import Avogadro


class GPUSim:
    def __init__(self, e_0: float, num_alphas: int, stepsize: float, nhit: int, stp_fname: str, cx_fname: str, total_cx_fname: str = "crossections/endf.csv", proton_factor: float = 0.5):
        # declaring this as a global to speed up multprocessing 
        #global alpha_path
        
        # declaring some constants 
        self.stp = read_stopping_power(stp_fname)
        self.e_0 = e_0
        self.num_alphas = num_alphas
        self.proton_factor = proton_factor
        self.stepsize = stepsize
        self.nhit = nhit

        self.alpha_factor = 0.1

        #picking a min theta so we neglect the small amounts of transferred energy
        self.theta_min = np.pi/4
        self.theta_max = np.pi

        # leaving these as constants in here since we are unlikely to change them
        self.density = 0.8562 
        self.mol_wt = 246.43

        # this is the lower bound any time anything goes to zero (ie. alpha
        # energy)
        # really speeds things up and also when it has that little energy the
        # scatters are going to be negligible
        self.epsilon = 0.1

        # declaring this as a class variable for now since I don't write to it
        # so it doesn't get copied on pickling

        # dump alpha path to disk if it doesn't exist and load it if it does
        alpha_path_fname = f"alpha_path_{str(e_0).replace(".","p")}"
        self.alpha_path = [] 
        if isfile(alpha_path_fname):
            with open(alpha_path_fname, 'rb') as f:
                self.alpha_path = pickle.load(f)
        else:
            self.alpha_path = gen_alpha_path(self.e_0, self.stp, epsilon=self.epsilon, stepsize=self.stepsize)
            with open(alpha_path_fname, 'wb') as f:
                pickle.dump(self.alpha_path, f)

        
        self.alpha_path = gen_alpha_path(self.e_0, self.stp, epsilon=self.epsilon, stepsize=self.stepsize)
        # can have no lock here since the array is read only
        #alpha_path = mparray(ctypes.c_double, apath_base)

        # now we handle setting up the cross section
        self.cx = pd.read_csv(cx_fname)
        # converting to radians
        # NOTE this means we need to scale the cx when integrating by 
        # 180/pi due to the transformation
        self.cx['theta'] = np.deg2rad(self.cx['theta'])
        self.cx = self.cx[self.cx['theta'] >= self.theta_min]
        self.cx.reset_index(inplace=True, drop=True)
        xy = self.cx[['energy', 'theta']].to_numpy()
        z = self.cx['cx'].to_numpy()
        self.cx_interpolator = LinearNDInterpolator(xy, z)
        # set up a lookup table for the riemann sums
        self.total_cx = pd.read_csv(total_cx_fname, names=['Energy', 'Total'])

        # this is only done once so can do it on the cpu
        self.alpha_steps = len(self.alpha_path)
        self.s_prob_lut = cp.array([self.scattering_probability(j) for j in self.alpha_path])

        self.alpha_path_gpu = cp.array(self.alpha_path)
        self.cx_inverse_dists = dict()
        for e in self.cx['energy'].unique():
            self.cx_inverse_dists[e] = self.gen_inverse_dist(e)

        # and set up class variable to store the outputs
        self._alpha_sim = []
        self._proton_sim = []
        self._quenched_spec = [] 
        self._result = []

    @property
    def alpha_sim(self):
        return self._alpha_sim
    
    @property
    def proton_sim(self):
        return self._proton_sim

    def pop_particle(self, idx: int) -> None:
        self._alpha_sim.pop(idx)
        self._proton_sim.pop(idx)

    def add_particle(self, alpha_val, proton_val) -> None:
        self._alpha_sim.append(alpha_val)
        self._proton_sim.append(proton_val)

    @property
    def quenched_spec(self):
        return self._quenched_spec

    @property
    def result(self):
        return self._result

    def gen_inverse_dist(self, ke):
        x = np.linspace(self.theta_min, self.theta_max, 100)
        y = self.cx_interpolator(ke, x)
        cdf_y = np.cumsum(y)
        cdf_y = cdf_y/cdf_y.max()
        inverse_cdf = interp1d(cdf_y, x)
        # this is a function
        return inverse_cdf

    def scattering_angle(self, ke) -> np.float32:
        # this might be a cheat but I think I'm going to just interpolate
        # between the inverse dist values for each KE
        dk = list(self.cx_inverse_dists.keys())
        dk.sort()
        
        i = 0
        j = 1
        while j < len(dk):
            if dk[i] < ke and dk[j] > ke:
                low_e = dk[i]
                low_interp = self.cx_inverse_dists[low_e]
                
                high_e = dk[j]
                high_interp = self.cx_inverse_dists[high_e]

                return np.interp(ke, [low_e, high_e], [low_interp(random.uniform(0, 1)), high_interp(random.uniform(0, 1))]) 
            i += 1
            j += 1
        return 0

    def total_crossection(self, ke : np.float64) -> np.float64:
        """Computes the total cross section with a trapezoidal riemann sum

        Args:
            ke (np.float64): The kinetic energy for the cross section 

        Returns:
            np.float64: The total cross section 
        """
        return 2*np.pi*np.interp(ke, self.total_cx['Energy'].to_numpy(), self.total_cx['Total'].to_numpy())
        #return np.trapz([i*(180/np.pi) for i in self.cx['cx'].to_numpy()], self.cx['theta'].to_numpy())

    
    def scattering_probability(self, ke) -> np.float64:
        sample_dim = 1
        sigma = self.total_crossection(ke)*1E-24

        # now let's do the same stuff as before to determine the probability
        rho = self.density # g/cm^3, see above
        n = Avogadro/(self.mol_wt) * self.stepsize * rho

        eff_a = sigma*n
        total_a = sample_dim**2
        #print(eff_a/total_a)
        return (eff_a/total_a)

    """
    def scattering_angle(self, ke : np.float64) -> np.float64:
        Samples from the differential cross section to get a scattering angle
        
        Args:
            ke (np.float64): The kinetic energy for the alpha

        Returns:
            np.float64: The sampled scattering angle
        
        while True:
            # first we sample from a uniform distribution of valid x-values
            xsample = np.float32(crandom.uniform(self.theta_min, self.theta_max).get())
            # then find the scaled differential crosssection at the x-sample
            scx = self.differential_cx(xsample, ke, scaled=True)
            # and then return the x-sample if a random number is less than that value
            if crandom.uniform(0., 1) < scx:
                return cp.float32(xsample.get())
    """
        

    def differential_cx(self, theta, ke, scaled=False):
        cx_pt = float(self.cx_interpolator((ke, theta)))
        if scaled == True:
            scale = self.cx_interpolator([(ke, i) for i in np.linspace(self.theta_min, self.theta_max, 10)]).max()
            return (1/scale)*cx_pt 
        return cx_pt 

    def particle_sim(self):
        """Runs the particle simulation step on the GPU. Computes only the
        scattered particles, and then the remainder is filled in the quenched
        spectrum step 
        """
        # first we compute a matrix of uniform random numbers on the GPU of size
        # (n_alphas, alpha_steps)
        scatter_rolls_gpu = crandom.uniform(low=0.0, high=1.0, size=(self.num_alphas, len(self.alpha_path)))
        # now we compare a precomputed table of scattering probabilities to each
        # column
        output_scatters_gpu = cp.less(scatter_rolls_gpu, cp.array(self.s_prob_lut[None, :]))
        scatter_alpha, scatter_step = cp.nonzero(output_scatters_gpu)
        # now, we take the array of nonzero indices and compute the scatters on
        # the CPU
        scattered_alphas = []
        print("Done GPU Particle Sim Step")
        if not (scatter_alpha.any() or scatter_step.any()):
            return
        for alpha, step in zip(scatter_alpha, scatter_step):
            # skip if it's not the first scatter per alpha
            if alpha.get() in scattered_alphas:
                continue
            # and add the current alpha to the list
            scattered_alphas.append(alpha.get())
            # grab the energy for the step which the scatter happened at
            step_energy = self.alpha_path[step.get()]
            #self._alpha_sim.extend(np.abs(np.diff(self.alpha_path[0:step.get()])))
            q_1 = self.alpha_quenched_value(self.alpha_path_gpu[:step])
            # compute scattering
            scatter_angle = self.scattering_angle(step_energy)
            transf = energy_transfer(step_energy, scatter_angle)
            a_e = transf.e_alpha
            p_e = transf.e_proton
            self._proton_sim.append(self.proton_factor*p_e)
            q_2 = self.alpha_quenched_value(cp.array(gen_alpha_path(a_e, self.stp, self.epsilon, self.stepsize)))
            #self._alpha_sim.append(np.float32((q_1 + q_2).get()))
            self._alpha_sim.append(np.float32((q_1 + q_2).get()))
            #self._alpha_sim[-1].extend(gen_alpha_path(a_e, self.stp,
            #epsilon=self.epsilon, stepsize=self.stepsize))

    def alpha_quenched_value(self, alpha_deps, alpha_factor = 1.0):
        # want the factor to be one here - so later we can plot for different
        # quenching factors
        return alpha_factor*cp.sum(cp.abs(cp.diff(alpha_deps)))
        
    def fill_spectrum(self, num_scatters):
        if self._quenched_spec == None:
            self._quenched_spec = []
        #a_path = np.frombuffer(alpha_path, dtype=np.float64)
        #qv = self.quenched_spectrum(ap)
        qv = self.alpha_factor*np.abs(np.sum(np.diff(self.alpha_path)))
        # fills the spectrum for loaded data 
        alphas_left = self.num_alphas - num_scatters
        print(f"Filling {alphas_left} events")
        for i in range(alphas_left):
            self._quenched_spec.append(qv)
        
    def quenched_spectrum(self):
        if (len(self._proton_sim) != 0) and (len(self._alpha_sim) != 0):
            self._quenched_spec.extend(np.add(np.multiply(self.alpha_factor, self._alpha_sim), np.multiply(self.proton_factor, self._proton_sim)))
        self.fill_spectrum(len(self._alpha_sim))

    def detsim(self):
        means = cp.array([e*self.nhit for e in self._quenched_spec])
        variances = cp.array([np.sqrt(e*self.nhit)/self.nhit for e in self._quenched_spec])
        self._result = np.asarray(cp.divide(crandom.normal(loc=means, scale=variances), self.nhit).get())