import cupy as cp
import numpy as np
import pandas as pd

import cupy.random as crandom

import random

from scatteringsim.utils import read_stopping_power, gen_alpha_path, energy_transfer
from scatteringsim.structures import ScatterFrame, AlphaEvent
from scipy.interpolate import LinearNDInterpolator

from scipy.constants import Avogadro


class GPUSim:
    def __init__(self, e_0: float, num_alphas: int, stepsize: float, nhit: int, stp_fname: str, cx_fname: str, proton_factor: float = 0.5):
        # declaring this as a global to speed up multprocessing 
        #global alpha_path
        
        # declaring some constants 
        self.stp = read_stopping_power(stp_fname)
        self.e_0 = e_0
        self.num_alphas = num_alphas
        self.proton_factor = proton_factor
        self.stepsize = stepsize
        self.nhit = nhit

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
        temp_es = []
        temp_cx = []
        for e in self.cx['energy'].unique():
            angles = self.cx[self.cx['energy'] == e]['theta']
            dcx = self.cx[self.cx['energy'] == e]['cx']
            if len(angles > 3):
                itg = np.trapz(dcx, angles)
                if(itg != 0):
                    temp_es.append(e)
                    temp_cx.append(itg)
        self.total_cx = pd.DataFrame(zip(temp_es, temp_cx), columns=['Energy',
        'Total'])
        del temp_es
        del temp_cx

        # this is only done once so can do it on the cpu
        self.alpha_steps = len(self.alpha_path)
        self.s_prob_lut = cp.array([self.scattering_probability(j) for j in self.alpha_path])

        # and set up class variable to store the outputs
        self._alpha_sim = []
        self._proton_sim = []
        self._quenched_spec = [] 
        self._result = []

    @property
    def result(self):
        return self._result

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

    def scattering_angle(self, ke : np.float64) -> np.float64:
        """Samples from the differential cross section to get a scattering angle
        
        Args:
            ke (np.float64): The kinetic energy for the alpha

        Returns:
            np.float64: The sampled scattering angle
        """
        while True:
            # first we sample from a uniform distribution of valid x-values
            xsample = random.uniform(self.theta_min, self.theta_max)
            # then find the scaled differential crosssection at the x-sample
            scx = self.differential_cx(xsample, ke, scaled=True)
            # and then return the x-sample if a random number is less than that value
            if random.random() < scx:
                return xsample 

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
        print(f"{len(scatter_alpha.get())} scatters.")
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
            # make an object to hold the data
            #p = AlphaEvent()
            # crucially, extend with the array instead of append (for faster
            # computation later)
            #p.alpha_path.extend(self.alpha_path[0:step])
            self._alpha_sim.extend(np.abs(np.diff(self.alpha_path[0:step.get()])))
            # compute scattering
            scatter_angle = self.scattering_angle(step_energy)
            a_e, p_e = energy_transfer(step_energy, scatter_angle)
            #p.alpha_path.extend(gen_alpha_path(a_e, self.stp,
            #epsilon=self.epsilon, stepsize=self.stepsize))
            self._proton_sim.append(p_e)
            self._alpha_sim[-1].extend(gen_alpha_path(a_e, self.stp, epsilon=self.epsilon, stepsize=self.stepsize))


    def fill_spectrum(self, num_scatters):
        global alpha_path
        if self._quenched_spec == None:
            self._quenched_spec = []
        #a_path = np.frombuffer(alpha_path, dtype=np.float64)
        #qv = self.quenched_spectrum(ap)
        qv = 0.1*np.abs(np.sum(np.diff(self.alpha_path)))
        print(qv)
        # fills the spectrum for loaded data 
        alphas_left = self.num_alphas - num_scatters
        print(f"Filling {alphas_left} events")
        for i in range(alphas_left):
            self._quenched_spec.append(qv)
        #self._result = [self.compute_smearing(i) for i in self._quenched_spec] 
        
    def quenched_spectrum(self):
        if len(self._proton_sim) != 0:
            proton_gpu = self.proton_factor*cp.array(self._proton_sim)
        else:
            proton_gpu = cp.array((0,))

        if len(self._proton_sim) != 0:
            alphas_gpu = 0.1*cp.array(self._alpha_sim)
        else:
            alphas_gpu = cp.array((0,))
        #alpha_quench = cp.sum(alphas_gpu)
        #proton_quench = cp.sum(proton_gpu)
        self._quenched_spec.extend(np.asarray(cp.add(alphas_gpu, proton_gpu).get()))
        self.fill_spectrum(len(alphas_gpu))

    def detsim(self):
        means = cp.array([e*self.nhit for e in self._quenched_spec])
        variances = cp.array([np.sqrt(e*self.nhit)/self.nhit for e in self._quenched_spec])
        self._result = crandom.normal(loc=means, scale=variances)