import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

import fastrand

import cupy as cp
import numpy as np
import pandas as pd

import cupy.random as crandom

from pathlib import Path
from dataclasses import astuple

import random

from scatteringsim.utils import read_stopping_power, gen_alpha_path, energy_transfer, find_nearest_idx, transform_energies
import scatteringsim.sim.sim_init as sim_init

from scatteringsim import parameters

from scatteringsim.structures import ScatteredDeposit


from scipy.constants import Avogadro
from scatteringsim.constants import *


class GPUSim:
    def __init__(self, e_0: float, num_alphas: int, stepsize: float, stp_fname: str, cx_fname: str, proton_factor: float = 0.5):
        # declaring this as a global to speed up multprocessing 
        #global alpha_path

        # declaring some constants 
        self.stp = read_stopping_power(stp_fname)
        self.e_0 = e_0
        self.proton_q_factor = proton_factor
        self.stepsize = stepsize

        self.alpha_factor = 0.1

        self.alpha_path = gen_alpha_path(self.e_0, self.stp, epsilon=parameters.epsilon, stepsize=self.stepsize)

        # prepare the cx and unpack it
        cx_pack = sim_init.prep_cx(pd.read_csv(cx_fname, dtype=np.float32))
        self.cx = cx_pack['differential']
        self.total_cx = cx_pack['total']
        del cx_pack

        self.cx_interpolator = sim_init.make_cx_interpolator(self.cx) 

        # find out how large the list of alpha deposits is so we can compute the
        # number of alphas we can fit on the GPU
        self.alpha_steps = len(self.alpha_path)
        
        self.s_prob_lut = cp.array([self.scattering_probability(j) for j in self.alpha_path])

        self.alpha_path_gpu = cp.array(self.alpha_path)
        self.cx_inverse_dists = dict()
        for e in self.cx['energy'].unique():
            if(len(self.cx[self.cx['energy'] == e]['theta']) > 3):
                angle_vals = np.linspace(parameters.theta_min, parameters.theta_max, 10000)
                theta_vals = np.array([self.cx_interpolator((e, i)) for i in angle_vals])
                self.cx_inverse_dists[e] = sim_init.gen_inverse_dist(angle_vals, theta_vals)
                #self.cx_inverse_dists[e] = self.gen_inverse_dist(e)

        # scale number of alphas based on total GPU mem
        # we do this after everything else is initalized so we know how much
        # VRAM is free
        d_width = 8 # size of floats being used, in bytes
        if num_alphas == -1:
            avail_mem = cp.cuda.Device().mem_info[0]
            n_alphas_max = avail_mem/(self.alpha_steps*d_width)
            self.num_alphas = int(0.8*n_alphas_max)
        else:
            self.num_alphas = num_alphas

        #print(f"Simulating {self.num_alphas} alpha particles!")
        
        # and set up class variable to store the outputs
        self._particle_results : list[ScatteredDeposit] = []
        self._quenched_spec = [] 
        self._result = []

    @property
    def particle_results(self):
        return self._particle_results

    @property
    def numalphas(self):
        return self.num_alphas

    @numalphas.setter
    def numalphas(self, val):
        self.num_alphas = val
    

    def pop_particle(self, idx: int) -> None:
        self._particle_results.pop(idx)

    def add_particle(self, alpha_val: float, proton_vals: list[float], particle_id: int = 0) -> None:
        self._particle_results.append(ScatteredDeposit(alpha_val, proton_vals, particle_id))

    def add_deposit(self, deposit: ScatteredDeposit) -> None:
        self._particle_results.append(deposit)

    @property
    def quenched_spec(self):
        return self._quenched_spec

    @property
    def quenching_factor(self):
        return self.proton_q_factor

    @property
    def result(self):
        return self._result

    @property
    def angle_range(self):
        return (parameters.theta_min, parameters.theta_max)

    def reset_sim(self):
        self._particle_results.clear()
        self._quenched_spec.clear()
        self._result.clear()

    @property
    def energy_range(self):
        e_0 = self.cx['energy'].to_numpy().min()
        e_m = self.cx['energy'].to_numpy().max()
        return (e_0, e_m)

    @property
    def interpolator(self):
        return self.cx_interpolator

    @property
    def diff_cx(self):
        return self.cx

    def scattering_angle(self, ke) -> np.float32:
        # this might be a cheat but I think I'm going to just interpolate
        # between the inverse dist values for each KE
        dk = list(self.cx_inverse_dists.keys())
        dk.sort()

        rsaved = random.uniform(0, 1)
        
        return np.interp(ke, dk, [self.cx_inverse_dists[i](rsaved) for i in dk])

    def gen_dist_samples(self, ke, nsamples):
        samples = [self.scattering_angle(ke) for i in range(nsamples)]
        return samples

    def get_cx(self, ke, npoints):
        cx = [self.cx_interpolator((ke, i)) for i in np.linspace(parameters.theta_min, parameters.theta_max, npoints)]
        return cx

    @property
    def angles(self):
        return (parameters.theta_min, parameters.theta_max)


    def total_crossection(self, ke : np.float32) -> np.float32:
        """Computes the total cross section with a trapezoidal riemann sum

        Args:
            ke (np.float32): The kinetic energy for the cross section 

        Returns:
            np.float32: The total cross section 
        """
        return 2*np.pi*np.interp(ke, self.total_cx['Energy'].to_numpy(), self.total_cx['Total'].to_numpy())

    
    def scattering_probability(self, ke) -> np.float32:
        sample_dim = 1
        sigma = self.total_crossection(ke)*1E-24

        # now let's do the same stuff as before to determine the probability
        rho = lab_density # g/cm^3, see above
        n = Avogadro/(lab_mol_wt) * self.stepsize * rho * proton_scaling_factor

        eff_a = sigma*n
        total_a = sample_dim**2
        return (eff_a/total_a)

    def differential_cx(self, theta, ke, scaled=False):
        cx_pt = float(self.cx_interpolator((ke, theta)))
        if scaled == True:
            scale = self.cx_interpolator([(ke, i) for i in np.linspace(parameters.theta_min, parameters.theta_max, 10)]).max()
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

        #print("Done GPU Particle Sim Step")
        #if not (scatter_alpha.any() or scatter_step.any()):
        #    return

        # group up the scatter indices
        scatter_points = {int(a.get()):list() for a in scatter_alpha}
        for a, idx in zip(scatter_alpha, scatter_step):
            scatter_points[int(a.get())].append(int(idx.get()))

        # now we do the computation
        for alpha in scatter_points.keys():
            # get the indices of the scatters for the current alpha
            scatters = scatter_points[alpha]
            particle_result = ScatteredDeposit(0, list(), alpha)

            # initialize the variables
            #scatter_num = 0
            # always get the first scatter
            step = 0
            # now iterare through the scattered indices
            for s in scatters:
                # check if we've jumped ahead of the scatter index
                if step < s:
                    # the scattering angles comes from the proton energy in the CM-frame
                    step_energy, e_alpha = transform_energies(self.alpha_path[s]) 
                    scatter_angle = self.scattering_angle(step_energy)

                    # the energy transfer is done in the lab frame
                    transf = energy_transfer(self.alpha_path[s], scatter_angle)

                    # record the scatter energy information
                    #self._particle_results.append(ScatteredDeposit(transf.e_alpha,
                    #transf.e_proton, scatter_num))
                    particle_result.alpha_energy = transf.e_alpha
                    particle_result.proton_energies.append(transf.e_proton)

                    # and now jump ahead in the alpha path 
                    step = find_nearest_idx(self.alpha_path, transf.e_alpha)

                    # and incremement the scatter number for later tracking
                    #scatter_num += 1
            self._particle_results.append(particle_result)

    def fill_spectrum(self):
        qv = self.alpha_factor*self.e_0

        # fills the spectrum for loaded data 
        alphas_left = self.num_alphas - len(self.quenched_spec)
        print(f"Filling {alphas_left} events")
        for i in range(alphas_left):
            self._quenched_spec.append(qv)
        
    def quenched_spectrum(self):
        for p in self._particle_results:
            self.quenched_spec.append(self.alpha_factor*p.alpha_energy + self.proton_q_factor*sum(p.proton_energies))

    def detsim(self):
        self._result = [random.gauss(e_i*parameters.nhit, np.sqrt(e_i*parameters.nhit))/parameters.nhit for e_i in self._quenched_spec]