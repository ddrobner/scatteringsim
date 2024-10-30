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

        print(f"Simulating {self.num_alphas} alpha particles!")
        
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
    

    def pop_particle(self, idx: int) -> None:
        self._particle_results.pop(idx)

    def add_particle(self, alpha_val, proton_val, scatter_num=0) -> None:
        self._particle_results.append(ScatteredDeposit(alpha_val, proton_val, scatter_num))

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

        print("Done GPU Particle Sim Step")
        #if not (scatter_alpha.any() or scatter_step.any()):
        #    return

        # group up the scatter indices
        scatter_points = {int(a.get()):list() for a in scatter_alpha}
        for a, idx in zip(scatter_alpha, scatter_step):
            scatter_points[int(a.get())].append(int(idx.get()))

        # now we do the computation
        for alpha in scatter_points.keys():
            scatters = scatter_points[alpha]

            # set this to "infinity" to always get the first scatter
            prev_scatter = float('inf')

            scatter_num = 0
            step = scatters[0]
            for s in scatters:
                if step < prev_scatter:
                    step_energy, e_alpha = transform_energies(self.alpha_path[s]) 
                    scatter_angle = self.scattering_angle(step_energy)
                    transf = energy_transfer(e_alpha, scatter_angle)

                    self._particle_results.append(ScatteredDeposit(transf.e_alpha, transf.e_proton, scatter_num))

                    step = find_nearest_idx(self.alpha_path, transf.e_alpha)

                    continue
                step = s
                scatter_num += 1

        """
        prev_scatter_alpha = 0
        prev_scatter_step = {k:0 for k in scatter_alpha}
        scatter_nums = {k:0 for k in scatter_alpha}
        for alpha, step in zip(scatter_alpha, scatter_step):

            if (alpha == prev_scatter_alpha and step > prev_scatter_step) or alpha != prev_scatter_alpha:
                step_energy, e_alpha = transform_energies(self.alpha_path[step])
                scatter_angle = self.scattering_angle(step_energy)
                transf = energy_transfer(e_alpha, scatter_angle)

                self._particle_results.append(ScatteredDeposit(transf.e_alpha, transf.e_proton, scatter_nums[scatter_alpha]))

                # now we need to do the step updating
                
                scatter_nums[alpha] += 1
        """

        """
        # iterate over matrix columns
        for alpha in output_scatters_gpu[scatter_alpha, :]:
            print(alpha)
            # now we walk through the alpha steps and jump forward accordingly
            # after scatters
            step = 0
            scatter_num = 0
            #print(len(self.alpha_path))
            while step < len(self.alpha_path):
                if alpha[step]:
                    step_energy, e_alpha = transform_energies(self.alpha_path[step])
                    scatter_angle = self.scattering_angle(step_energy)
                    transf = energy_transfer(e_alpha, scatter_angle)
                    
                    self._particle_results.append(ScatteredDeposit(transf.e_alpha, transf.e_proton, scatter_num))
                    scatter_num += 1

                    step = find_nearest_idx(self.alpha_path, transf.e_alpha)
                    continue

                step += 1
        """

    # TODO combine the single and multi scatter functions into one 
    def compute_scatter(self, scatter_alpha, scatter_step):
        scattered_alphas = []
        for alpha, step in zip(scatter_alpha, scatter_step):
            # skip if it's not the first scatter per alpha
            if alpha.get() in scattered_alphas:
                continue
            # and add the current alpha to the list
            scattered_alphas.append(alpha.get())
            # grab the energy for the step which the scatter happened at
            lab_frame_alpha_e = self.alpha_path[step.get()]
            palpha_lab = -1*np.sqrt(2*m_alpha*lab_frame_alpha_e*mev_to_j)
            v_cm = palpha_lab/(m_alpha + m_proton)
            step_energy = 0.5*m_proton*np.power(v_cm, 2)
            e_alpha = 0.5*m_alpha*np.power(v_cm + palpha_lab/m_alpha, 2)
            #step_energy = self.alpha_path[step.get()]
            #self._alpha_sim.extend(np.abs(np.diff(self.alpha_path[0:step.get()])))
            # compute scattering
            scatter_angle = self.scattering_angle(step_energy)
            transf = energy_transfer(e_alpha, scatter_angle)
            a_e = transf.e_alpha
            p_e = transf.e_proton
            if np.isnan(p_e):
                print(f"Proton Energy is NaN! Scattering Angle is: {scatter_angle}")

            # Ignoring alpha deposits, since that takes up a LOT of runtime
            # (60%+) and we are only interested in protons
            #q_1 = self.alpha_quenched_value(self.alpha_path_gpu[:step])
            #q_2 = self.alpha_quenched_value(cp.array(gen_alpha_path(a_e, self.stp, self.epsilon, self.stepsize)))
            #self._alpha_sim.append(np.float32((q_1 + q_2).get()))

            #self.multiscatter(a_e)

            self._particle_results.append(ScatteredDeposit(a_e, p_e, 0))
            #self._proton_sim.append(p_e)
            #self._scatter_num.append(0)
    # check for additional scatters on the CPU - TODO run on GPU
    # it is a pretty big pain to deal with inhomogenous arrays, so for now we do
    # this
    # there are very few scatters rel. to the number of particles - so this
    # shouldn't be too bad
    def multiscatter(self, e_i, n_scatter=1):
        # we cheat a little here sacrificing some accuracy - instead of
        # generating a new alpha path we slice the old one to the nearest index
        # with the given energy
         alpha_path = self.alpha_path[find_nearest_idx(self.alpha_path, e_i):]
         for i in range(len(alpha_path)):
             if self.scattering_probability(alpha_path[i]) > fastrand.pcg32bounded(2**32 - 1)/(2**32 - 1):
                palpha_lab = -1*np.sqrt(2*m_alpha*alpha_path[i]*mev_to_j)
                v_cm = palpha_lab/(m_alpha + m_proton)
                e_p = 0.5*m_proton*np.power(v_cm, 2)


                q_1 = self.alpha_quenched_value(cp.array(alpha_path[0:i]))
                # the alpha spectrum part isn't so important here
                self._alpha_sim.append(q_1.get())
                scatter_angle = self.scattering_angle(e_p)
                transf = energy_transfer(v_cm + palpha_lab/m_alpha, scatter_angle)
                a_e = transf.e_alpha
                p_e = transf.e_proton

                if np.isnan(p_e):
                    print(f"Proton Energy is NaN! Scattering Angle is: {scatter_angle}")

                self._proton_sim.append(p_e)
                self._scatter_num.append(n_scatter)
                # limit this to three scatters - I think anything more is a
                # little too much
                # ignore super low energy scatters
                if n_scatter <= 3 and len(alpha_path) > 10000:
                    # recursion???
                    self.multiscatter(a_e, n_scatter+1)

        # this (regrettably) uses a bit of recursion. It will terminate no
        # matter what after the third scatter (beyond that is highly unlikely
        # and the energy will be quite low), as well as if there aren't any
        # second scatters

            
    def alpha_quenched_value(self, alpha_deps, alpha_factor = 1.0):
        # want the factor to be one here - so later we can plot for different
        # quenching factors
        return alpha_factor*cp.sum(cp.abs(cp.diff(alpha_deps)))
        
    def fill_spectrum(self):
        qv = self.alpha_factor*np.abs(np.sum(np.diff(self.alpha_path)))

        # fills the spectrum for loaded data 
        alphas_left = self.num_alphas - len(self.quenched_spec)
        print(f"Filling {alphas_left} events")
        for i in range(alphas_left):
            self._quenched_spec.append(qv)
        
    def quenched_spectrum(self):
        if len(self.particle_results) != 0:
            alpha_sim = []
            proton_sim = []
            for d in self.particle_results:
                alpha_sim.append(d.alpha_energy)
                proton_sim.append(d.proton_energy)
            self._quenched_spec.extend(np.add(np.multiply(self.alpha_factor, alpha_sim), np.multiply(self.proton_q_factor, proton_sim)))

    def detsim(self):
        self._result = [random.gauss(e_i*parameters.nhit, np.sqrt(e_i*parameters.nhit))/parameters.nhit for e_i in self._quenched_spec]