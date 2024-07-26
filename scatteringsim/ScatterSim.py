from scatteringsim.utils import read_stopping_power, gen_alpha_path, energy_transfer
from scatteringsim.structures import *

from scipy.constants import Avogadro
from scipy.interpolate import LinearNDInterpolator
import random
from multiprocessing import Pool, cpu_count
from multiprocessing import RawArray as mparray
from decimal import Decimal

from tqdm import tqdm

import numpy as np
import pandas as pd

import ctypes

alpha_path = None

class ScatterSim:
    def __init__(self, e_0: float, num_alphas: int, stepsize: float, nhit: int, stp_fname: str, cx_fname: str, proton_factor: float = 0.5):
        # declaring this as a global to speed up multprocessing 
        global alpha_path
        
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
        apath_base = gen_alpha_path(self.e_0, self.stp, epsilon=self.epsilon, stepsize=self.stepsize)
        # can have no lock here since the array is read only
        alpha_path = mparray(ctypes.c_double, apath_base)

        # now we handle setting up the cross section
        self.cx = pd.read_csv(cx_fname)
        # converting to radians
        # NOTE this means we need to scale the cx when integrating by 
        # 180/pi due to the transformation
        self.cx['theta'] = np.deg2rad(self.cx['theta'])
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

        # this is hardcoded for the fixed energy cx sim
        # because the function call along with the manual computation was VERY
        # expensive for each step
        # and will be reserved for the 2D cx only
        self.scattering_probability = 2*np.pi*6.576617367299405e-08
        
        # and set up class variable to store the outputs
        self._alpha_sim = None
        self._quenched_spec = None
        self._result = None

    @property
    def numalphas(self):
        return self.num_alphas

    @property
    def step_size(self):
        return self.stepsize

    @property
    def step_size_latex(self):
        s = f"{Decimal(str(self.stepsize)):.2E}"
        n = int(float(s.split("E")[0]))
        mag = s.split("E")[1]
        
        #return f"${n}^{{{mag}}}$"
        return f"$10^{{{mag}}}$"
        

    @property
    def protonfactor(self):
        return self.proton_factor

    @protonfactor.setter
    def quenching_factor(self, factor):
        self.proton_factor = factor

    @property
    def alpha_sim(self) -> list[AlphaEvent]:
        """Particle Simulation Results

        Returns:
            list[AlphaEvent]: A list of AlphaEvent objects containing the
            results for each particle 
        """
        return self._alpha_sim
    
    @property
    def quenched_spectrum(self) -> list[np.float64]:
        """Simulated quenched spectrum

        Returns:
            list[np.float64]: A list of the quenched energy values for each particle 
        """
        return self._quenched_spec

    @property
    def result(self) -> list[np.float64]:
        """Simulation results with detector simulation
        
        Returns:
            list[np.float64]: A list of the simulated energy values (one per particle)
        """
        return self._result

    """
    def differential_cx(self, theta, ke, scaled=False):
        cx_pt = float(self.cx_interpolator((ke, theta)))
        if scaled == True:
            scale = self.cx_interpolator([(ke, i) for i in np.linspace(self.theta_min, self.theta_max, 10)]).max()
            return (1/scale)*cx_pt 
        return cx_pt
    """

    def differential_cx(self, theta : np.float64, ke : np.float64, scaled=False) -> np.float64:
        """Computed the differential cross section at a point

        Args:
            theta (np.float64): The angle to compute the cross section at 
            ke (np.float64): The kinetic energy for the cross section 
            scaled (bool, optional): Normalize so that the maximum value
            returned is 1. Useful for anything involving probabilities. Defaults to False.

        Returns:
            np.float64: _description_
        """
        theta_s = self.cx['theta'].to_numpy()
        cx_s = self.cx['cx'].to_numpy()
        cx_pt = np.interp(theta, theta_s, cx_s)
        if scaled:
            scale = np.interp(self.theta_min, theta_s, cx_s)
            return cx_pt/scale
        return cx_pt

    def total_crossection(self, ke : np.float64) -> np.float64:
        """Computes the total cross section with a trapezoidal riemann sum

        Args:
            ke (np.float64): The kinetic energy for the cross section 

        Returns:
            np.float64: The total cross section 
        """
        #return np.interp(ke, self.total_cx['Energy'].to_numpy(),
        #self.total_cx['Total'].to_numpy())
        return np.trapz([i*(180/np.pi) for i in self.cx['cx'].to_numpy()], self.cx['theta'].to_numpy())

    # moving this to a class method to avoid all of this passing variables
    # around nonsense
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


    """
    def scattering_probability(self, ke) -> np.float64:
        sample_dim = 1
        sigma = self.total_crossection(ke)*1E-24

        # now let's do the same stuff as before to determine the probability
        rho = self.density # g/cm^3, see above
        n = Avogadro/(self.mol_wt) * self.stepsize * rho

        eff_a = sigma*n
        total_a = sample_dim**2
        print(eff_a/total_a)
        return eff_a/total_a
    """

    def scatter_sim(self) -> AlphaEvent:
        """Function to simulate a single alpha particle

        Returns:
            AlphaEvent: The simulation data for the simulated particle 
        """
        global alpha_path
        a_path = np.frombuffer(alpha_path, dtype=np.float64)
        alpha_out = [a_path]
        proton_event_path, scatter_e = [], []
        scattered = False
        for s in range(len(a_path)):
            #if self.scattering_probability(a_path[s]) > np.random.uniform(low=0., high=1.):
            if self.scattering_probability > np.random.uniform(low=0., high=1.):
                if not scattered:
                    # don't create these until there is a scattering 
                    alpha_out = [] 
                scattered = True
                # doing this here since this runs far more rarely
                scatter_angle = self.scattering_angle(a_path[s])
                transfer_e = energy_transfer(a_path[s], scatter_angle)
                print(f"Scattered: {round(scatter_angle, 4)}rad {transfer_e.e_proton}MeV p+")
                # this should only store a reference to the alpha path and not copy
                alpha_out.append(a_path[0:s])
                proton_event_path.append(transfer_e.e_proton)
                scatter_e.append(ScatterFrame(transfer_e.e_alpha, transfer_e.e_proton, scatter_angle))
                alpha_out.append(gen_alpha_path(transfer_e.e_alpha, self.stp, stepsize=self.stepsize, epsilon=self.epsilon))
                break
        #if len(alpha_out) == 0:
        #    # this happens if we scatter on the first step
        #    alpha_out = [alpha_path]
        for a in range(len(alpha_out)):
            if len(alpha_out[a]) == 0:
                alpha_out[a] = [np.float64(0)]
                

        return AlphaEvent(alpha_out, proton_event_path, scatter_e)

    def quenched_spectrum(self, sim_data: AlphaEvent,  proton_factor: float, alpha_factor: float=0.1) -> list[np.float64]:
        """Computes the quenched value for a result from scatter_sim

        Args:
            sim_data (AlphaEvent): The output of a particle simulation 
            proton_factor (float): The proton quenching factor 
            alpha_factor (float, optional): The alpha quenching factor. Defaults to 0.1.

        Returns:
            list[np.float64]: The quenched energy for the input particle
        """

        # TODO update everything so that this doesn't return a list and instead
        # the single value
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

    def particle_scint_sim(self, args) -> tuple[AlphaEvent, list[np.float64]]:
        """Simulation and quenching for a single alpha particle

        Args:
            args (iterable): A size-4 iterable containing args and kwargs for
            each of scatter_sim and quenched_spectrum respectively

        Returns:
            tuple[AlphaEvent, list[np.float64]]: _description_
        """
        scatter_args, scatter_kwargs, quench_args, quench_kwargs = args
        a_part = self.scatter_sim(*scatter_args, **scatter_kwargs)
        q_part = self.quenched_spectrum(a_part, *quench_args, **quench_kwargs)
        return (a_part, q_part[0])
    
    def compute_smearing(self, e_i : np.float64, nhit : int) -> np.float64:
        """Detector smearing of quenched energy values

        Args:
            e_i (np.float64): Quenched energy 
            nhit (int): Detector smearing factor in nhits/mev 

        Returns:
            np.float64: Detector simulated energy spectrum
        """
        return random.gauss(e_i*nhit, np.sqrt(e_i*nhit))/nhit

    def start(self):
        """Starts the simulation using the parameters given in the constructor
        """
        print("Starting Simulation.....")
        # open a pool as well as create a progress bar
        with Pool(cpu_count()) as p, tqdm(total=self.num_alphas) as pbar:
            # store the apply_async results
            r = [p.apply_async(self.particle_scint_sim, ((tuple(), dict(), (self.proton_factor,), {"alpha_factor":0.1}),), callback=lambda _: pbar.update(1)) for i in range(self.num_alphas)]
            # and if we have all of those done unpack them into a list
            print("Simulation Started!")
            if(len(r) == self.num_alphas):
                tmp_res = [i.get() for i in r]
            p.close()
            pbar.close()
            
        self._alpha_sim, self._quenched_spec = zip(*tmp_res)
        # now we do the det smearing
        self._result = [self.compute_smearing(i, self.nhit) for i in self._quenched_spec] 

    def particle_sim(self):
        """Performs only the particle simulation
        """
        with Pool(cpu_count()) as p, tqdm(total=self.num_alphas) as pbar:
            r = [p.apply_async(self.scatter_sim, callback=lambda _: pbar.update(1)) for i in range(self.num_alphas)]
            print("Simulation Started!")
            if(len(r) == self.num_alphas):
                tmp_res = [i.get() for i in r]
            p.close()
            pbar.close()
        self._alpha_sim = tmp_res

    def recompute_spectrum(self):
        """Recomputes the quenched spectrum and detector smearing, using the
        same particle simulation
        """
        with Pool(cpu_count()) as p:
            self._quenched_spec = p.starmap(self.quenched_spectrum, [(i, self.proton_factor) for i in self._alpha_sim])
        self._quenched_spec = [l for ls in self._quenched_spec for l in ls]
            
        self._result = [self.compute_smearing(i, self.nhit) for i in self._quenched_spec] 