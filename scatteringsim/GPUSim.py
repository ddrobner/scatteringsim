import cupy as cp
import numpy as np
import pandas as pd

import cupy.random as crandom

from scatteringsim.utils import read_stopping_power, gen_alpha_path
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
        #self._alpha_sim = None
        #self._quenched_spec = None 
        #self._result = None

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

    def particle_sim(self):
        scatter_rolls_gpu = crandom.uniform(low=0.0, high=1.0, size=(self.num_alphas, len(self.alpha_path)))
        #vless = cp.vectorize(cp.less)
        output_scatters_gpu = cp.less(scatter_rolls_gpu, self.alpha_path[:, None])
        output_scatters = output_scatters_gpu.asnumpy()
        return output_scatters.nonzero()