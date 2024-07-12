from scatteringsim.helpers.crossection import diffcx_riemann_sum, scattering_probability, scattering_angle
from scatteringsim.helpers.energytransfer import energy_transfer
from scatteringsim.structures import AlphaEvent, ScatterFrame
from scatteringsim.helpers.alphapath import gen_alpha_path

from multiprocessing import Pool
from random import gauss
from numpy import sqrt as nsqrt

import pandas as pd

import random
import copy

def scatter_sim(e_0: float, alpha_path : list, stp : pd.DataFrame, tck, stepsize=0.001, epsilon=0.1, density=0.8562) -> AlphaEvent:
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
    scatter_e = []
    alpha_out = []
    for s in range(len(a_path)):
        rsum = diffcx_riemann_sum(a_path[s], tck)
        if scattering_probability(a_path[s], stepsize, rsum, density=density) > random.random():
            scatter_angle = scattering_angle(a_path[s], tck)
            transfer_e = energy_transfer(a_path[s], tck, scatter_angle=scatter_angle)
            a_path[s] = transfer_e.e_alpha
            #a_path = a_path[0:s-1]
            alpha_out.append(a_path[0:s-1])
            proton_event_path.append(transfer_e.e_proton)
            scatter_e.append(ScatterFrame(scatter_angle, transfer_e.e_proton, scatter_angle))
            alpha_out.append(gen_alpha_path(transfer_e.e_alpha, stp, stepsize=stepsize, epsilon=epsilon))
            break

    return AlphaEvent(alpha_out, proton_event_path, scatter_e)

def sim_wrapper(arg):
    args, kwargs = arg
    return scatter_sim(*args, **kwargs)

def start_sim(e_0: float, n_particles: int, stp: pd.DataFrame, tck, stepsize=0.001, epsilon=0.1, density=0.8562):
    alpha_path = gen_alpha_path(e_0, stp, epsilon=epsilon, stepsize=stepsize)
    arg = (e_0, alpha_path, stp, tck)
    kwargs = {'stepsize': stepsize, 'epsilon': epsilon, 'density': density}
    with Pool() as p:
        sim_data = p.map(sim_wrapper, [(arg, kwargs) for i in range(n_particles)])
        p.close()
        p.join()

    return sim_data

def quenched_spectrum(sim_data: AlphaEvent,  proton_factor: float, alpha_factor: float=0.1) -> None:
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

def quenched_spectrum_wrapper(arg):
    args, kwargs = arg
    return quenched_spectrum(*args, **kwargs)

def quenched_spectrum_multithread(sim_data: list[AlphaEvent], proton_factor: float, alpha_factor: float=0.1) -> list[list]:
    q_spec = []
    arg = (proton_factor,)
    kwargs = {'alpha_factor': alpha_factor}

    with Pool() as p:
        q_spec = p.map(quenched_spectrum_wrapper, [((i, *arg), kwargs) for i in sim_data])
        p.close()
        p.join()

    q_spec_flattened = [l
                        for ls in q_spec
                        for l in ls
                        ]

    return q_spec_flattened

def compute_smearing(e_i, nhit):
   return gauss(e_i*nhit, nsqrt(e_i*nhit))/nhit

def smeared_spectrum(quenched_spectrum: list[float], nhit: int):
    with Pool() as p:
        smeared_spec = p.starmap(compute_smearing, [(i, nhit) for i in quenched_spectrum])
        p.close()
        p.join()
    return smeared_spec