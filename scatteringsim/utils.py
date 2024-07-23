import pandas as pd
import numpy as np
import numpy.typing as npt

from pathlib import Path
from scatteringsim.structures import ScatterFrame

def read_stopping_power(filename) -> pd.DataFrame:
    stoppingpowers = pd.read_csv(Path(filename))
    # rename cols to make them easier to reference
    stoppingpowers.columns = ["KE", "electron", "nuclear", "total"]
    return stoppingpowers

def stp_interp(energy:float, stp: pd.DataFrame) -> np.float64:
    # NOTE this assumes that the stopping powers are sorted
    # we get them this way from ASTAR so it's not an issue, but we can fix that if need be
    for k in stp.index:
        if energy <= stp["KE"][k+1] and energy >= stp["KE"][k]:
            return np.interp(energy, list(stp["KE"]), list(stp["total"]))
        else:
            return ((list(stp["total"])[-1])/list(stp["KE"])[-1])*energy

def gen_alpha_path(e_0, stp, epsilon=0.1, stepsize=0.001) -> npt.NDArray[np.float64]:
    e_i = e_0
    alpha_path = []
    while e_i > epsilon:
        alpha_path.append(e_i)
        e_i = e_i - stp_interp(e_i, stp)*stepsize
    return np.array(alpha_path)

def energy_transfer(e_alpha, scatter_angle):
    m_alpha = np.float128(6.646E-27) # kg
    m_proton = np.float128(1.6726E-27) # kg

    e_alpha = np.float128(e_alpha)

    Theta = scatter_angle

    frac_energy = (np.power(m_alpha, 2) + 2*m_alpha*m_proton*np.cos(Theta) + np.power(m_proton, 2))/(np.power(m_alpha + m_proton, 2)) - 1
    ealpha_f = e_alpha*frac_energy + e_alpha
    eproton_f = np.abs(e_alpha*frac_energy)

    return ScatterFrame(ealpha_f, eproton_f, Theta)