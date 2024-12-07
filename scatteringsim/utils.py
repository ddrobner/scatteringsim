import pandas as pd
import numpy as np
import numpy.typing as npt

from pathlib import Path
from scatteringsim.constants import *
from scatteringsim.structures import ScatterFrame

def read_stopping_power(filename: str) -> pd.DataFrame:
    """Reads stopping power tables

    Args:
        filename (str): The cross section table filename

    Returns:
        pd.DataFrame: A dataframe with the table data
    """
    stoppingpowers = pd.read_csv(Path(filename))
    # rename cols to make them easier to reference
    stoppingpowers.columns = ["KE", "electron", "nuclear", "total"]
    return stoppingpowers

def stp_interp(energy:float, stp: pd.DataFrame) -> np.float32:
    """Interpolates stopping power

    Args:
        energy (float): Energy
        stp (pd.DataFrame): The stopping power dataframe

    Returns:
        np.float32: The stopping power value at the given energy
    """
    # NOTE this assumes that the stopping powers are sorted
    # we get them this way from ASTAR so it's not an issue, but we can fix that if need be
    for k in stp.index:
        if energy <= stp["KE"][k+1] and energy >= stp["KE"][k]:
            return np.interp(energy, list(stp["KE"]), list(stp["total"]))
        else:
            return ((list(stp["total"])[-1])/list(stp["KE"])[-1])*energy

def gen_alpha_path(e_0: float, stp: pd.DataFrame, epsilon=0.1, stepsize=0.001) -> npt.NDArray[np.float32]:
    """Generates alpha energy deposits

    Args:
        e_0 (float): The initial alpha energy
        stp (pd.DataFrame): Stopping power
        epsilon (float, optional): Min value to stop at. Defaults to 0.1.
        stepsize (float, optional): The stepsize. Defaults to 0.001.

    Returns:
        npt.NDArray[np.float32]: An array of the alpha deposits
    """
    e_i = e_0
    alpha_path = []
    while e_i > epsilon:
        alpha_path.append(e_i)
        e_i = e_i - stp_interp(e_i, stp)*stepsize
    return np.array(alpha_path)

def energy_transfer(e_alpha: float, scatter_angle: float) -> ScatterFrame:
    """Computes energy transfer from alpha to proton

    Args:
        e_alpha (float): The alpha energy in lab frame 
        scatter_angle (float): A scattering angle 

    Returns:
        ScatterFrame: The energies after scattering 
    """
    Theta = scatter_angle

    frac_energy = (np.power(m_alpha, 2) + 2*m_alpha*m_proton*np.cos(Theta) + np.power(m_proton, 2))/(np.power(m_alpha + m_proton, 2)) - 1
    ealpha_f = e_alpha*frac_energy + e_alpha
    eproton_f = np.abs(e_alpha*frac_energy)

    return ScatterFrame(np.float32(ealpha_f), np.float32(eproton_f), np.float32(Theta))

def find_nearest_idx(arr: np.ndarray, val: float) -> int:
    """Finds closest index to value

    Args:
        arr (_type_): The array to search
        val (_type_): The value to look for

    Returns:
        int: Closest index
    """
    #arr = np.asarray(arr)
    #idx = (np.abs(arr - val)).argmin()
    inv_idx = np.searchsorted(arr[::-1], val)
    return len(arr) - inv_idx

def transform_energies(alpha_energy_lab: np.float32):
    """Changes from lab frame to CM-frame

    Args:
        alpha_energy (np.float32): The alpha energy (initial proton energy
        assumed to be zero) 
    """
    #palpha_lab = -1*np.sqrt(2*m_alpha*alpha_energy_lab*mev_to_j)
    #v_cm = palpha_lab/(m_alpha + m_proton)
    #step_energy = 0.5*m_proton*np.power(v_cm, 2)
    #e_alpha = 0.5*m_alpha*np.power(v_cm + palpha_lab/m_alpha, 2)
    palpha_lab = np.sqrt(2*m_alpha*alpha_energy_lab*mev_to_j)
    valpha_lab = palpha_lab/m_alpha
    valpha_cmf = palpha_lab*(m_proton/(m_alpha*(m_alpha + m_proton)))
    vproton_cmf = -1*palpha_lab/(m_alpha+m_proton)

    eproton_cmf = 0.5*m_proton*np.power(vproton_cmf, 2)
    ealpha_cmf = 0.5*m_alpha*np.power(valpha_cmf, 2)
    
    return (eproton_cmf*j_to_mev, ealpha_cmf*j_to_mev)