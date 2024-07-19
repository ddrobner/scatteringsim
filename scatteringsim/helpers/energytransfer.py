from scatteringsim.structures import ScatterFrame
import numpy as np

def energy_transfer(e_alpha, scatter_angle):
    m_alpha = np.float128(6.646E-27) # kg
    m_proton = np.float128(1.6726E-27) # kg

    e_alpha = np.float128(e_alpha)

    Theta = scatter_angle

    frac_energy = (np.power(m_alpha, 2) + 2*m_alpha*m_proton*np.cos(Theta) + np.power(m_proton, 2))/(np.power(m_alpha + m_proton, 2)) - 1
    ealpha_f = e_alpha*frac_energy + e_alpha
    eproton_f = np.abs(e_alpha*frac_energy)

    return ScatterFrame(ealpha_f, eproton_f, Theta)
