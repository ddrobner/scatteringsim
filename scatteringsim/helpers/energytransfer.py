from scatteringsim.helpers.crossection import scattering_angle
from scatteringsim.structures import ScatterFrame
import numpy as np

def energy_transfer(e_i, scatter_angle=None):
    m_alpha = 6.646E-27 # kg
    m_proton = 1.6726E-27 # kg

    mev_to_j = 1.602176e-13
    ei_j = e_i * mev_to_j

    if scatter_angle is None:
        scatter_angle = scattering_angle(e_i)
    # just identifying scatter_angle as theta_1 to match the algebra
    # this is in the lab frame
    Theta = scatter_angle

    # and now we determine the final velocity of the alpha in the lab frame

    # first thing we need to do is to work out the velocity of the alpha
    # particle in the lab frame
    # note that the initial proton velocity in the lab frame is 0
    frac_energy = 2*m_alpha*m_proton*(np.cos(Theta) - 1)/((m_alpha + m_proton)**2)
    valpha_i = np.sqrt(2*ei_j/m_alpha)
    valpha_f = valpha_i * frac_energy
    vproton_f = (1-frac_energy)*valpha_i

    eproton_f = (1/mev_to_j)*0.5*m_proton*(vproton_f**2)
    ealpha_f = (1/mev_to_j)*0.5*m_alpha*(valpha_f**2)

    #theta_1 = np.arctan(m_proton*np.sin(Theta)/(np.cos(Theta) - (m_alpha/m_proton)))

    #return (ealpha_f, eproton_f, Theta)
    return ScatterFrame(ealpha_f, eproton_f, Theta)
