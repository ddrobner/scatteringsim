import numpy as np

m_alpha = np.float128(6.646E-27) # kg
m_proton = np.float128(1.6726E-27) # kg
mev_to_j = np.float32(1/6.242E12)
# maybe floating point division error?
j_to_mev = np.float32(6.242E12)

lab_density = 0.8562
proton_scaling_factor = 25
lab_mol_wt = 234