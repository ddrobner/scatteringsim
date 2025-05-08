from numpy import pi

# this is the lower bound any time anything goes to zero (ie. alpha
# energy)
# really speeds things up and also when it has that little energy the
# scatters are going to be negligible
theta_min = pi/12
theta_max = pi

e_proton_min = 0.2

e_max = 9.0

epsilon = 0.1

nhit = 200
max_particle_scatters = 3
scatter_e_min = 0.95