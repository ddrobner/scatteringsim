from numpy import pi

# this is the lower bound any time anything goes to zero (ie. alpha
# energy)
# really speeds things up and also when it has that little energy the
# scatters are going to be negligible
table_min = pi/4
theta_min = pi/12
theta_max = pi

epsilon = 0.1

nhit = 200