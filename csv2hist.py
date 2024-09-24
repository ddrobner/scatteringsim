import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt

from scatteringsim import GPUSim

import sys

# run this script by calling python csv2hist.py <input_filename>.csv
# <output_filename>.root

# all properties aside from CX are irrelevant here
s = GPUSim(5.3, 1000, 1E-6, 200, "stoppingpowers/lab.csv", sys.argv[1])

# read cx from the GPUSim obj
cx = s.diff_cx

n_bins = 10000
energy_vals = np.linspace(*s.energy_range, n_bins)
angle_vals = np.linspace(*s.angle_range, n_bins)

cx_vals = []

for e in energy_vals:
    for a in angle_vals:
        cx_vals.append(s.interpolator((e, a))) 

h = np.histogram2d(energy_vals, angle_vals, weights=cx_vals, bins=[n_bins, n_bins])

with uproot.recreate(sys.argv[2]) as f:
    f['cx'] = h
    