import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt

from scatteringsim import GPUSim

import itertools

import sys

# run this script by calling python csv2hist.py <input_filename>.csv
# <output_filename>.root

# all properties aside from CX are irrelevant here
s = GPUSim(5.3, 1000, 1E-6, 200, "stoppingpowers/lab.csv", sys.argv[1])

# read cx from the GPUSim obj
cx = s.diff_cx

n_bins = 1000
energy_vals = np.linspace(*s.energy_range, n_bins)
angle_vals = np.linspace(*s.angle_range, n_bins)

comb_vals = []

cx_vals = []
for e in energy_vals:
    for a in angle_vals:
        cx_vals.append(s.interpolator((e, a)))
        comb_vals.append((e, a))

cx_vals = np.array(cx_vals)
np.nan_to_num(cx_vals, copy=False, nan=0.0)
h = np.histogram2d(np.array([i[0] for i in comb_vals]), np.array([i[1] for i in comb_vals]), weights=cx_vals, bins=[n_bins, n_bins])

with uproot.recreate(sys.argv[2]) as f:
    f['cx'] = h