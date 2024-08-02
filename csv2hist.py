import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt

import sys

# run this script by calling python csv2hist.py <input_filename>.csv <output_filename>.root

cx = pd.read_csv(sys.argv[1])
# converting to radians
cx['theta'] = np.deg2rad(cx['theta'])
# TODO figure this out
#cx['cx'] = 180/np.pi * cx['cx']

n_xbins = len(cx['energy'].unique())
n_ybins = len(cx['theta'].unique())

h = np.histogram2d(cx['energy'].to_numpy(), cx['theta'].to_numpy(), weights=cx['cx'].to_numpy(), bins=[n_xbins, n_ybins])

with uproot.recreate(sys.argv[2]) as f:
    f['cx'] = h
    