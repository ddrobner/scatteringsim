import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt

from scatteringsim import GPUSim
#from matplotlib.colors import LogNorm

#from scipy.interpolate import LinearNDInterpolator

#from matplotlib import cm
#import matplotlib.colors as mcolors
import itertools

import sys

# run this script by calling python csv2hist.py <input_filename>.csv
# <output_filename>.root

# all properties aside from CX are irrelevant here
s = GPUSim(5.3, -1, 1E-6, 200, "stoppingpowers/lab.csv", sys.argv[1])

# read cx from the GPUSim obj
cx = s.diff_cx
print(cx)
print(cx[cx['theta']==np.pi/12])

n_bins = 1000
#energy_vals = np.linspace(*s.energy_range, n_bins)
#angle_vals = np.linspace(*s.angle_range, n_bins)
yarr = np.linspace(s.angle_range[0]+0.0001, s.angle_range[1]-0.0001, n_bins)
xarr = np.linspace(*s.energy_range, n_bins)

comb_vals = []

cx_vals = []
for x in range(len(xarr)):
    for y in range(len(yarr)):
        cx_vals.append((s.interpolator(xarr[x], yarr[y])))
        comb_vals.append((xarr[x], yarr[y]))

#Xt, Yt = np.meshgrid(xarr, yarr)
cx_vals = np.array(cx_vals)
np.nan_to_num(cx_vals, copy=False, nan=0.0)

for i in cx_vals:
    if i < 0:
        print(i)

hist_args = ([i[0] for i in comb_vals], [i[1] for i in comb_vals])
hist_kwargs = {"weights": cx_vals, "bins":(n_bins, n_bins)}


"""
# --------------------------------------
xy = cx[['energy', 'theta']].to_numpy()
z = cx['cx'].to_numpy()
cx_interp = LinearNDInterpolator(xy, z)

ar = s.angle_range
er = s.energy_range

#yarr = np.linspace(infer_theta_min+0.01, ar[1], 1000)
#yarr = np.linspace(infer_theta_min+0.01, infer_theta_max-0.01, 1000)
#xarr = np.linspace(*er, 1000)
#zarr = np.array([cx_interp(k) for k in zip(xarr, yarr)])
Zt = np.zeros((xarr.shape[0], yarr.shape[0]))

for x in range(len(xarr)):
    for y in range(len(yarr)):
        Zt[y][x] = cx_interp((xarr[x], yarr[y]))

np.nan_to_num(Zt, copy=False, nan=0.0)
#np.reshape(zarr, (xarr.shape[0], yarr.shape[0]))
print(Zt.shape)

Xt, Yt = np.meshgrid(xarr, yarr)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Xt, Yt, Zt, cmap=cm.jet, norm=mcolors.LogNorm(vmin=abs(Zt.min())+0.001, vmax=abs(Zt.max())), linewidth=0)
#surf = ax.plot_surface(Xt, Yt, Zt, cmap=cm.jet, linewidth=0)
ax.view_init(azim=30)
ax.set_xlabel("Energy (MeV)")
ax.set_ylabel(r"$\theta (rad)$")
ax.set_zlabel(r"$\sigma$ (Barns)")
ax.set_title(r"Differential Cross Section ($\theta_{min}=\pi/12$, $\theta_{max} = \pi$)")
ax.set_box_aspect(None, zoom=0.85)
#fig.show()
cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
cbar.ax.set_xlabel(r"$\sigma$ (Barns)")
plt.tight_layout()
plt.show()
#===============================================

Xt, Yt = np.meshgrid(xarr, yarr)
"""



h = np.histogram2d(*hist_args, **hist_kwargs)
#hs, xedg, yedg = np.histogram2d(*hist_args, **hist_kwargs)
#plt.imshow(hs, interpolation='none', origin='lower', extent=[xedg[0], xedg[-1], yedg[0], yedg[-1]], norm=LogNorm(vmin=cx_vals.min()+0.001, vmax=cx_vals.max()))
#plt.show()

with uproot.recreate(sys.argv[2]) as f:
    f['cx'] = h