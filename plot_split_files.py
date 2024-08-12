import argparse

from pathlib import Path
from scatteringsim import ScatterSim, bin_file
from math import floor, log

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_alphas', type=int, help="Number of alphas per file")
parser.add_argument('-q', '--quenching', type=float)
parser.add_argument('-i', '--input', help="Must be directory")
parser.add_argument('-e', '--energy', type=float, default=8.0)
parser.add_argument('-b', '--bins', type=int, default=30)
parser.add_argument('-l', '--bin_lower', type=float, default=0.0)
parser.add_argument('-u', '--bin_upper', type=float, default=2.5)

args = parser.parse_args()

input_dir = Path(args.input)

if not input_dir.is_dir():
    raise SystemExit("Input Must Be A Directory!") 

# set up the simulation and stuff
s = ScatterSim(args.energy, args.n_alphas, 1E-6, 200, "stoppingpowers/lab.csv", "crossections/combined_new3.csv", proton_factor=args.quenching)

# initialize our histogram arrays
counts = np.zeros(args.bins)
bins = np.linspace(args.bin_lower, args.bin_upper, args.bins+1, endpoint=True)

def quenching_wrapper(alphaevent):
    return s.quenched_spectrum(alphaevent)

for in_f in input_dir.iterdir():
    i_counts, i_b = bin_file(in_f, bins, s)
    counts += i_counts

s.fill_spectrum(np.sum(counts))
c_alpha, b_alpha = np.histogram(s.result, bins)
counts += c_alpha

def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '{}{}'.format(int(number / k**magnitude), units[magnitude])

fig, ax = plt.subplots()

ax.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
plt.yscale('log')
plt.xticks([round(i, 2) for i in bins])
plt.xticks(rotation=60)
plt.ticklabel_format(style='plain', axis='x')
ax.set_title(f"{human_format(int(s.numalphas))} Alphas Spectrum (Quenching Factor {args.quenching}, Stepsize {s.step_size_latex})")
ax.set_xlabel("Energy (MeV)")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(f"100k_biglims_{str(s.quenching_factor).replace('.', 'p')}.png")
fig.clear()