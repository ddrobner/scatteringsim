import pickle
import argparse

from numpy import histogram
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)

from pathlib import Path
from scatteringsim.utils import gen_alpha_path

parser = argparse.ArgumentParser()
parser.add_argument('i', '--input', type=Path)
parser.add_argument('-o', '--output', type=Path)
parser.add_argument('-n', '--n_alphas', type=int)
parser.add_argument('-e', '--energy', type=float)
parser.add_argument('-s', '--stoppingpower', type=Path, default=Path("stoppingpowers/lab.csv"))

args = parser.parse_args()

hist_data = []

# in order to actually plot the thing I need to keep everything in memory at
# some point... so there's really no getting around it 
n_scatters = 0
with open(args.input, 'rb') as f:
    up = pickle.Unpickler(f)
    p = up.load()
    for ap in p[0]:
        hist_data.extend(ap)
    for pp in p[1]:
        hist_data.extend(pp)
    n_scatters += 1

for i in range(args.n_alphas - n_scatters):
    hist_data.extend(gen_alpha_path(args.energy, args.stoppingpower, stepsize=1E-6))

counts, bins = histogram(hist_data, 30)
fig, ax = plt.subplots()
ax.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
plt.yscale('log')
plt.xticks([round(i, 2) for i in bins])
plt.xticks(rotation=60)
plt.ticklabel_format(style='plain', axis='x')
ax.set_title("Alpha Simulation Only")
ax.set_xlabel("Energy (MeV)")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(args.output)
fig.clear()