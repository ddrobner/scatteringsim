from scatteringsim import GPUSim
from pathlib import Path
from numpy import histogram
from math import floor, log

import matplotlib.pyplot as plt

from scatteringsim.structures import ScatteredDeposit
from scatteringsim import parameters

import argparse
import pickle

plt.rcParams['figure.figsize'] = (12, 8)

parser = argparse.ArgumentParser(prog='SplitFilePlot', description='Finalizes analysis of split simulation')

parser.add_argument('-i', '--input', type=Path)
parser.add_argument('-q', '--quenching', type=float, default=0.4)
parser.add_argument('-f', '--fill', action=argparse.BooleanOptionalAction)
parser.add_argument('-p', '--file-prefix', type=str)
parser.add_argument('--bins', default=30, type=int)
parser.add_argument('--stats', action='store_true')
parser.add_argument('--low_cut', default=0.0, type=float)

args = parser.parse_args()

with open(args.input/"run_info.pkl", 'rb') as f:
    run_info : dict = pickle.load(f) 

s = GPUSim(run_info['energy'], run_info['num_alphas'], run_info['stepsize'], run_info['stoppingpower'], run_info['cross_section'], proton_factor=args.quenching)

for i_f in args.input.iterdir():
    if i_f.name == "run_info.pkl":
        continue
    with open(i_f, 'rb') as f:
        up = pickle.Unpickler(f)
        while True:
            try:
                p_data = up.load()
                if max(p_data.proton_energies) > args.low_cut:
                    s.add_deposit(p_data)
            except EOFError:
                break

n_bins = args.bins

n_scatters = len(s.particle_results)
s.quenched_spectrum()
if(args.fill):
    s.fill_spectrum()
s.detsim()

def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '{}{}'.format(int(number / k**magnitude), units[magnitude])

print(f"Result len: {len(s.result)}")
counts, bins = histogram(s.result, n_bins, range=(0,max(s.result)))
fig, ax = plt.subplots()
ax.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
plt.yscale('log')
plt.xticks([round(i, 2) for i in bins])
plt.xticks(rotation=60)
plt.ticklabel_format(style='plain', axis='x')
ax.set_title(f"{human_format(int(s.numalphas))} {run_info['energy']}MeV Alphas  Spectrum (Quenching Factor {args.quenching}) {'(Scatters Only)' if not args.fill else ''}")
ax.set_xlabel("Energy (MeV)")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(f"{args.file_prefix}_{str(s.quenching_factor).replace('.', 'p')}.png")

if args.stats:
    total_alphas = run_info['num_alphas']

    scatter_counts_cut = {1:0, 2:0, 3:0}
    scatter_counts = {1:0, 2:0, 3:0}

    for sc in s.particle_results:
        if (len(sc.proton_energies) > 0) and  (sc.proton_energies[0] >= parameters.scatter_e_min):
            scatter_counts_cut[1] += 1
        scatter_counts[1] += 1

        if len(sc.proton_energies) > 1:
            if sc.proton_energies[1] >= parameters.scatter_e_min:
                scatter_counts_cut[2] += 1
            scatter_counts[2] += 1
        
        if len(sc.proton_energies) > 2:
            if sc.proton_energies[2] >= parameters.scatter_e_min:
                scatter_counts_cut[3] += 1
            scatter_counts[3] += 1

    print(f"Info:")
    print(f"Total No Scatter: {run_info['num_alphas'] - scatter_counts[1]}")
    print(f"Total/Fraction Scatter: {scatter_counts[1]} / {scatter_counts[1]/run_info['num_alphas']}")
    print(f"Total/Fraction > 1 Scatters: {scatter_counts[2]} / {scatter_counts[2]/run_info['num_alphas']}")
    print(f"Total/Fraction > 2 Scatters: {scatter_counts[3]}/ {scatter_counts[3]/run_info['num_alphas']}")
    print()
    print(f"Total/Fraction Scatter > 0.95 MeV: {scatter_counts_cut[1]} / {scatter_counts_cut[1]/run_info['num_alphas']}")
    print(f"Total/Fraction > 1 Scatters > 0.95 MeV: {scatter_counts_cut[2]} / {scatter_counts_cut[2]/run_info['num_alphas']}")
    print(f"Total/Fraction > 2 Scatters > 0.95 MeV: {scatter_counts_cut[3]}/ {scatter_counts_cut[3]/run_info['num_alphas']}")