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
                s.add_deposit(p_data)
            except EOFError:
                break

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
counts, bins = histogram(s.result, 30, range=(0,max(s.result)))
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

total_alphas = run_info['num_alphas']
scattered = len(s.particle_results)
no_scatter = total_alphas - scattered
double_scatter = 0
triple_scatter = 0

for sc in s.particle_results:
    if len(sc.proton_energies) > 1:
        double_scatter += 1
    
    if len(sc.proton_energies) > 2:
        triple_scatter += 1
    
    

print(f"Info:")
print(f"Total No Scatter: {no_scatter}")
print(f"Total/Fraction Scatter: {scattered} / {scattered/run_info['num_alphas']}")
print(f"Total/Fraction > 1 Scatters: {double_scatter} / {double_scatter/run_info['num_alphas']}")
print(f"Total/Fraction > 2 Scatters: {triple_scatter}/ {triple_scatter/run_info['num_alphas']}")