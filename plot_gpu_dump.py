from scatteringsim import GPUSim
from pathlib import Path
from numpy import histogram
from math import floor, log

import matplotlib.pyplot as plt

import argparse
import pickle

plt.rcParams['figure.figsize'] = (12, 8)

parser = argparse.ArgumentParser(prog='SplitFilePlot', description='Finalizes analysis of split simulation')

parser.add_argument('-n', '--num_alphas', type=int)
parser.add_argument('-i', '--input', type=Path)
parser.add_argument('-s', '--stepsize', type=float, default=1E-6)
parser.add_argument('-t', '--stoppingpower', default="stoppingpowers/lab.csv")
parser.add_argument('-c', '--crosssection', default='crossections/combined_new3.csv')
parser.add_argument('-e', '--energy', type=float, default=8.0)
parser.add_argument('-q', '--quenching', type=float, default=0.4)
parser.add_argument('-f', '--fill', type=bool, default=True)
parser.add_argument('-p', '--file-prefix', type=str)

args = parser.parse_args()

s = GPUSim(args.energy, args.num_alphas, args.stepsize, 200, args.stoppingpower, args.crosssection, proton_factor=args.quenching)

for i_f in args.input.iterdir():
    with open(i_f, 'rb') as f:
        up = pickle.Unpickler(f)
        while True:
            try:
                p_data = up.load()
                s.add_particle(p_data[0], p_data[1])
            except EOFError:
                break

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
counts, bins = histogram(s.result, 30)
fig, ax = plt.subplots()
ax.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
plt.yscale('log')
plt.xticks([round(i, 2) for i in bins])
plt.xticks(rotation=60)
plt.ticklabel_format(style='plain', axis='x')
ax.set_title(f"{human_format(int(s.numalphas))} {args.energy}MeV Alphas  Spectrum (Quenching Factor {args.quenching}) {'(Scatters Only)' if args.scatter else ''}")
ax.set_xlabel("Energy (MeV)")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(f"{args.file_prefix}_{str(s.quenching_factor).replace('.', 'p')}.png")
