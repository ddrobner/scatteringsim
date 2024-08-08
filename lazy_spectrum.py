import pickle
import argparse
import multiprocessing as mp

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)

from numpy import histogram
from math import log, floor

from scatteringsim import ScatterSim
from scatteringsim.structures import ScatterFrame, AlphaEvent

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-q', '--quenching', type=float)
parser.add_argument('-n', '--num_alphas', type=int)

args = parser.parse_args()

s = ScatterSim(8, 10000, 1E-6, 200, "stoppingpowers/lab.csv", "crossections/combined_new3.csv", proton_factor=args.quenching)

def quenching_wrapper(alphaevent):
    return s.quenched_spectrum(alphaevent)


with open(args.input, 'rb') as f:
    with mp.Pool() as p:
        unpickler = pickle.Unpickler(f)
        processes = []
        while True:
            try:
                p_data = unpickler.load()
                scatters = [ScatterFrame(*i) for i in p_data[2]]
                ap = AlphaEvent(p_data[0], p_data[1], scatters)
                t = p.apply_async(quenching_wrapper, args=(ap,))
                processes.append(t)
            except EOFError:
                break
        p.close()
        p.join()
        results = [t.get() for t in processes]

k = [s.append_q(i) for i in results]
s.fill_spectrum(len(results))

fig, ax = plt.subplots()

def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '{}{}'.format(int(number / k**magnitude), units[magnitude])

counts, bins = histogram(s.result, 30)
ax.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
plt.yscale('log')
plt.xticks([round(i, 2) for i in bins])
plt.xticks(rotation=60)
plt.ticklabel_format(style='plain', axis='x')
ax.set_title(f"{human_format(int(s.numalphas))} Alphas Spectrum (Quenching Factor {args.quenching}, Stepsize {s.step_size_latex})")
ax.set_xlabel("Energy (MeV)")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(f"10k_biglims_{str(s.quenching_factor).replace('.', 'p')}.png")
fig.clear()
