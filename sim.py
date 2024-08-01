from scatteringsim import ScatterSim

from numpy import histogram
from math import log, floor

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.formatter.useoffset'] = False

s = ScatterSim(8.0, 10000, 1E-6, 200, "stoppingpowers/lab.csv", "crossections/diffcx_2p02MeV.csv", proton_factor=0.3)
s.particle_sim()
q_factors = [0.3, 0.4, 0.5]

# small function to format the number of alphas
def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '{}{}'.format(int(number / k**magnitude), units[magnitude])

fig, ax = plt.subplots()

for q in q_factors:
    s.quenching_factor = q
    s.recompute_spectrum()

    
    counts, bins = histogram(s.result, 30)
    ax.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
    plt.yscale('log')
    plt.xticks([round(i, 2) for i in bins])
    plt.xticks(rotation=60)
    plt.ticklabel_format(style='plain', axis='x')
    ax.set_title(f"{human_format(int(s.numalphas))} Alphas Spectrum (Quenching Factor {q}, Stepsize {s.step_size_latex})")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"10k_biglims_{str(q).replace('.', 'p')}.png")
    fig.clear()
