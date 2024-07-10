from scatteringsim import start_sim, quenched_spectrum_multithread, read_stopping_power

import matplotlib.pyplot as plt
import numpy as np

import random

stp = read_stopping_power("stoppingpowers/lab.csv")

sim_data = start_sim(8, 10000, stp, stepsize=1E-6)
q_spec = quenched_spectrum_multithread(sim_data, 0.4)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.formatter.useoffset'] = False
nhit = 200 

q_spec_smeared = []

for i in range(len(q_spec)):
    e_i = q_spec[i]
    e_f = (random.gauss(e_i*nhit, np.sqrt(e_i*nhit)))/nhit
    q_spec_smeared.append(e_f)

counts, bins = np.histogram(q_spec_smeared, 30)
plt.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
plt.yscale('log')
plt.xticks([round(i, 2) for i in bins])
plt.xticks(rotation=60)
plt.ticklabel_format(style='plain', axis='x')
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.title("10k Alphas Quenched Spectrum (Smeared, Quenching Factor 0.5, Stepsize $10^{-6}$)")
plt.xlabel("Energy (MeV)")
plt.ylabel("Count")
plt.xlim((0.6, 2.5))
plt.tight_layout()
plt.savefig("10k_quenched_smeared_0p4q_new2.jpg")