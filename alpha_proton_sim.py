from scatteringsim import start_sim, quenched_spectrum_multithread, read_stopping_power, smeared_spectrum, compute_interp_cx 

import matplotlib.pyplot as plt
import numpy as np

stp = read_stopping_power("stoppingpowers/lab.csv")

tck = compute_interp_cx("crossections/combined.csv")

sim_data = start_sim(8, 50, stp, tck, stepsize=1E-5)
q_spec = quenched_spectrum_multithread(sim_data, 0.35)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.formatter.useoffset'] = False
nhit = 300 

q_spec_smeared = []
q_spec_smeared = smeared_spectrum(q_spec, nhit)

counts, bins = np.histogram(q_spec_smeared, 30)
plt.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
plt.yscale('log')
plt.xticks([round(i, 2) for i in bins])
plt.xticks(rotation=60)
plt.ticklabel_format(style='plain', axis='x')
#plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.title("10k Alphas Quenched Spectrum (Smeared, Quenching Factor 0.5, Stepsize $10^{-6}$)")
plt.xlabel("Energy (MeV)")
plt.ylabel("Count")
#plt.xlim((0.6, 2.5))
plt.tight_layout()
plt.savefig("10k_quenched_smeared_0p35q_new2.jpg")
