from scatteringsim import ScatterSim

from numpy import histogram

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.formatter.useoffset'] = False

#s = ScatterSim(8.0, 10000, 1E-6, 200, "stoppingpowers/lab.csv", "crossections/combined_new.csv", proton_factor=0.35)
s = ScatterSim(8.0, 30, 1E-6, 200, "stoppingpowers/lab.csv", "crossections/diffcx_2p02MeV.csv", proton_factor=0.35)
s.start()

counts, bins = histogram(s.result, 30)
plt.hist(bins[:-1], bins, weights=counts, rwidth=0.8)
plt.yscale('log')
plt.xticks([round(i, 2) for i in bins])
plt.xticks(rotation=60)
plt.ticklabel_format(style='plain', axis='x')
plt.title("10k Alphas Quenched Spectrum (Smeared, Quenching Factor 0.5, Stepsize $10^{-6}$)")
plt.xlabel("Energy (MeV)")
plt.ylabel("Count")
plt.tight_layout()
#plt.savefig("10k_quenched_smeared_0p35q_new2.jpg")
plt.savefig("test.png")
