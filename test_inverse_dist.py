from scatteringsim import GPUSim

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 8)

s = GPUSim(5.3, 10000, 1E-6, 200, "stoppingpowers/lab.csv", "crossections/combined_new3.csv", proton_factor=0.5)


samp = s.gen_dist_samples(5.3, 1E6)
sample_counts, sample_bins = np.histogram(samp, 50)

dist_points = s.get_cx(5.3, 1000)
dist_points_norm = (dist_points/np.max(dist_points))*np.max(sample_counts)

fig, ax = plt.subplots()

ax.hist(sample_bins[:-1], sample_bins, weights=sample_counts)
ax.plot(np.linspace(*s.angles, 1000), dist_points)
ax.set_xlabel("Theta")
ax.set_ylabel("Dist (arb.)")

fig.tight_layout()
fig.savefig("distcomp.png")
