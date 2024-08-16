from scatteringsim import GPUSim

s = GPUSim(8.0, 10000, 1E-6, 200, "stoppingpowers/lab.csv", "crossections/combined_new3.csv")
print(s.particle_sim())