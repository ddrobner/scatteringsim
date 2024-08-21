from scatteringsim import GPUSim

s = GPUSim(5.3, 10000, 1E-6, 200, "stoppingpowers/lab.csv", "crossections/combined_new3.csv")
s.particle_sim()

print(len(s.result))
