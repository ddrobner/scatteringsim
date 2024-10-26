from scatteringsim import GPUSim

s = GPUSim(8.0, 100, 1E-6, "stoppingpowers/lab.csv", "crossections/combined_new3.csv")
s.particle_sim()
s.quenched_spectrum()
s.detsim()

print(s.result())
