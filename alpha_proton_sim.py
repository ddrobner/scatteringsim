from scatteringsim import start_sim, quenched_spectrum_multithread, read_stopping_power

stp = read_stopping_power("stoppingpowers/lab.csv")

sim_data = start_sim(8, 500, stp, nbins=50, stepsize=1E-6)
q_spec = quenched_spectrum_multithread(sim_data, 0.4)
print(q_spec)