import argparse
from scatteringsim import GPUSim 
from pathlib import Path
from dataclasses import astuple
import pickle

parser = argparse.ArgumentParser(prog='AlphaDumperGPU', description='Dumps Alpha Sim results computed on GPU to disk')
parser.add_argument('-n', '--num_alphas', type=int)
parser.add_argument('-o', '--output', type=Path)
parser.add_argument('-s', '--stepsize', type=float, default=1E-6)
parser.add_argument('-t', '--stoppingpower', default="stoppingpowers/lab.csv")
parser.add_argument('-c', '--crosssection', default='crossections/combined_new3.csv')
parser.add_argument('-e', '--energy', type=float, default=8.0)

args = parser.parse_args()

#s = ScatterSim(args.energy, args.num_alphas, args.stepsize, 200,
#args.stoppingpower, args.crosssection, proton_factor=0.3)
s = GPUSim(args.energy, args.num_alphas, args.stepsize, 200, args.stoppingpower, args.crosssection, proton_factor=0.3)
s.particle_sim()

# this is a bit weird, but I'm doing this here to free the memory of each alpha
# as it gets dumped
with open(args.output, 'wb') as f:
    # pickle allows you to load incrementally if you write incrementally :)
    # so I can write one alpha at a time to the file, and load it as such in the future
    pickler = pickle.Pickler(f)
    while len(s.alpha_sim) > 0 and len(s.proton_sim) > 0:
        pickler.dump((s.alpha_sim[0], s.proton_sim[0]))
        s.pop_particle(0)