import argparse
from scatteringsim import ScatterSim
from pathlib import Path
from dataclasses import astuple
from numpy import array2string
from scatteringsim.structures import ScatterFrame, AlphaEvent
import pickle

parser = argparse.ArgumentParser(prog='AlphaDumper', description='Dumps Alpha Sim results to disk')
parser.add_argument('-n', '--num_alphas', type=int)
parser.add_argument('-o', '--output', type=Path)
parser.add_argument('-s', '--stepsize', type=float, default=1E-6)
parser.add_argument('-t', '--stoppingpower', default="stoppingpowers/lab.csv")
parser.add_argument('-c', '--crosssection', default='crossections/combined_new3.csv')

args = parser.parse_args()

s = ScatterSim(8.0, args.num_alphas, args.stepsize, 200, args.stoppingpower, args.crosssection, proton_factor=0.3)
s.particle_sim()

fmt_args = {'max_line_width': 100000000, 'threshold': 100000000}

# this is a bit weird, but I'm doing this here to free the memory of each alpha
# as it gets dumped
with open(args.output, 'wb') as f:
    # pickle allows you to load incrementally if you write incrementally :)
    # so I can write one alpha at a time to the file, and load it as such in the future
    pickler = pickle.Pickler(f)
    while len(s.alpha_sim) > 0:
        if(len(s.alpha_sim[0].proton_scatters)) > 0:
            f_data = [s.alpha_sim[0].alpha_path, s.alpha_sim[0].proton_scatters, [astuple(i) for i in s.alpha_sim[0].scatter_energy]]
            pickler.dump(f_data)
        s.pop_particle(0)