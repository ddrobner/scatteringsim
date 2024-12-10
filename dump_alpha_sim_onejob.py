import argparse
from scatteringsim import GPUSim 
from pathlib import Path
from tqdm import tqdm
import pickle

from time import time as current_timestamp

parser = argparse.ArgumentParser(prog='AlphaDumperGPU', description='Dumps Alpha Sim results computed on GPU to disk')
parser.add_argument('-n', '--num_alphas', type=int, help="Number of alphas to simulate. A value of -1 dynamically scales the size to available VRAM.")
parser.add_argument('-b', '--batchsize', type=int, help="Number of alphas to run at once")
parser.add_argument('-o', '--output', type=Path, help="Folder to output runs to")
parser.add_argument('-s', '--stepsize', type=float, default=1E-6)
parser.add_argument('-t', '--stoppingpower', default="stoppingpowers/lab.csv")
parser.add_argument('-c', '--crosssection', default='crossections/combined_new3.csv')
parser.add_argument('-e', '--energy', type=float, default=8.0)

args = parser.parse_args()

#s = ScatterSim(args.energy, args.num_alphas, args.stepsize, 200,
#args.stoppingpower, args.crosssection, proton_factor=0.3)
s = GPUSim(args.energy, args.batchsize, args.stepsize, args.stoppingpower, args.crosssection, proton_factor=0.3)


# dump sim parameters to a metadata file
run_info = {"num_alphas": args.num_alphas, "stepsize": args.stepsize, "stoppingpower": args.stoppingpower, "cross_section": args.crosssection, "energy": args.energy, "timestamp": current_timestamp()}
with open(args.output/"run_info.pkl", 'wb') as f:
   pickle.dump(run_info, f) 

# if batchsize is -1 get number of alphas from gpusim
batchsize = args.batchsize
if batchsize == -1:
    batchsize = s.num_alphas

sim_alphas = args.num_alphas

#if args.num_alphas % args.batchsize != 0:
#    print("WARNING: Rounding number of alphas to closest number that the batchsize divides")
#    sim_alphas = args.num_alphas - (args.num_alphas - batchsize)

num_alphas_run = 0

# this is a bit weird, but I'm doing this here to free the memory of each alpha
# as it gets dumped

progress_bar = tqdm(total=sim_alphas)

run_num = 0
while num_alphas_run < sim_alphas:
    if num_alphas_run + batchsize > sim_alphas:
        s.numalphas = sim_alphas - num_alphas_run
    s.particle_sim()
    if (len(s.particle_results) > 0):
        with open(args.output/f"{str(run_num)}.pkl", 'wb') as f:
            # pickle allows you to load incrementally if you write incrementally :)
            # so I can write one alpha at a time to the file, and load it as such in the future
            pickler = pickle.Pickler(f)
            while len(s.particle_results) > 0:
                pickler.dump(s.particle_results[0])
                s.pop_particle(0)
    s.reset_sim()
    num_alphas_run += batchsize
    progress_bar.update(batchsize)
    run_num += 1

progress_bar.close()