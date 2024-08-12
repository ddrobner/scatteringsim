from typing import Union, Optional
from pathlib import Path
from multiprocessing import Pool
from numpy.typing import ArrayLike
from numpy import histogram

import pickle

from scatteringsim.structures import AlphaEvent, ScatterFrame
from scatteringsim import ScatterSim

def bin_file(file: Union[str, Path], bins: ArrayLike, sim_obj: ScatterSim):
    # bins only the scatters and then we fill in the rest later
    with open(file, 'rb') as f:
        up = pickle.Unpickler(f) 
        results = []
        while True:
            try:
                p_data = up.load()
                scatters = [ScatterFrame(*i) for i in p_data[2]]
                ap = AlphaEvent(p_data[0], p_data[1], scatters)
                results.append(sim_obj.quenched_spectrum(ap))
            except EOFError:
                break
    return histogram(results, bins)
        