from typing import Union, Optional
from pathlib import Path
from multiprocessing import Pool
from numpy.typing import ArrayLike
from numpy import histogram

import pickle

from scatteringsim.structures import AlphaEvent, ScatterFrame
from scatteringsim import ScatterSim

def bin_file(file: Union[str, Path], bins: ArrayLike, sim_obj: ScatterSim, quenching_wrapper: callable):
    with open(file, 'rb') as f:
        up = pickle.Unpickler(f) 
        processes = []
        with Pool() as p:
            while True:
                try:
                    p_data = up.load()
                    scatters = [ScatterFrame(*i) for i in p_data[2]]
                    ap = AlphaEvent(p_data[0], p_data[1], scatters)
                    t = p.apply_async(quenching_wrapper, args=(ap,))
                    processes.append(t) 
                except EOFError:
                    break
        p.close()
        p.join()
        results = [t.get() for t in processes]

    k = [sim_obj.append_q(i) for i in results]
    sim_obj.fill_spectrum(len(results))
    return histogram(sim_obj.result, bins)
        