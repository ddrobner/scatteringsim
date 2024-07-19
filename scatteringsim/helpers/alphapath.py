from scatteringsim.helpers.stoppingpower import stp_interp
import numpy as np
import numpy.typing as npt

def gen_alpha_path(e_0, stp, epsilon=0.1, stepsize=0.001) -> npt.NDArray[np.float64]:
    e_i = e_0
    alpha_path = []
    while e_i > epsilon:
        alpha_path.append(e_i)
        e_i = e_i - stp_interp(e_i, stp)*stepsize
    return np.array(alpha_path)