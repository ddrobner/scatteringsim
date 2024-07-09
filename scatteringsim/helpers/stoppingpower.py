import pandas as pd
from numpy import interp, float64

def stp_interp(energy:float, stp: pd.DataFrame) -> float64:
    # NOTE this assumes that the stopping powers are sorted
    # we get them this way from ASTAR so it's not an issue, but we can fix that if need be
    for k in stp.index:
        if energy <= stp["KE"][k+1] and energy >= stp["KE"][k]:
            return interp(energy, list(stp["KE"]), list(stp["total"]))
        else:
            return ((list(stp["total"])[-1])/list(stp["KE"])[-1])*energy