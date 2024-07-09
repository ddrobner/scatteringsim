import pandas as pd
from numpy import interp, float64
from pathlib import Path

def read_stopping_power(filename) -> pd.DataFrame:
    stoppingpowers = pd.read_csv(Path(filename))
    # rename cols to make them easier to reference
    stoppingpowers.columns = ["KE", "electron", "nuclear", "total"]
    return stoppingpowers

def stp_interp(energy:float, stp: pd.DataFrame) -> float64:
    # NOTE this assumes that the stopping powers are sorted
    # we get them this way from ASTAR so it's not an issue, but we can fix that if need be
    for k in stp.index:
        if energy <= stp["KE"][k+1] and energy >= stp["KE"][k]:
            return interp(energy, list(stp["KE"]), list(stp["total"]))
        else:
            return ((list(stp["total"])[-1])/list(stp["KE"])[-1])*energy