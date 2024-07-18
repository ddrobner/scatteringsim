import pandas as pd

cx = pd.read_csv("crossections/combined.csv")

# conclusion.... I don't need to do any of this indexing shit

from scipy.interpolate import bisplrep, bisplev

theta = cx['theta']
energy = cx['energy']
diffcx = cx['cx']
tck = bisplrep(theta, energy, diffcx)
print(bisplev(0.2, 8, tck))