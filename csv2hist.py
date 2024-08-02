import pandas as pd
import numpy as np
import uproot

from hist import Hist

import sys

# run this script by calling python csv2hist.py <input_filename>.csv <output_filename>.root

cx = pd.read_csv(sys.argv[1])
# converting to radians
cx['theta'] = np.deg2rad(cx['theta'])
cx['cx'] = 180/np.pi * cx['cx']




#with uproot.recreate(sys.argv[1]) as f:
    