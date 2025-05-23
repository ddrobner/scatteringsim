import pandas as pd
import sys
import numpy as np

cx_tups = []

with open(sys.argv[1], 'r') as f:
    ek = 0
    for l in f:
        if "E_kin" in l or ek == 0:
            try:
                ek = float(l.rstrip().split(" ")[1])
            except:
                print(l.rstrip().split("  "))
        elif "&" in l:
            ek = 0
        else:
            line_arr = l.rstrip().split("   ")
            ang = np.deg2rad(float(line_arr[0].replace(" ","")))
            #ang = 2*np.rad2deg(np.arccos(1 - ang/(2*np.pi)))
            ang = np.rad2deg(2*np.pi*np.arcsin(ang/(2*np.pi)))
            cx = float(line_arr[1].replace(" ", ""))
            # convert cx from mb to b
            cx = 2*np.pi*0.001*cx
            cx_tups.append((ek, ang, cx))

cx_df = pd.DataFrame(cx_tups, columns=['energy','theta','cx'])
cx_df.to_csv(sys.argv[2], index=False)