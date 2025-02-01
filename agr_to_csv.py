import pandas as pd
import sys

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
            ang = float(line_arr[0].replace(" ",""))
            cx = float(line_arr[1].replace(" ", ""))
            cx_tups.append((ek, ang, cx))

cx_df = pd.DataFrame(cx_tups, columns=['energy','theta','cx'])
cx_df.to_csv(sys.argv[2], index=False)