from scipy.interpolate import LinearNDInterpolator

import pandas as pd
import numpy as np

from scatteringsim import parameters

def make_cx_interpolator(cx: pd.DataFrame):
    xy = cx[['energy', 'theta']].to_numpy()
    z = cx['cx'].to_numpy()
    return LinearNDInterpolator(xy, z)


def prep_cx(diff_cx: pd.DataFrame) -> dict[pd.DataFrame]:
    """Prepares Cross Section in simulation init

    Args:
        diff_cx (pd.DataFrame): The raw differential cross section read from the csv 

    Returns:
        dict[pd.DataFrame]: The prepared cross sections, key 'total' is the
        integrated cross section, and key 'differential' has the differential
        cross section
    """

    cx = diff_cx

    # converting to radians
    # NOTE this means we need to scale the cx when integrating by 
    # 180/pi due to the transformation
    cx = cx.groupby("energy").filter(lambda x: len(x) > 3)
    cx['theta'] = np.deg2rad(cx['theta'])
    cx = cx[cx['theta'] >= parameters.table_min]
    cx.reset_index(inplace=True, drop=True)

    cx.sort_values(['energy', 'theta'], ignore_index=True, ascending=[True, True], inplace=True)
    cx.reset_index(drop=True, inplace=True)

    # now let's do the inference
    for e in cx['energy'].unique():

        angles = cx[cx['energy'] == e]['theta'].to_numpy()

        # do the lower inference here
        theta_0 = angles.min()
        #cx_0 = np.float32(cx[(cx['energy'] == e) &
        #(cx['theta'] == theta_0)]['cx'])[0]
        cx_0 = np.float32(cx[cx['energy'] == e][cx['theta'] == theta_0]['cx'])[0]
        k_0 = cx_0 - 1/theta_0

        infer_pts_low = np.linspace(parameters.theta_min, theta_0, 5)[:-1]
        for p in infer_pts_low:
            d = pd.DataFrame([(e, p, (1/p) + k_0)], columns=cx.columns)
            cx = pd.concat([cx, d])

        # and now the upper inference
        theta_m = angles.max()
        #cx_m = np.float32(cx[(cx['energy'] == e) & (cx['theta'] == theta_m)]['cx'])[0]
        cx_m = np.float32(cx[cx['energy'] == e][cx['theta'] == theta_m]['cx'])[0]
        km = cx_m - 1/(theta_m)

        infer_pts_up = np.linspace(theta_m, parameters.theta_max, 5)[1:]
        for p in infer_pts_up:
            d = pd.DataFrame([(e, p, (1/p) + km)], columns=cx.columns)
            cx = pd.concat([cx, d])


    # now we re-sort things and fix the indexing
    cx.sort_values(['energy', 'theta'], ignore_index=True, ascending=True, inplace=True)
    cx.reset_index(inplace=True, drop=True)

    temp_es = []
    temp_cx = []
    for e in cx['energy'].unique():
        angles = cx[cx['energy'] == e]['theta']
        dcx = cx[cx['energy'] == e]['cx']
        if len(angles > 3):
            itg = np.trapz(dcx, angles)
            if(itg != 0):
                temp_es.append(e)
                temp_cx.append(itg)
    total_cx = pd.DataFrame(zip(temp_es, temp_cx), columns=['Energy', 'Total'])
    del temp_es
    del temp_cx

    return {'total': total_cx, 'differential': cx}


def gen_inverse_dist(angles, cx):
    x = angles
    y = cx
    np.nan_to_num(y, copy=False)
    cdf_y = np.cumsum(y)
    cdf_y = cdf_y/cdf_y.max()
    #cdf_y = y/y.sum()
    def inverse_cdf(rval):
        return np.interp(rval, cdf_y, x)
    # this is a function
    return inverse_cdf 