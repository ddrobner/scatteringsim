from scipy.interpolate import LinearNDInterpolator, interp1d

import pandas as pd
import numpy as np

import typing

from scatteringsim import parameters

def make_cx_interpolator(cx: pd.DataFrame) -> 'LinearNDInterpolator':
    """Makes an interpolator object for the passed cross section dataframe 

    Args:
        cx (pd.DataFrame): The differential cross section

    Returns:
        LinearNDInterpolator: A scipy LinearNDInterpolator object to interpolate
        the dataframe
    """
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
    # pi/180 due to the transformation
    cx = cx.groupby("energy").filter(lambda x: len(x) > 3)
    cx['theta'] = np.deg2rad(cx['theta'])
    cx = cx[cx['theta'] >= parameters.theta_min]
    cx = cx[cx['energy'] <= parameters.e_max]
    #cx = cx[cx['energy'] % 0.5 == 0]
    cx.reset_index(inplace=True, drop=True)

    #scale mb/sr to b/sr
    cx['cx'] = cx['cx']*0.001

    cx.sort_values(['energy', 'theta'], ignore_index=True, ascending=[True, True], inplace=True)
    cx.reset_index(drop=True, inplace=True)

    temp_es = []
    temp_cx = []
    for e in cx['energy'].unique():
        angles = cx[cx['energy'] == e]['theta']
        dcx = cx[cx['energy'] == e]['cx']
        if len(angles > 3):
            itg = np.trapz(dcx, angles)
            if(itg != 0):
                temp_es.append(e)
                temp_cx.append(itg*(np.pi/180))
                #temp_cx.append(itg/e)

    total_cx = pd.DataFrame(zip(temp_es, temp_cx), columns=['Energy', 'Total'])
    del temp_es
    del temp_cx

    return {'total': total_cx, 'differential': cx}


def gen_inverse_dist(angles: np.ndarray, cx: np.ndarray) -> typing.Callable:
    """Generates inverse distributions for scatter angle sampling for a
    monoenergetic cross section

    Args:
        angles (np.ndarray): Array of angles
        cx (np.ndarray): Array of differential cross section values

    Returns:
        typing.Callable: An interpolator which takes a uniformly sampled random number to give a
        sampled angle
    """
    x = angles
    y = cx
    np.nan_to_num(y, copy=False)
    cdf_y = np.cumsum(y)
    cdf_y = cdf_y/cdf_y.max()
    def inverse_cdf(rval):
        return np.interp(rval, cdf_y, x)
    # this is a function
    return inverse_cdf 