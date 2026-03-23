# libs.py

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import argparse
import matplotlib.pyplot as plt
import textwrap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import uuid
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from scipy import signal
from pycwt import wavelet

def findtarget(file_path, columnnames):
    for pathsplit in file_path.split("_"):
        if pathsplit in columnnames:
            return pathsplit
    return None

def wavelet_filter(df, dt=1, freq_band=[50, 70], mother=wavelet.Morlet(6), detrend=True):
    """
    Performs wavelet filtering on a specified df in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the timeseries data.
        dt (float, optional): The time step of the data (default: 1).
        freq_band (list, optional): The lower and upper frequency cut-off for the bandpass filter (default: [50, 70]).
        mother (pycwt.wavelet, optional): The wavelet function to use (default: Morlet(6)).
        detrend (bool, optional): Whether to detrend the timeseries before filtering (default: True).

    Returns:
        pd.Series: The filtered timeseries as a Pandas Series.
    """

    timeseries = df.values

    if detrend:
        # Linear detrending
        times = np.arange(len(timeseries))
        p = np.polyfit(times, timeseries, 1)
        trend = np.polyval(p, times)
        timeseries_detrended = timeseries - trend
    else:
        timeseries_detrended = timeseries

    # Normalize the detrended timeseries
    dat_norm = timeseries_detrended / timeseries_detrended.std()

    # Wavelet parameters
    dj = 1/12  # Twelve sub-octaves per octave
    s0 = dt  # Starting scale
    J = 7/dj  # Seven powers of two with dj sub-octaves

    # Compute wavelet transform
    wavelet_transform, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
    period = 1. / freqs

    # Select data within the specified frequency band
    sel = np.where((period >= freq_band[0]) & (period <= freq_band[-1]))[0]
    Cdelta = mother.cdelta

    # Reconstruct filtered data
    reconstruct = np.sum(wavelet_transform.real[sel, :] / np.sqrt(scales[sel, None]), axis=0)
    reconstruct = (dj * np.sqrt(dt) / (Cdelta * np.pi**(-1/4))) * reconstruct * timeseries_detrended.std()

    # Add back the trend if detrended
    if detrend:
        reconstruct += trend

    return pd.Series(reconstruct, index=df.index)

def wavelet_filter_dataframe(df, isseries, dt=1, freq_band=[50, 70], mother=wavelet.Morlet(6), detrend=True):
    """
    Performs wavelet filtering on multiple columns of a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the timeseries data.
        dt (float, optional): The time step of the data (default: 1).
        freq_band (list, optional): The lower and upper frequency cut-off for the bandpass filter (default: [50, 70]).
        mother (pycwt.wavelet, optional): The wavelet function to use (default: Morlet(6)).
        detrend (bool, optional): Whether to detrend the timeseries before filtering (default: True).

    Returns:
        pd.DataFrame: The DataFrame with filtered columns.
    """

    if(isseries):
        return wavelet_filter(df, dt, freq_band=freq_band, mother=mother, detrend=detrend)
    else:
        columns = df.columns
        filtered_df = df.copy()

        for column in columns:
            filtered_df[column] = wavelet_filter(df[column], dt, freq_band, mother, detrend)

            

        return filtered_df
