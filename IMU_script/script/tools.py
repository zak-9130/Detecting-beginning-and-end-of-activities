#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 05:37:04 2021

@author: zackinho91
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy
import signal
import seaborn as sns
import matplotlib

def filter_data(data, sample_rate, low_pass = 2, order = 4):
    """Applique un filtre basique passe bas (butter, ordre 4) pour lisser le signal"""

    # TODO: Ameliorer la gestion des NaN. Pour le moment les NaN A la fin de l'enregistrement sont supprimé et les autres sont remplacés par 0.
    try:
        data = np.array(data)
        low_pass = low_pass/(sample_rate/2)
        b, a = scipy.signal.butter(order, low_pass, btype = 'lowpass')
        data_filtered = np.zeros(data.shape)
        if len(data.shape) > 1:
            #Supression des nan en fin de signal
            while np.isnan(data[-1,:]).any():
                data = data[:-1,:]
                data_filtered = data_filtered[:-1,:]
            data = np.nan_to_num(data)
            for col in range(data.shape[1]):
                data_filtered[:,col] = signal.filtfilt(b, a, data[:,col])
        else:
            while np.isnan(data[-1]):
                data = data[:-1]
                data_filtered = data_filtered[:-1]
            data = np.nan_to_num(data)

            data_filtered =scipy.signal.filtfilt(b, a, data)
        if np.isnan(data_filtered).any():
            print("Output contain NaN")

    except ValueError:
        print("Value Error, output will be full of 0")
        data_filtered =  np.full(1000,0)
    return data_filtered


def lag_finder(y1, y2, sr):
    n = len(y1)

    corr = scipy.signal.correlate(y2, y1, mode='same') / np.sqrt(scipy.signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('angular velocity is ' + str(delay) + ' behind acceleration')

    plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()