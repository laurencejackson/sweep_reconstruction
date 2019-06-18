"""
List of functions to produce figures describing output of reconstruction toolkit

Laurence Jackson, BME, KCL, 2019
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_respiration_frequency(freqmat, respii, freqs, centerline, width, sz, name='respiration_frequency'):

    plt.imshow(freqmat[:, respii[0]], aspect='auto')
    plt.imshow(freqmat, vmax=np.mean(freqmat) * 1.5, extent=(freqs[0], freqs[sz[2] - 1], sz[0], 0), aspect='auto')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Row index')
    plt.axhline(centerline - width, color='black')
    plt.axhline(centerline, color='red')
    plt.axhline(centerline + width, color='black')

    plt.savefig('fig_' + name, dpi=300)
