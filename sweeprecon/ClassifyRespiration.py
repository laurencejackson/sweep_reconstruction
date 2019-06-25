"""
Class containing data and functions for classifying respiration states from a 1D respiration signal

Laurence Jackson, BME, KCL 2019
"""

import numpy as np

from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess


class ClassifyRespiration(object):

    def __init__(self, resp_trace):
        """init object"""
        self._resp_trace = resp_trace
        self.index = np.zeros([self._resp_trace.shape[0]])

    def classify_states(self, nstates):
        """Classify every image in nstates[int] states"""

        # find max/min resp positions
        max_positive, max_negative = self.bounds_peaks()

        bounds = np.zeros([self._resp_trace.shape[0], nstates + 1])
        for ww in range(0, nstates - 1):
            bounds[:, ww + 1] = (((max_positive - max_negative) / nstates) * (ww + 1)) + max_negative

        bounds[:, 0] = max_negative
        bounds[:, nstates] = max_positive

        print('Assigning states ... ')
        for ww in range(0, nstates):
            rowidx = np.where(np.logical_and(self._resp_trace >= bounds[:, ww], self._resp_trace <= bounds[:, ww + 1]))
            self.index[rowidx] = ww + 1

    def bounds_peaks(self):
        """Finds max/min bounds by tracing along peak locations then smoothing"""

        peaks_pos, _ = find_peaks(self._resp_trace, height=0)
        peaks_neg, _ = find_peaks(-1 * self._resp_trace, height=0)

        xx = np.linspace(0, len(self._resp_trace) - 1, len(self._resp_trace))

        conn_pos = np.interp(xx, peaks_pos, self._resp_trace[peaks_pos] + (0.3 * self._resp_trace[peaks_pos]))
        conn_neg = np.interp(xx, peaks_neg, self._resp_trace[peaks_neg] + (0.3 * self._resp_trace[peaks_neg]))

        smooth = len(self._resp_trace) * 30e-5
        lim_pos = lowess(conn_pos, xx, is_sorted=True, frac=smooth)[:, 1]
        lim_neg = lowess(conn_neg, xx, is_sorted=True, frac=smooth)[:, 1]

        return lim_pos, lim_neg
