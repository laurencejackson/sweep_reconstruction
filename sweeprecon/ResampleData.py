"""
Class containing data and functions for re-sampling 3D data into respiration resolved volumes

Laurence Jackson, BME, KCL 2019
"""

import numpy as np


class ResampleData(object):

    def __init__(self,
                 image,
                 states,
                 resolution='isotropic',
                 interp_method='fast_linear'
                 ):

        self._image = image
        self._resolution = resolution
        self._states = states
        self._interp_method = interp_method
        self._nstates = np.max(states)

    def run(self):

        # initialise output volume
        self._init_vols()

        if self._interp_method == 'fast_linear':
            self._interp_fast_linear()
        else:
            raise Exception('\nInvalid data re-sampling method\n')

    def _init_vols(self):
        """pre-allocates memory for interpolated volumes"""
        self._img_4d = np.zeros([self._image.shape[0],
                                 self._image.shape[1],
                                 int((self._image.header['pixdim'][3] * self._image.header['dim'][3]) / self._resolution),
                                 self._nstates]
                                )

    def _interp_fast_linear(self):

        pass
