"""
Class containing data and functions for estimating respiration siganl from 3D data

Laurence Jackson, BME, KCL 2019
"""

from sweeprecon.io.ImageData import ImageData


class EstimateRespiration(object):

    def __init__(self,
                 img,
                 method='body_area',
                 crop_data=False,
                 ):

        self._img = img
        self._resp_method = method
        self._crop_data = crop_data

    def run(self):

        if self._resp_method == 'body_area':
            self._method_body_area()
        else:
            raise Exception('\nInvalid respiration estimate method\n')

    def _method_body_area(self):

        if self._crop_data:
            print('Cropping respiration area...')
            self._auto_crop()

        print('Initialising boundaries...')
        self._initialise_boundaries()

        print('Refining boundaries...')
        self._refine_boundaries()

    def _auto_crop(self):

        pass

    def _initialise_boundaries(self):
        pass

    def _refine_boundaries(self):
        pass
