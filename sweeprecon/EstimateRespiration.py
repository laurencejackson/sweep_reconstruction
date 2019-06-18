"""
Class containing data and functions for estimating respiration siganl from 3D data

Laurence Jackson, BME, KCL 2019
"""

import numpy as np

import sweeprecon.utilities.PlotFigures as PlotFigures

from scipy.ndimage import gaussian_filter


class EstimateRespiration(object):

    def __init__(self,
                 img,
                 method='body_area',
                 disable_crop_data=False,
                 plot_figures=True
                 ):
        """
        Defines methods for estimating respiration from an image
        :param img: ImageData object
        :param method: method for estimating respiration - only 'body_area' available at the moment
        :param disable_crop_data: flag to indicate whether the data should be cropped
        """

        self._image = img
        self._resp_method = method
        self._plot_figures = plot_figures
        self._disable_crop_data = disable_crop_data

    def run(self):

        if self._resp_method == 'body_area':
            self._method_body_area()
        else:
            raise Exception('\nInvalid respiration estimate method\n')

    def _method_body_area(self):

        if not self._disable_crop_data:
            print('Cropping respiration area...')
            self._auto_crop()

        print('Initialising boundaries...')
        self._initialise_boundaries()

        print('Refining boundaries...')
        self._refine_boundaries()

    def _auto_crop(self, resp_min=0.2, resp_max=0.4, crop_fraction=0.4):
        """
        Finds the best region to crop the image based on the respiratory content of the image
        :param resp_min: lower band of respiration frequency
        :param resp_max: upper band of respiration frequency
        :param crop_fraction: percentage of image to crop
        :return:
        """

        fs = self._image.get_fs()
        sz = self._image.img.shape
        freqmat = np.zeros((sz[0], sz[2]))

        for ln in range(0, sz[0]):
            lineseries = self._image.img[:, ln, :]
            frq = np.fft.fft(lineseries, axis=1)
            freq_line = np.sum(abs(np.fft.fftshift(frq)), axis=0)
            freq_line = (freq_line - np.min(freq_line)) / max((np.max(freq_line) - np.min(freq_line)), 1)
            freqmat[ln, :] = freq_line

        freqmat = gaussian_filter(freqmat, sigma=1.2)
        freqs = np.fft.fftshift(np.fft.fftfreq(sz[2], 1 / fs))
        respii = np.where((freqs > resp_min) & (freqs < resp_max))
        respspectrum = np.sum(freqmat[:, respii[0]], axis=1)
        respspectrum_c = np.convolve(respspectrum, np.ones(int(sz[0] * (crop_fraction * 1.2))), mode='same')
        centerline = np.argmax(respspectrum_c)
        width = int(sz[0] * crop_fraction * 0.5)

        if self._plot_figures:
            PlotFigures.plot_respiration_frequency(freqmat, respii, freqs, centerline, width, sz)

        # crop data to defined limits
        rect = np.array([[centerline - width, 0], [centerline + width, sz[0]]], dtype=int)
        self._image.square_crop(rect=rect)

        # write output
        self._image.write_nii('resp_cropped', prefix='IMG_3D_')

    def _initialise_boundaries(self):
        pass

    def _refine_boundaries(self):
        pass
