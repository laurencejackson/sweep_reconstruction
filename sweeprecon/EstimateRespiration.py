"""
Class containing data and functions for estimating respiration siganl from 3D data

Laurence Jackson, BME, KCL 2019
"""

import time
import numpy as np

import sweeprecon.utilities.PlotFigures as PlotFigures
from joblib import delayed, Parallel

from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d
from skimage.restoration import denoise_tv_bregman

# debug import
import matplotlib.pyplot as plt


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

        # variables
        self.resp_raw = None
        self.resp_trend = None
        self.resp_trace = None

    def run(self):
        """Runs chosen respiration estimating method"""

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
        """Initialises body area boundaries"""

        # Filter image data to reduce errors in first contour estimate
        filtered_image = self._process_slices_parallel(self._filter_median, self._image.img, cores=4)

        # determine threshold of background data
        thresh = np.max(filtered_image[[0, filtered_image.shape[0] - 1], :, :]) - (2 * np.std(filtered_image[[0, filtered_image.shape[0] - 1], :, :]))

        # apply threshold - and always include top and bottom two rows in mask
        img_thresh = filtered_image <= thresh
        img_thresh[[0, filtered_image.shape[0] - 1], :, :] = 1  # always include top and bottom two rows in mask



        # write filtered data to image
        self._image.set_data(initialized_image)
        self._image.write_nii('initialised_', prefix='IMG_3D_')

    def _refine_boundaries(self):

        pass

    def _sum_mask_data(self):
        pass

    def _gpr_filter(self):
        pass

    # ___________________________________________________________________
    # __________________________ Static Methods__________________________

    @ staticmethod
    def _filter_median(img, kernel_size=5):
        return medfilt2d(img, [kernel_size, kernel_size])  # median filter more robust to bands in balanced images

    @staticmethod
    def _filter_denoise(image, weight=0.001):
        return denoise_tv_bregman(image, weight=weight)

    @staticmethod
    def _process_slices_parallel(function_name, img, cores=4):
        """
        Runs a defined function over the slice direction on parallel threads
        :param function_name: function to be performed (must operate on a 2D image)
        :param img: image volume (3D)
        :param cores: number of cores to run on [default: 4]
        :return:
        """
        # start timer
        t1 = time.time()

        # run parallel function
        sub_arrays = Parallel(n_jobs=cores)(  # Use n cores
            delayed(function_name)(img[:, :, zz])  # Apply function_name
            for zz in range(0, img.shape[2]))  # For each 3rd dimension

        # print function duration info
        print('%s took: %.1fs' % (function_name.__name__, (time.time() - t1)))

        # return recombined array
        return np.stack(sub_arrays, axis=2)
