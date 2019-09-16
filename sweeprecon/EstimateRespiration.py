"""
Class containing data and functions for estimating respiration siganl from 3D data

Laurence Jackson, BME, KCL 2019
"""

import time
import numpy as np
import copy

import sweeprecon.utilities.PlotFigures as PlotFigures

from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter, morphology, binary_fill_holes
from scipy.signal import medfilt2d
from skimage import restoration, measure, segmentation, exposure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class EstimateRespiration(object):

    def __init__(self,
                 img,
                 write_paths,
                 args,
                 method='body_area',
                 disable_crop_data=False,
                 plot_figures=True,
                 n_threads=0
                 ):
        """
        Defines methods for estimating respiration from an image
        :param img: ImageData object
        :param method: method for estimating respiration - only 'body_area' available at the moment
        :param disable_crop_data: flag to indicate whether the data should be cropped
        """

        self._image = img
        self._image_initialised = copy.deepcopy(img)
        self._image_refined = copy.deepcopy(img)

        self._resp_method = method
        self._plot_figures = plot_figures
        self._disable_crop_data = disable_crop_data

        self.resp_raw = None
        self.resp_trend = None
        self.resp_trace = None

        self._args = args

        self._n_threads = n_threads

        self._write_paths = write_paths

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

        print('Extracting respiration...')
        self._sum_mask_data()
        self._gpr_filter()

    def _auto_crop(self, resp_min=0.15, resp_max=0.4):
        """
        Finds the best region to crop the image based on the respiratory content of the image
        :param resp_min: lower band of respiration frequency
        :param resp_max: upper band of respiration frequency
        :return:
        """

        fs = self._image.get_fs()
        sz = self._image.img.shape
        freqmat = np.zeros((sz[0], sz[2]))
        print('Crop fraction: %f' % self._args.crop_fraction)
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
        respspectrum_c = np.convolve(respspectrum, np.ones(int(sz[0] * (self._args.crop_fraction * 1.2))), mode='same')
        centerline = np.argmax(respspectrum_c)
        width = int(sz[0] * self._args.crop_fraction * 0.5)

        if self._plot_figures:
            PlotFigures.plot_respiration_frequency(freqmat, respii, freqs, centerline, width, sz)

        # crop data to defined limits
        rect = np.array([[centerline - width, 0], [centerline + width, sz[0]]], dtype=int)

        # crop all image copies
        self._image.square_crop(rect=rect)
        self._image_initialised.square_crop(rect=rect)
        self._image_refined.square_crop(rect=rect)

        # write output
        self._image.write_nii(self._write_paths.path_cropped())

    def _initialise_boundaries(self):
        """Initialises body area boundaries"""

        # Filter image data to reduce errors in first contour estimate
        filtered_image = self._process_slices_parallel(self._filter_median,
                                                       self._image.img,
                                                       cores=self._n_threads
                                                       )

        # determine threshold of background data
        # thresh = np.mean(filtered_image[[0, filtered_image.shape[0] - 1], :, :]) + (0.25 * np.std(filtered_image[[0, filtered_image.shape[0] - 1], :, :]))
        thresh = np.max(filtered_image[[0, filtered_image.shape[0] - 1], :, :])

        # apply threshold - always include top and bottom two rows in mask (limited to sagittal at the moment)
        img_thresh = filtered_image <= thresh
        img_thresh[[0, filtered_image.shape[0] - 1], :, :] = 1  # always include most anterior/posterior rows in mask

        # take components connected to anterior/posterior sides
        labels = measure.label(img_thresh, background=0, connectivity=1)
        ac_mask = np.zeros(labels.shape).astype(bool)
        ac_mask[(labels == labels[0, 0, 0]) | (labels == labels[filtered_image.shape[0] - 1, 0, 0])] = True

        for zz in range(0, ac_mask.shape[2]):
            ac_mask[:, :, zz] = binary_fill_holes(ac_mask[:, :, zz])

        ac_mask = morphology.binary_erosion(ac_mask, structure=np.ones((2, 6, 1)))  # erode primarily in FH direction to reduce risk of contour propagation inside the body

        # second pass component connected to outside edges
        ac_mask[[0, ac_mask.shape[0] - 1], :, :] = True
        for zz in range(0, ac_mask.shape[2]):
            labels = measure.label(ac_mask[:, :, zz], background=False, connectivity=1)
            tmp = np.zeros(labels.shape).astype(bool)
            tmp[(labels == labels[0, 0]) | (labels == labels[ac_mask.shape[0] - 1, 0])] = True
            ac_mask[:, :, zz] = tmp

        # write initialised contour data to new image
        self._image_initialised.set_data(ac_mask)
        self._image_initialised.write_nii(self._write_paths.path_initialised_contours())

    def _refine_boundaries(self):
        """Refines body area estimates using Chan-Vese active contour model"""

        # refine segmentation
        print('Contour refinement method: %s' % self._args.ba_method)
        if self._args.ba_method == 'cv':

            filtered_image = self._process_slices_parallel(self._filter_adaptive_hist_eq,
                                                           self._image.img,
                                                           cores=self._n_threads)

            filtered_image = self._process_slices_parallel(self._filter_denoise,
                                                           self._image.img,
                                                           cores=self._n_threads)

            refined_contours = self._process_slices_parallel(self._segment_cv,
                                                             filtered_image,
                                                             self._image_initialised.img,
                                                             cores=self._n_threads)
        elif self._args.ba_method == 'gac':
            # filter/pre-process image
            filtered_image = self._process_slices_parallel(self._filter_adaptive_hist_eq,
                                                           self._image.img,
                                                           cores=self._n_threads)

            filtered_image = self._process_slices_parallel(self._filter_denoise,
                                                           self._image.img,
                                                           cores=self._n_threads)

            filtered_image = self._process_slices_parallel(self._filter_median,
                                                           filtered_image,
                                                           cores=self._n_threads)

            filtered_image = self._process_slices_parallel(self._filter_inv_gauss,
                                                           filtered_image,
                                                           cores=self._n_threads)

            refined_contours = self._process_slices_parallel(self._segment_gac,
                                                             filtered_image,
                                                             self._image_initialised.img,
                                                             cores=self._n_threads)
        else:
            print('invalid body area method')

        # save filtered image
        self._image_refined.set_data(filtered_image)
        self._image_refined.write_nii(self._write_paths.path_filtered_contours())

        # invert mask
        refined_contours = (refined_contours == 0) * 1

        # crop refined boundaries to avoid edge effects
        self._image_refined.set_data(refined_contours)

        # take 10% off upper and lower edges
        cropval = int(0.1 * refined_contours.shape[1])
        rect = np.array([[0 + cropval, 0], [refined_contours.shape[1]-1-cropval, refined_contours.shape[0]]], dtype=int)
        self._image_refined.square_crop(rect=rect)

        # write contour data to file
        self._image_refined.write_nii(self._write_paths.path_refined_contours())

    def _sum_mask_data(self):
        """Sums pixels in refined mask"""
        self.resp_raw = np.squeeze(np.sum(self._image_refined.img, axis=(0, 1)))

    def _gpr_filter(self):
        """Removes low frequency global changes in body area to extract respiration trace only"""

        # define GPR kernel
        kernel = 1.0 * RBF(length_scale=8.0, length_scale_bounds=(4, 20)) \
                 + WhiteKernel(noise_level=50, noise_level_bounds=(10, 1e+3))

        # fit GPR model
        X = np.arange(self.resp_raw.shape[0]).reshape(-1, 1)
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0).fit(X, self.resp_raw)

        # filter signal to extract respiration
        self.resp_trend, y_cov = gp.predict(X, return_cov=True)
        self.resp_trace = self.resp_raw - self.resp_trend

    # ___________________________________________________________________
    # __________________________ Static Methods__________________________

    @ staticmethod
    def _filter_median(img, kernel_size=5):
        """
        Median filter
        :param imgs: slice to filter [2D]
        :param kernel_size: size of median kernel
        :return:
        """
        return medfilt2d(img, [kernel_size, kernel_size])  # median filter more robust to bands in balanced images

    @staticmethod
    def _filter_denoise(img, weight=0.001):
        """
        TV denoising
        :param imgs: slice to denoise [2D]
        :param weight: TV weight
        :return:
        """
        return restoration.denoise_tv_bregman(img, weight=weight)

    @staticmethod
    def _filter_inv_gauss(img, alpha=8, sigma=1.1):
        """
        TV denoising
        :param imgs: slice to denoise [2D]
        :param weight: TV weight
        :return:
        """
        return segmentation.inverse_gaussian_gradient(img, alpha=alpha, sigma=sigma)

    @staticmethod
    def _filter_adaptive_hist_eq(img):
        """
        adaptive histogram equalisation
        :param imgs: slice to equalise [2D]
        :return:
        """
        return exposure.equalize_adapthist(img.astype('uint16'), clip_limit=0.04)

    @staticmethod
    def _segment_cv(img, init_level_set, iterations=100):
        """
        refines initial segmentation contours using chan vese segmentation model
        :param img: slice to segment:
        :param init_level_set: initial level set - starting segmentation
        :param iterations: number of refinement iterations
        :return:
        """
        return segmentation.morphological_chan_vese(img,
                                                    iterations,
                                                    init_level_set=init_level_set,
                                                    smoothing=9,
                                                    lambda1=1.0,
                                                    lambda2=1.0
                                                    )

    @staticmethod
    def _segment_gac(img, init_level_set, iterations=100):
        """
        refines initial segmentation contours using geodesic active contours
        :param imgs: list of 2 images [2D] imgs[0] = slice to segment: imgs[1] = initial level set
        :param iterations: number of refinement iterations
        :return:
        """
        return segmentation.morphological_geodesic_active_contour(img,
                                                                  iterations,
                                                                  init_level_set=init_level_set,
                                                                  smoothing=3,
                                                                  balloon=1.2
                                                                  )

    @staticmethod
    def _process_slices_parallel(function_name, *vols, cores=0):
        """
        Runs a defined function over the slice direction on parallel threads
        :param function_name: function to be performed (must operate on a 2D image)
        :param *vols: image volumes (3D) to pass to function - must be same size
        :param cores: number of cores to run on [default: 1 or max - 1]
        :return:
        """

        # cores defaults to number of CPUs - 1
        if cores is 0:
            cores = max(1, cpu_count() - 1)

        pool = Pool(cores)

        # start timer
        t1 = time.time()

        # convert to list
        vols = list(vols)

        sub_arrays = pool.starmap_async(function_name,
                                        [([vols[v][:, :, zz] for v in range(0, vols.__len__())])
                                         for zz in range(0, vols[0].shape[2])]).get()

        # print function duration info
        print('%s duration: %.1fs [%d processes]' % (function_name.__name__, (time.time() - t1), cores))

        pool.close()
        pool.join()

        # return recombined array
        return np.stack(sub_arrays, axis=2)
