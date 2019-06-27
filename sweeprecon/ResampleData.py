"""
Class containing data and functions for re-sampling 3D data into respiration resolved volumes

Laurence Jackson, BME, KCL 2019
"""

import copy
import time
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from joblib import delayed, Parallel, cpu_count


class ResampleData(object):

    def __init__(self,
                 image,
                 states,
                 slice_locations,
                 write_paths,
                 resolution='isotropic',
                 interp_method='fast_linear'
                 ):
        """
        initilises ResampleData
        :param image: ImageData object containing input image
        :param states: vector mapping each slice of image to a respiration state
        :param slice_locations: vector mapping each slice to a spatial position relative to origin
        :param write_paths: WritePaths object containing paths to write output
        :param resolution: resolution of output - defualts to 'isotropic' but also takes a float [mm]
        :param interp_method: method for interpolation, 'fast_linear' or 'gpr' (slower and smoother)
        """

        self._image = image
        self._image_4d = copy.deepcopy(image)
        self._states = states
        self._slice_locations = slice_locations
        self._write_paths = write_paths
        self._interp_method = interp_method
        self._resolution = resolution
        self._nstates = np.max(states)

    def run(self):
        """Runs chosen re-sampling scheme """

        # initialise output volume
        self._get_query_points()
        self._init_vols()

        # perform chosen interpolation
        if self._interp_method == 'fast_linear':
            self._interp_fast_linear()
        if self._interp_method == 'gpr':
            self._interp_gpr()
        else:
            raise Exception('\nInvalid data re-sampling method\n')

        # write output
        self._write_resampled_data()

    def _write_resampled_data(self):
        """Saves re-sampled image"""
        # TODO: can only correct for interpolation in z at the moment this is all i need for now but update for xy soon
        self._image_4d.nii.affine[:, 2] = self._image.nii.affine[:, 2] * (self._dxyz[2] / self._image.nii.header['pixdim'][3])

        self._image_4d.set_data(self._img_4d)
        self._image_4d.write_nii(self._write_paths.path_interpolated_4d)

    def _init_vols(self):
        """pre-allocates memory for interpolated volumes"""
        print('Pre-allocating 4D volume')
        self._img_4d = np.zeros(np.array([self._image.img.shape[0],
                                self._image.img.shape[1],
                                (self._image.nii.header['pixdim'][3] * self._image.nii.header['dim'][3]) / self._dxyz[2],
                                self._nstates]).astype(int)
                                )

    def _get_query_points(self):
        """Defines query points for interpolation according to resolution definition"""

        if self._resolution == 'isotropic':
            self._dxyz = np.array([self._image.nii.header['pixdim'][1],
                                   self._image.nii.header['pixdim'][1],
                                   self._image.nii.header['pixdim'][1]])
            nslices = int((self._image.nii.header['pixdim'][3] * self._image.nii.header['dim'][3]) / self._dxyz[2])

        else:
            self._dxyz = np.array([self._resolution,
                                   self._resolution,
                                   self._resolution])
            nslices = int((self._image.nii.header['pixdim'][3] * self._image.nii.header['dim'][3]) / self._dxyz[2])

        # define query points
        self._xq = np.linspace(0,
                               (self._dxyz[0] * (nslices-1)),
                               nslices)  # z-query points

        self._yq = np.linspace(0,
                               (self._dxyz[1] * (nslices-1)),
                               nslices)  # z-query points

        self._zq = np.linspace(0,
                               (self._dxyz[2] * (nslices-1)),
                               nslices)  # z-query points

    def _interp_fast_linear(self):
        """Linear interpolation onto regular grid - fastest interpolation method"""

        # define indexed points
        self._xi = np.int_(np.linspace(0, self._image.nii.header['dim'][1] - 1, self._image.nii.header['dim'][1]))
        self._yi = np.int_(np.linspace(0, self._image.nii.header['dim'][2] - 1, self._image.nii.header['dim'][2]))

        print('Interpolating z dimension')
        for ww in range(1, self._nstates + 1):

            slice_idx = np.where(self._states == ww)
            zs = (self._slice_locations[slice_idx, ]).flatten()  # z-sample points

            for xx in np.nditer(self._xi):
                for yy in np.nditer(self._yi):
                    z_interp = np.interp(self._zq, zs, self._image.img[xx, yy, slice_idx].flatten())
                    self._img_4d[xx, yy, :, ww - 1] = z_interp.flatten()

    def _interp_gpr(self):
        """Interpolates according to a gaussian regression model"""
        # define indexed points
        self._xi = np.int_(np.linspace(0, self._image.nii.header['dim'][1] - 1, self._image.nii.header['dim'][1]))
        self._yi = np.int_(np.linspace(0, self._image.nii.header['dim'][2] - 1, self._image.nii.header['dim'][2]))

        # Instantiate a Gaussian Process model
        kernel = 1.0 * RBF(length_scale=10, length_scale_bounds=(5, 20)) \
                 + WhiteKernel(noise_level=50, noise_level_bounds=(10, 100))

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4)

        cores = cpu_count()

        # start timer
        t1 = time.time()

        for ww in range(1, self._nstates + 1):
            print('Interpolating resp window: %d' % ww)

            slice_idx = np.where(self._states == ww)
            zs = (self._slice_locations[slice_idx, ]).flatten()  # z-sample points

            sub_arrays = Parallel(n_jobs=cores)(delayed(self._gpr_fit_line)  # function name
                                                (self._image.img[xx, yy, slice_idx].flatten().reshape(-1, 1),  # input args
                                                 zs.reshape(-1, 1), self._zq.reshape(-1, 1), gp)
                                                for xx in np.nditer(self._xi) for yy in np.nditer(self._yi))  # loop def

            index = 0
            for xx in np.nditer(self._xi):
                for yy in np.nditer(self._yi):
                    self._img_4d[xx, yy, :, ww-1] = sub_arrays[index].flatten()
                    index = index + 1

        # print function duration info
        print('%s duration: %.1fs [%d threads]' % ('_interp_gpr', (time.time() - t1), cores))

    @staticmethod
    def _gpr_fit_line(y, zs, zq, gp):
        gp.fit(zs, y)
        y_gpr = gp.predict(zq)
        return y_gpr
