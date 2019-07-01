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
                 interp_method,
                 resolution='isotropic',
                 kernel_dims=1
                 ):
        """
        initilises ResampleData
        :param image: ImageData object containing input image
        :param states: vector mapping each slice of image to a respiration state
        :param slice_locations: vector mapping each slice to a spatial position relative to origin
        :param write_paths: WritePaths object containing paths to write output
        :param interp_method: method for interpolation, 'fast_linear' or 'gpr' (slower and smoother)
        :param resolution: resolution of output - defaults to 'isotropic' but also takes a float [mm]
        """

        self._image = image
        self._image_4d = copy.deepcopy(image)
        self._states = states
        self._slice_locations = slice_locations
        self._write_paths = write_paths
        self._interp_method = interp_method
        self._resolution = resolution
        self._nstates = np.max(states)
        self._kernel_dims = kernel_dims

    def run(self):
        """Runs chosen re-sampling scheme """

        # initialise output volume
        self._get_query_points()
        self._init_vols()

        # perform chosen interpolation
        print('Re-sampling method: %s' % self._interp_method)
        if self._interp_method == 'fast_linear':
            self._interp_fast_linear()
        elif self._interp_method == 'gpr':
            self._interp_gpr()
        else:
            raise Exception('\nInvalid data re-sampling method: %s\n' % self._interp_method)

        # write output
        self._write_resampled_data()

    def _write_resampled_data(self):
        """Saves re-sampled image"""
        # TODO: can only correct for interpolation in z at the moment
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

        for ww in range(1, self._nstates + 1):
            print('Interpolating resp window: %d' % ww)

            slice_idx = np.where(self._states == ww)
            zs = (self._slice_locations[slice_idx, ]).flatten()  # z-sample points

            for xx in np.nditer(self._xi):
                for yy in np.nditer(self._yi):
                    z_interp = np.interp(self._zq, zs, self._image.img[xx, yy, slice_idx].flatten())
                    self._img_4d[xx, yy, :, ww - 1] = z_interp.flatten()

    def _interp_gpr(self, kernel_3d=False):
        """Interpolates according to a gaussian regression model"""
        # define indexed points
        self._xi = np.int_(np.linspace(0, self._image.nii.header['dim'][1] - 1, self._image.nii.header['dim'][1]))
        self._yi = np.int_(np.linspace(0, self._image.nii.header['dim'][2] - 1, self._image.nii.header['dim'][2]))

        # Instantiate a Gaussian Process kernel
        kernel = 1.0 * RBF(length_scale=12, length_scale_bounds=(5, 25)) \
                 + WhiteKernel(noise_level=20, noise_level_bounds=(5, 50))

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)

        cores = max(1, cpu_count())
        length_scale = 3

        if self._kernel_dims > 1:
            kernel_3d = True

        # start timer
        t1 = time.time()

        for ww in range(1, self._nstates + 1):
            print('Interpolating resp window: %d' % ww)

            slice_idx = np.where(self._states == ww)
            self._zs = (self._slice_locations[slice_idx, ]).flatten()  # z-sample points

            sub_arrays = Parallel(n_jobs=cores)(delayed(self._gpr_fit_line)  # function name
                                                (self._get_training_y(xx, yy, slice_idx, # input args
                                                                      kernel_3d=kernel_3d, length_scale=length_scale),
                                                 self._get_training_x(xx, yy, slice_idx,
                                                                      kernel_3d=kernel_3d, length_scale=length_scale),
                                                 self._get_zq(xx, yy,
                                                              kernel_3d=kernel_3d, length_scale=length_scale),
                                                 gp)
                                                for xx in np.nditer(self._xi) for yy in np.nditer(self._yi))  # loop def

            # insert interpolated data into pre-allocated volume
            index = 0
            for xx in np.nditer(self._xi):
                for yy in np.nditer(self._yi):
                    self._img_4d[xx, yy, :, ww-1] = sub_arrays[index].flatten()
                    index = index + 1

        # print function duration info
        print('%s duration: %.1fs [%d threads]' % ('_interp_gpr', (time.time() - t1), cores))

    def _get_training_y(self, x, y, slice_idx, kernel_3d=False, length_scale=2):
        """Gets array of training point values"""
        if kernel_3d and length_scale > 1:
            training_x = self._get_pixels_xy(x, y, slice_idx, length_scale=length_scale)
            training_y = self._image.img[training_x[0, :], training_x[1, :], training_x[2, :]].transpose()
        else:  # standard 1D GPR
            training_y = self._image.img[x, y, slice_idx].flatten().reshape(-1, 1)

        return training_y

    def _get_training_x(self, x, y, slice_idx, kernel_3d=False, length_scale=2):
        """Gets array of training point co-ordinates"""
        if kernel_3d and length_scale > 1:
            training_x = self._get_pixels_xy(x, y, slice_idx, length_scale=length_scale).transpose()

            training_x = training_x.astype(float)
            training_x[:, 0] = training_x[:, 0] * self._image.nii.header['pixdim'][1]
            training_x[:, 1] = training_x[:, 1] * self._image.nii.header['pixdim'][2]
            training_x[:, 2] = training_x[:, 2] * self._image.nii.header['pixdim'][3]
        else:
            training_x = self._zs.reshape(-1, 1)

        return training_x

    def _get_zq(self, x, y, kernel_3d=False, length_scale=2):
        """returns the relative query points"""
        if kernel_3d and length_scale > 1:
            X, Y, Z = np.meshgrid(x, y, self._zq)
            zq = np.vstack([X.ravel() * self._image.nii.header['pixdim'][1],
                            Y.ravel() * self._image.nii.header['pixdim'][2],
                            Z.ravel()]).transpose()
        else:
            zq = self._zq.reshape(-1, 1)
        return zq

    def _get_pixels_xy(self, x, y, slice_idx, length_scale=2):
        """Gets valid xy pixel indices"""

        dxy = np.floor(length_scale/2).astype(int)  # range of returned values

        # define x and y ranges inside image range
        x_min = max(x - dxy, 0)
        x_max = min(x + dxy + 1, self._image.img.shape[0])
        y_min = max(y - dxy, 0)
        y_max = min(y + dxy + 1, self._image.img.shape[1])

        # define pixel indices
        x_locs = np.arange(x_min, x_max)
        y_locs = np.arange(y_min, y_max)
        z_locs = np.array(slice_idx)

        X, Y, Z = np.meshgrid(x_locs, y_locs, z_locs)

        return np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

    @ staticmethod
    def _gpr_fit_line(y, X, zq, gp):
        """Simple parallel function to fit GPR model to one line of z data"""
        gp.fit(X, y)
        return gp.predict(zq)
