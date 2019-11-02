"""
Class containing data and functions for re-sampling 3D data into respiration resolved volumes

Laurence Jackson, BME, KCL 2019
"""

import os
# limit threading to reduce cpu overhead in parallel processes - must be done before importing num/scipy
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import copy
import time
import multiprocessing as mp
import numpy as np

from scipy import interpolate
from skimage.filters import frangi


class ResampleData(object):

    def __init__(self,
                 image,
                 states,
                 slice_locations,
                 write_paths,
                 args,
                 resolution='isotropic',
                 kernel_dims=1,
                 n_threads=0
                 ):
        """
        initilises ResampleData
        :param image: ImageData object containing input image
        :param states: vector mapping each slice of image to a respiration state OR matrix with graph locations
        :param slice_locations: vector mapping each slice to a spatial position relative to origin
        :param write_paths: WritePaths object containing paths to write output
        :param interp_method: method for interpolation, 'fast_linear' or 'rbf' (slower but smoother)
        :param resolution: resolution of output - defaults to 'isotropic' but also takes a float [mm]
        """

        self._image = image
        self._image_4d = copy.deepcopy(image)
        self._image_resp_3d = copy.deepcopy(image)
        self._states = states
        if self._states.ndim > 1:
            self._graph_resample = True
            self._nstates = 1
        else:
            self._nstates = np.max(states)

        self._slice_locations = slice_locations
        self._write_paths = write_paths
        self._args = args
        self._resolution = resolution
        self._kernel_dims = kernel_dims
        self._n_threads = n_threads

    def run(self):
        """Runs chosen re-sampling scheme """
        # initialise output volume
        self._get_query_points()
        self._init_vols()

        # perform fast interpolation
        print('Performing fast interpolation')
        self._interp_fast_linear()

        # perform chosen refined interpolation
        print('Re-sampling method: %s' % self._args.interpolator)
        if self._args.interpolator == 'rbf':
            self._interp_rbf()

    def _write_resampled_data(self, image_obj, path):
        """Saves re-sampled image"""
        # TODO: can only correct for interpolation in z at the moment
        image_obj.nii.affine[:, 2] = self._image.nii.affine[:, 2] * (self._dxyz[2] / self._image.nii.header['pixdim'][3])
        image_obj.write_nii(path)

    def _init_vols(self):
        """pre-allocates memory for interpolated volumes"""
        print('Pre-allocating 4D volume')
        if
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

    def _define_index_xy(self):
        """Defines indexed points in x and y"""
        self._xi = np.int_(np.linspace(0, self._image.nii.header['dim'][1] - 1, self._image.nii.header['dim'][1]))
        self._yi = np.int_(np.linspace(0, self._image.nii.header['dim'][2] - 1, self._image.nii.header['dim'][2]))

    def _interp_fast_linear(self):
        """Linear interpolation onto regular grid - fastest interpolation method"""

        # define indexed points
        self._define_index_xy()

        for ww in range(1, self._nstates + 1):
            print('Interpolating resp window: %d' % ww)

            for xx in np.nditer(self._xi):
                for yy in np.nditer(self._yi):
                    slice_idx = np.where(self._states == ww)
                    zs = (self._slice_locations[slice_idx, ]).flatten()  # z-sample points
                    z_interp = np.interp(self._zq, zs, self._image.img[xx, yy, slice_idx].flatten())
                    self._img_4d[xx, yy, :, ww - 1] = z_interp.flatten()

            # save single resp state volumes
            self._image_resp_3d.set_data(self._img_4d[:, :, :, ww - 1])
            self._write_resampled_data(self._image_resp_3d, self._write_paths.path_interpolated_3d_linear(ww))
            print('---')

        # write to file
        self._image_4d.set_data(self._img_4d)
        self._write_resampled_data(self._image_4d, self._write_paths.path_interpolated_4d_linear())

        # if frangi filter 4D linear volume
        if self._args.frangi:
            img_frangi = copy.deepcopy(self._image_4d)
            img_frangi.apply_spatial_filter(frangi, 4, sigmas=(0.75, 2.0, 0.25), alpha=0.5, beta=0.5,
                                            gamma=90, black_ridges=False)
            self._write_resampled_data(img_frangi, self._write_paths.path_interpolated_4d_linear(pre='frangi_'))

    def _interp_rbf(self):
        """Interpolate using radial basis function"""

        # re-initialise vols
        self._init_vols()
        self._define_index_xy()
        
        if self._n_threads is 0:
            cores = max(1, mp.cpu_count() - 1)
        else:
            cores = self._n_threads

        print('Starting pool [%d processes]' % cores)
        pool = mp.Pool(cores)

        t1 = time.time()

        for ww in range(1, self._nstates + 1):
            print('Interpolating resp window: %d' % ww)
            tt = time.time()
            slice_idx = np.where(self._states == ww)
            self._zs = (self._slice_locations[slice_idx, ]).flatten()  # z-sample points

            sub_arrays = pool.starmap_async(self._rbf_interp_line,
                                         [(self._get_training_y(xx, yy, slice_idx, kernel_dim=self._kernel_dims),
                                         self._get_training_x(xx, yy, slice_idx, kernel_dim=self._kernel_dims),
                                         self._get_zq(xx, yy, kernel_dim=self._kernel_dims))
                                          for xx in np.nditer(self._xi) for yy in np.nditer(self._yi)]).get()

            print('%s duration: %.1fs' % ('_interp_rbf', (time.time() - tt)))

            # insert interpolated data into pre-allocated volume
            print('\ncollecting data')
            index = 0
            for xx in np.nditer(self._xi):
                for yy in np.nditer(self._yi):
                    self._img_4d[xx, yy, :, ww - 1] = sub_arrays[index].flatten()
                    index = index + 1

            # save single resp state volumes
            self._image_resp_3d.set_data(self._img_4d[:, :, :, ww - 1])
            self._write_resampled_data(self._image_resp_3d, self._write_paths.path_interpolated_3d(ww))

            print('---')

        pool.close()
        pool.join()

        # reset python environ to normalise threading
        os.environ.clear()

        # write full 4D interp volume
        self._image_4d.set_data(self._img_4d)
        self._write_resampled_data(self._image_4d, self._write_paths.path_interpolated_4d())

        # if frangi filter 4D rbf volume
        if self._args.frangi:
            img_frangi = copy.deepcopy(self._image_4d)
            img_frangi.apply_spatial_filter(frangi, 4, sigmas=(0.75, 2.0, 0.25), alpha=0.5, beta=0.5,
                                            gamma=90, black_ridges=False)
            self._write_resampled_data(img_frangi, self._write_paths.path_interpolated_4d(pre='frangi_'))

        # print function duration info
        print('%s duration: %.1fs' % ('_interp_rbf', (time.time() - t1)))

    def _get_training_y(self, x, y, slice_idx, kernel_dim=1):
        """Gets array of training point values"""
        if kernel_dim > 1:
            training_x = self._get_pixels_xy(x, y, slice_idx, kernel_dim=kernel_dim)
            training_y = self._image.img[training_x[0, :], training_x[1, :], training_x[2, :]].transpose()
        else:  # standard 1D GPR
            training_y = self._image.img[x, y, slice_idx].flatten().reshape(-1, 1)

        return training_y

    def _get_training_x(self, x, y, slice_idx, kernel_dim=1):
        """Gets array of training point co-ordinates"""
        if kernel_dim > 1:
            training_x = self._get_pixels_xy(x, y, slice_idx, kernel_dim=kernel_dim).transpose()

            training_x = training_x.astype(float)
            training_x[:, 0] = training_x[:, 0] * self._image.nii.header['pixdim'][1]
            training_x[:, 1] = training_x[:, 1] * self._image.nii.header['pixdim'][2]
            training_x[:, 2] = training_x[:, 2] * self._image.nii.header['pixdim'][3]
        else:
            training_x = self._zs.reshape(-1, 1)

        return training_x

    def _get_zq(self, x, y, kernel_dim):
        """returns the relative query points"""
        if kernel_dim > 1:
            X, Y, Z = np.meshgrid(x, y, self._zq)
            zq = np.vstack([X.ravel() * self._image.nii.header['pixdim'][1],
                            Y.ravel() * self._image.nii.header['pixdim'][2],
                            Z.ravel()]).transpose()
        else:
            zq = self._zq.reshape(-1, 1)
        return zq

    def _get_pixels_xy(self, x, y, slice_idx, kernel_dim=1):
        """Gets valid xy pixel indices"""

        dxy = np.floor(kernel_dim/2).astype(int)  # range of returned values

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
    def _rbf_interp_line(y, X, zq):
        """Simple function to fit RBF model to one line of z data"""
        t1 = time.time()
        rbfi = interpolate.Rbf(X[:, 0], X[:, 1], X[:, 2], y[:, ], function='multiquadric', epsilon=0.6, smooth=4)
        z_pred = rbfi(zq[:, 0], zq[:, 1], zq[:, 2])

        sys.stdout.write('\r' + 'CPU time per line = {:.3f}'.format(time.time() - t1))

        return z_pred
