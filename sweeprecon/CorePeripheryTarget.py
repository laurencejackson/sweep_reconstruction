"""
Class containing data and functions for estimating respiration siganl from 3D data

Laurence Jackson, BME, KCL 2019
"""

import time
import numpy as np
import copy

from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class CorePeripheryTarget(object):

    def __init__(self, img, local_patch_size, args, write_paths):
        """Initialise"""
        self._image = img
        self._filtered_image = copy.deepcopy(img)
        self._img_local = copy.deepcopy(img)
        self._local_patch_size = local_patch_size
        self._args = args
        self._write_paths = write_paths
        self._nsx = 8
        self._nsy = 8
        self._adj = np.zeros((self._nsx, self._nsy, self._image.img.shape[2], self._image.img.shape[2]))
        self._sim = np.zeros(local_patch_size[2])

    def run(self):

        self._pre_process_image()
        self._generate_adjacency_matrix()

    def _pre_process_image(self):
        """Correct respiraiton using core/periphery networks"""

        print('Filtering image....')
        filtered_data = self._process_slices_parallel(self._filter_gaussian,
                                                      self._image.img,
                                                      cores=self._args.n_threads)

        self._filtered_image.set_data(filtered_data)

    def _generate_adjacency_matrix(self):
        """ Find local similarity measure"""
        px = np.linspace(self._local_patch_size[0] / 2, self._image.img.shape[0] - self._local_patch_size[0] / 2, self._nsx).astype(int)
        py = np.linspace(self._local_patch_size[1] / 2, self._image.img.shape[1] - self._local_patch_size[1] / 2, self._nsy).astype(int)

        for nx, xx in enumerate(px):
            for ny, yy in enumerate(py):
                print((xx, yy))
                self._extract_local_patch(xx, yy)
                self._adj[nx, ny, :, :] = self._local_sim(xx, yy)
                self._locs[nx, nx, :] = self._core_periphery()

    def _extract_local_patch(self, xx, yy):
        x1 = np.int_(xx - self._local_patch_size[0] / 2)
        x2 = np.int_(xx + self._local_patch_size[0] / 2)
        y1 = np.int_(yy - self._local_patch_size[1] / 2)
        y2 = np.int_(yy + self._local_patch_size[1] / 2)

        local_rect = np.array([[x1, y1],
                               [x2, y2]], dtype=int)

        self._img_local.square_crop(local_rect)

    def _local_sim(self, xx, yy):
        # loop over target slices
        sim_mat = np.zeros((self._image.img.shape[2], self._image.img.shape[2]))
        for tt in range(0, self._img_local.img.shape[2] - 1):
            # loop over test slices
            _offset1 = 0
            _offset2 = self._local_patch_size[2]
            for nn, zz in enumerate(range(tt - int(self._local_patch_size[2] / 2), tt + int(self._local_patch_size[2] / 2))):
                if zz < 0:
                    self._sim[nn] = 0
                    _offset1 += 1
                elif zz > self._image.img.shape[2] - 1:
                    self._sim[nn] = 0
                    _offset2 += -1
                else:
                    self._sim[nn] = self._zncc(self._img_local.img[:, :, zz], self._img_local.img[:, :, tt])

            sim_mat[tt, range(max(0, tt - int(self._local_patch_size[2] / 2)), min(tt + int(self._local_patch_size[2] / 2), self._image.img.shape[2]))] = self._sim[range(_offset1, _offset2),]  # .clip(min=0)  # set negative values to 0

        return sim_mat

    def _core_periphery(self):
        """sliding window core/periphery graph"""
        pass

    @staticmethod
    def _zncc(img1, img2):
        """
        Return zero normalised cross correlation
        """
        return (1/img1.size) * np.sum( (1/(np.std(img1.ravel())*np.std(img2.ravel()))) * (img1.ravel()-np.mean(img1.ravel())) * (img2.ravel()-np.mean(img2.ravel())))

    @ staticmethod
    def _filter_gaussian(img, sigma=0.75):
        """
        Median filter
        :param imgs: slice to filter [2D]
        :param kernel_size: size of median kernel
        :return:
        """
        return gaussian_filter(img, sigma)  # gaussian filter

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