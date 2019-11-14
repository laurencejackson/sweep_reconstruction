"""
Class containing data and functions for estimating respiration siganl from 3D data

Laurence Jackson, BME, KCL 2019
"""
import os
import sys
import time
import numpy as np
import copy
import itertools
import random

from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter


class CorePeripheryTarget(object):

    def __init__(self, img, local_patch_size, args, write_paths):
        """Initialise"""
        self._image = img
        self._filtered_image = copy.deepcopy(img)
        self._img_local = None # copy.deepcopy(img)
        self._local_patch_size = local_patch_size
        self._args = args
        self._write_paths = write_paths
        self._nsx = 8
        self._nsy = 8
        self._adj = np.zeros((self._nsx, self._nsy, self._image.img.shape[2], self._image.img.shape[2]))
        self._sim = np.zeros(local_patch_size[2])
        self.locs = np.zeros((self._nsx, self._nsy, self._image.img.shape[2]))

        # vars
        self.slice_thickness = self._args.thickness
        self.window_size = min(0.5 * self._local_patch_size[2] - 1, (self._args.window_size / self._image.nii.header['pixdim'][3])).astype(int)

    def run(self):

        self._pre_process_image()
        self._generate_locs()

    def _pre_process_image(self):
        """Correct respiraiton using core/periphery networks"""

        print('Filtering image....')
        filtered_data = self._process_slices_parallel(self._filter_gaussian,
                                                      self._image.img,
                                                      cores=self._args.n_threads)

        self._filtered_image.set_data(filtered_data)

    def _generate_locs(self):
        """ Find local similarity measure"""
        x1 = max(self._local_patch_size[0]/2, (self._image.img.shape[0] / (self._nsx + 1)))
        x2 = min(self._image.img.shape[0] - self._local_patch_size[0]/2, (self._image.img.shape[0] - (self._image.img.shape[0] / (self._nsx+1))))
        y1 = max(self._local_patch_size[1]/2, (self._image.img.shape[1] / (self._nsy + 1)))
        y2 = min(self._image.img.shape[1] - self._local_patch_size[1]/2, (self._image.img.shape[1] - (self._image.img.shape[1] / (self._nsy + 1))))

        self.px = np.linspace(x1, x2, self._nsx).astype(int)
        self.py = np.linspace(y1, y2, self._nsy).astype(int)

        for nx, xx in enumerate(self.px):
            for ny, yy in enumerate(self.py):
                print(' Analysing patch (%d,%d): %d/%d' % (xx, yy, (nx*self.px.size) + ny + 1, self.px.size * self.py.size), end=' ', flush=True)
                self._extract_local_patch(xx, yy, focus=True)
                self._adj[nx, ny, :, :] = self._local_sim()
                self.locs[nx, ny, :] = self._core_periphery(np.squeeze(self._adj[nx, ny, :, :]))

    def _extract_local_patch(self, xx, yy, focus=False):
        """Extracts patch centred at xx, yy"""
        x1 = np.int_(xx - self._local_patch_size[0]/2)
        x2 = np.int_(xx + self._local_patch_size[0]/2)
        y1 = np.int_(yy - self._local_patch_size[1]/2)
        y2 = np.int_(yy + self._local_patch_size[1]/2)

        # self._img_local.square_crop(local_rect)
        self._img_local = np.zeros((self._local_patch_size[0], self._local_patch_size[1], self._image.img.shape[2]))
        self._img_local[:, :, :] = self._image.img[x1:x2, y1:y2, 0:self._image.img.shape[2]]

        if focus:
            sig = self._local_patch_size[0] / 2
            xxlin = np.arange(0, self._local_patch_size[0])
            tkx = np.exp(-np.power(xxlin - sig, 2.) / (2 * np.power(sig, 2.)))
            C = np.ones((self._local_patch_size[0], self._local_patch_size[1]))
            C = tkx[:, np.newaxis] * C * tkx[np.newaxis, :]
            C = (C - np.min(C)) / (np.max(C) - np.min(C))
            for zz in range(0, self._image.img.shape[2]):
                self._img_local[:, :, zz] = self._img_local[:, :, zz] * C

    def _local_sim(self):
        """Creates adjacency matrix from local patch"""
        print('->Calculating local similarity', end=' ', flush=True)
        # loop over target slices
        sim_mat = np.zeros((self._img_local.shape[2], self._img_local.shape[2]))

        for tt in range(0, self._img_local.shape[2] - 1):
            # loop over test slices
            img2 = self._img_local[:, :, tt].ravel()
            ti1 = max(0, tt - int(self._local_patch_size[2] / 2))
            ti2 = min(self._image.img.shape[2] - 1, tt + int(self._local_patch_size[2] / 2))
            _sim = np.zeros((ti2 - ti1))
            for nn, zz in enumerate(range(ti1, ti2)):
                img1 = self._img_local[:, :, zz].ravel()
                _sim[nn] = self._zncc(img1, img2)

            sim_mat[tt, range(ti1, ti2)] = _sim

        return sim_mat

    def _core_periphery(self, C, WindowSize=16):
        """sliding window core/periphery graph"""
        print('->Assigning core/periphery ', flush=True)
        gamma_inc = 0.005
        gamma_max = 1.8
        vecCore = np.zeros(C.shape[0])
        controlMethod = 'maxSeparation'  # make variable for future development options
        max_sep_fraction = 2.0
        coreMask = np.zeros((WindowSize, C.shape[0]))

        for n in range(0, C.shape[1]-WindowSize):
            gamma = 1
            Caux = C[n:n + WindowSize, n: n + WindowSize]
            if controlMethod == 'maxSeparation':
                slice_thickness_n = self.slice_thickness / self._image.nii.header['pixdim'][3]
                max_separation = min(round(max_sep_fraction * slice_thickness_n), WindowSize-3)
                longest_sep = 0
                coreMask[:, n] = self._core_periphery_dir(Caux, gamma)[0]
                while longest_sep < max_separation and np.sum(coreMask[:, n]) > 3:
                    coreMask[:, n] = self._core_periphery_dir(Caux, gamma)[0]
                    longest_sep = max(len(list(y)) for (c, y) in itertools.groupby(coreMask[:, n]) if c==0)
                    gamma = gamma + gamma_inc

                    if gamma > gamma_max:
                        print(':Gamma max reached:', end=' ')
                        break

                gamma_opt = gamma - (2 * gamma_inc)
                coreMask[:, n] = self._core_periphery_dir(Caux, gamma_opt)[0]

            vecCore[np.argwhere(coreMask[:, n] > 0) + n] = vecCore[np.argwhere(coreMask[:, n] > 0) + n] + 1

        locs = vecCore > (0.2 * WindowSize)

        return locs

    def _core_periphery_dir(self, W, gamma=1, C0=None, seed=None):
        """
        Credit for _get_rng and _core_periphery_dir goes to bctpy and Roan LaPlante (https://github.com/aestrivex/bctpy)

       The optimal core/periphery subdivision is a partition of the network
        into two nonoverlapping groups of nodes, a core group and a periphery
        group. The number of core-group edges is maximized, and the number of
        within periphery edges is minimized.
        The core-ness is a statistic which quantifies the goodness of the
        optimal core/periphery subdivision (with arbitrary relative value).
        The algorithm uses a variation of the Kernighan-Lin graph partitioning
        algorithm to optimize a core-structure objective described in
        Borgatti & Everett (2000) Soc Networks 21:375-395
        See Rubinov, Ypma et al. (2015) PNAS 112:10032-7
        Parameters
        ----------
        W : NxN np.ndarray
            directed connection matrix
        gamma : core-ness resolution parameter
            Default value = 1
            gamma > 1 detects small core, large periphery
            0 < gamma < 1 detects large core, small periphery
        C0 : NxN np.ndarray
            Initial core structure
        seed : hashable, optional
            If None (default), use the np.random's global random state to generate random numbers.
            Otherwise, use a new np.random.RandomState instance seeded with the given value.
        """
        rng = self._get_rng(seed)
        n = len(W)
        np.fill_diagonal(W, 0)

        if C0 == None:
            C = rng.randint(2, size=(n,))
        else:
            C = C0.copy()

        # methodological note, the core-detection null model is not corrected
        # for degree cf community detection (to enable detection of hubs)

        s = np.sum(W)
        p = np.mean(W)
        b = W - gamma * p
        B = (b + b.T) / (2 * s)
        cix, = np.where(C)
        ncix, = np.where(np.logical_not(C))
        q = np.sum(B[np.ix_(cix, cix)]) - np.sum(B[np.ix_(ncix, ncix)])

        # sqish

        flag = True
        it = 0
        while flag:
            it += 1
            if it > 100:
                print('Infinite Loop - aborted')
                break

            flag = False
            # initial node indices
            ixes = np.arange(n)

            Ct = C.copy()
            while len(ixes) > 0:
                Qt = np.zeros((n,))
                ctix, = np.where(Ct)
                nctix, = np.where(np.logical_not(Ct))
                q0 = (np.sum(B[np.ix_(ctix, ctix)]) -
                      np.sum(B[np.ix_(nctix, nctix)]))
                Qt[ctix] = q0 - 2 * np.sum(B[ctix, :], axis=1)
                Qt[nctix] = q0 + 2 * np.sum(B[nctix, :], axis=1)

                max_Qt = np.max(Qt[ixes])
                u, = np.where(np.abs(Qt[ixes] - max_Qt) < 1e-10)
                # tunourn
                u = u[rng.randint(len(u))]
                Ct[ixes[u]] = np.logical_not(Ct[ixes[u]])
                # casga

                ixes = np.delete(ixes, u)

                if max_Qt - q > 1e-10:
                    flag = True
                    C = Ct.copy()
                    cix, = np.where(C)
                    ncix, = np.where(np.logical_not(C))
                    q = (np.sum(B[np.ix_(cix, cix)]) -
                         np.sum(B[np.ix_(ncix, ncix)]))

        cix, = np.where(C)
        ncix, = np.where(np.logical_not(C))
        q = np.sum(B[np.ix_(cix, cix)]) - np.sum(B[np.ix_(ncix, ncix)])

        return C, q

    @staticmethod
    def _zncc(img1, img2):
        """
        Return zero normalised cross correlation
        """
        return (1/img1.size) * np.sum((1/(np.std(img1)*np.std(img2))) * (img1-np.mean(img1)) * (img2-np.mean(img2)))

    @ staticmethod
    def _filter_gaussian(img, sigma=0.75):
        """
        Median filter
        :param imgs: slice to filter [2D]
        :param kernel_size: size of median kernel
        :return:
        """
        return gaussian_filter(img, sigma=sigma)  # gaussian filter

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

    @staticmethod
    def _get_rng(seed=None):
        """
        Credit for _get_rng and _core_periphery_dir goes to bctpy and Roan LaPlante (https://github.com/aestrivex/bctpy)

        By default, or if `seed` is np.random, return the global RandomState
        instance used by np.random.
        If `seed` is a RandomState instance, return it unchanged.
        Otherwise, use the passed (hashable) argument to seed a new instance
        of RandomState and return it.
        Parameters
        ----------
        seed : hashable or np.random.RandomState or np.random, optional
        Returns
        -------
        np.random.RandomState
        """
        if seed is None or seed == np.random:
            return np.random.mtrand._rand
        elif isinstance(seed, np.random.RandomState):
            return seed
        try:
            rstate = np.random.RandomState(seed)
        except ValueError:
            rstate = np.random.RandomState(random.Random(seed).randint(0, 2 ** 32 - 1))
        return rstate
