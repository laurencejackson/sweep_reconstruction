"""
Class containing data and functions for reconstructing image data.

Laurence Jackson, BME, KCL 2019
"""

import copy
import numpy as np


class Reconstruction(object):
    """Class containing data and functions for reconstructing image data"""

    def __init__(self,
                 image,
                 target,
                 write_paths,
                 states,
                 patches=True):
        """
        Initialise reconstruction object
        :param image: ImageData object input
        :param target: ImageData object target
        :param write_paths: WritePaths object
        """

        self._image = image
        self._target = target
        self._write_paths = write_paths
        self._states = states
        self._patches = patches

    def run(self):
        """Runs reconstruction pipeline"""
        if self._patches:
            print('Extracting patches')
            self._extract_patches(self._image, target=False)
            self._extract_patches(self._target, target=True)

        # perform first SVR phase
        
        # perform 2nd SVR phase (all slices)


    def _extract_patches(self, image, patch_size=None, patch_stride=None, target=False):
        """Extracts 2D patches with overlap and preserved geometry as NIfTI files"""
        if patch_size is None:
            patch_size = np.array([image.img.shape[0] / 2.1, image.img.shape[1] / 2.1, 0]).astype(int)

        if patch_stride is None:
            patch_stride = (patch_size * 0.3).astype(int)

        rect_list = self._patch_list_xy(image.img.shape, patch_size, patch_stride)
        zlist, zsegs = self._patch_list_dim(image.img.shape[2], patch_size[2], patch_stride[2])

        for patch_ind in range(rect_list.shape[0]):
            crop_rect = np.array([[rect_list[patch_ind, 0], rect_list[patch_ind, 1]],
                              [rect_list[patch_ind, 2], rect_list[patch_ind, 3]]]).astype(int)

            for zz in range(zlist.shape[0]):
                if target:  # loop
                    for ww in range(0, image.img.shape[3]):
                        temp_image = copy.deepcopy(image)
                        temp_image.square_crop(rect=crop_rect, crop_t=ww)
                        temp_image.write_nii(self._write_paths.path_patch_img(patch_ind, zz, ww=ww, target=target))
                else:
                    temp_image = copy.deepcopy(image)
                    temp_image.square_crop(rect=crop_rect)
                    temp_image.write_nii(self._write_paths.path_patch_img(patch_ind, zz, target=target))

                    for ww in range(1,  max(self._states) + 1):
                        slice_idx = np.where((self._states != ww) &
                                             (self._states != 0) &
                                             (range(0, temp_image.img.shape[2]) >= zlist[zz, 0]) &
                                             (range(0, temp_image.img.shape[2]) <= zlist[zz, 1]))

                        exc_list = np.array(slice_idx).astype(int)
                        np.savetxt(self._write_paths.path_patch_txt(patch_ind, zz, ww), exc_list, fmt='%d', newline=' ')

    def _patch_list_xy(self, img_size, patch_size, patch_stride):
        """ Modifies stride so that the desired patch size fits within the image slice - returns rect list"""

        xlist, xsegs = self._patch_list_dim(img_size[0], patch_size[0], patch_stride[0])
        ylist, ysegs = self._patch_list_dim(img_size[1], patch_size[1], patch_stride[1])

        rect_list = np.zeros([(xsegs * ysegs), 4])
        rect_list[0, :] = [xlist[0, 0], ylist[0, 0], xlist[0, 1], ylist[0, 1]]

        for xi in range(0, xsegs):
            for yi in range(0, ysegs):
                rect_list[(xi * ysegs) + yi, :] = [xlist[xi, 0], ylist[yi, 0], xlist[xi, 1], ylist[yi, 1]]

        return rect_list

    @ staticmethod
    def _patch_list_dim(dim_size, patch_size, stride):
        """Returns start/stop points for patching"""
        if patch_size == 0:  # zeros size indicates use whole dimension
            plist = np.array([0, dim_size], ndmin=2)
            nsegs = 1
        else:
            nsegs = np.round((dim_size + stride) / (patch_size - stride)).astype(int)
            plist = np.zeros([nsegs, 2])
            plist[0, :] = [0, patch_size]
            plist[-1, :] = [dim_size - patch_size, dim_size]

            for ii in range(1, nsegs-1):
                plist[ii, :] = [(ii * (patch_size - stride)), ((ii+1) * (patch_size - stride) + stride)]

        return plist, nsegs
