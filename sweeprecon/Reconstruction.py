"""
Class containing data and functions for reconstructing image data.

Laurence Jackson, BME, KCL 2019
"""

import os
import copy
import numpy as np
import subprocess


class Reconstruction(object):
    """Class containing data and functions for reconstructing image data"""

    def __init__(self,
                 image,
                 target,
                 write_paths,
                 states,
                 args,
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
        self._args = args

    def run(self):
        """Runs reconstruction pipeline"""
        if self._patches:
            print('Extracting patches')
            self._extract_patches(self._image, target=False)
            self._extract_patches(self._target, target=True)

        # perform first SVR pass
        self._svr_options_init()
        print('performing 1st SVR pass')

        # TODO read more opts from args

        # loop over resp states
        opts = {'thickness': self._args.thickness}
        self._process_patches('reconstructAngio', opts)

    def _process_patches(self, function_path, opts):
        """Loop over patch directory structure and apply function"""

        # loop over resp states
        nstacks = 1  # likely to only work with one but I'll make this a variable anyway
        for ww in range(1, np.max(self._states) + 1):
            # loop over patches
            for source_dir in self._write_paths.patch_dir_list:

                basename = os.path.basename(os.path.normpath(source_dir))
                output_path = os.path.join(self._write_paths.path_patch_recon(basename, ww))
                source_path = os.path.join(source_dir, basename + '.nii.gz')
                exclude_path = os.path.join(source_dir, basename + '_excludes_' + str(ww) + '.txt')
                target_path = os.path.join(source_dir, 'target_' + basename + '_' + str(ww) + '.nii.gz')

                self._perform_svr(function_path, output_path, nstacks, source_path,
                                  target_path, exclude_path, opts=opts)

    def _perform_svr(self, function_path, output_path, nstacks, source_path, target_path, exclude_path, opts=None):
        """Co-registers slices from source to slice in target"""

        # update options dict with given values
        if opts is not None:
            for key, value in opts.items():
                self._svr_opts[key] = value

        # parse options string
        opts_string = ''
        for key, value in self._svr_opts.items():
            if type(value) is bool:
                if value is True:
                    string_val = '-' + str(key) + ' '
                else:
                    continue
            else:
                string_val = '-' + str(key) + ' ' + str(value) + ' '

            opts_string += string_val

        # parse argument string
        command_string = str('%s %s %d %s -template %s -excluded_file %s %s' % (function_path, output_path, nstacks, source_path,
                                                                 target_path, exclude_path, opts_string))
        print(command_string)
        subprocess.run(command_string.split())

    def _svr_options_init(self):
        """Initialise SVR options with defaults"""

        self._svr_opts = {
            "iterations": 1,
            "thickness": 2.5,
            "resolution": 1.0,
            "sr_iterations": 3,
            "filter": 10,
            "lastIter": 0.03,
            "delta": 400,
            "lambda": 0.035,

            "gaussian_only": False,
            "svr_only": True,
            "no_intensity_matching": True,
            "no_sr": False,
            "no_robust_statistics": True,
            "reg_log": False,
        }

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

                    for ww in range(1,  np.max(self._states) + 1):
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
