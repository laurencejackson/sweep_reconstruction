"""
Class containing data and functions for reconstructing image data.

Laurence Jackson, BME, KCL 2019
"""

import os
import copy
import numpy as np
import subprocess
import shutil

from skimage.filters import frangi
from sweeprecon.io.ImageData import ImageData


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
        self._target_list = []
        self._patch_list = []
        self._recon_patch_list = []

    def run(self):
        """Runs reconstruction pipeline"""
        if self._patches:
            if os.path.isdir(self._write_paths._patches_folder):
                print('removing existing patches')
                shutil.rmtree(self._write_paths._patches_folder)
            print('Extracting patches')
            self._extract_patches(self._image, target=False)
            self._extract_patches(self._target, target=True)

        # perform first SVR pass
        self._svr_options_init()

        # TODO read more opts from args
        opts = {'thickness': self._args.thickness,
                'ffd': self._args.free_form_deformation}

        self._process_patches('reconstructAngio', opts)

    def _process_patches(self, function_path, opts):
        """Loop over patch directory structure and apply function"""

        # loop over resp states
        nstacks = 1  # likely to only work with one but I'll make this a variable anyway
        self._set_resp_range()

        for ww in self.resp_range:
            # loop over patches
            for source_dir in self._write_paths.patch_dir_list:

                basename = os.path.basename(os.path.normpath(source_dir))
                output_path = os.path.join(self._write_paths.path_patch_recon(basename, ww))
                source_path = os.path.join(source_dir, basename + '.nii.gz')
                exclude_path = os.path.join(source_dir, basename + '_excludes_' + str(ww) + '.txt')
                target_path = os.path.join(source_dir, 'target_' + basename + '_' + str(ww) + '.nii.gz')

                self._recon_patch_list.append(output_path)

                self._perform_svr(function_path, output_path, nstacks, source_path,
                                  target_path, exclude_path, opts=opts)

            # combine patches
            self._recombine_patches('combine_patches')

            # rename output
            os.rename('combined.nii.gz', self._write_paths.path_combined_patches(ww))

            if self._args.frangi:
                img_frangi = ImageData(self._write_paths.path_combined_patches(ww))
                img_frangi.apply_spatial_filter(frangi, 3, sigmas=(0.75, 2.0, 0.25), alpha=0.5, beta=0.5,
                                                gamma=90, black_ridges=False)
                img_frangi.write_nii(self._write_paths.path_combined_patches(ww, pre='frangi_'))

            # clear patch list
            self._recon_patch_list = []

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
        command_string = str(
            '%s %s %d %s -template %s %s' %
            (function_path, output_path, nstacks, source_path, target_path, opts_string))

        print(command_string)
        subprocess.run(command_string.split())

    def _recombine_patches(self, function_path):
        """Recombines patches to common space"""
        # combine_patches
        # Usage: combine_patches[target_volume][output_resolution][N][stack_1]..[stack_N]
        command_string_1 = str('%s %s %f %d' % (function_path,
                                                self._write_paths.path_interpolated_3d_linear(1),
                                                self._svr_opts['resolution'],
                                                self._npatches))

        command_string = command_string_1 + ' ' + ' '.join(self._recon_patch_list)
        print(command_string)
        # combine patches
        subprocess.run(command_string.split())

    def _svr_options_init(self):
        """Initialise SVR options with defaults"""

        self._svr_opts = {
            # variable options
            "thickness": 2.5,
            "ffd": False,

            # hard-coded options
            "iterations": 1,
            "resolution": 0.75,
            "sr_iterations": 4,
            # "filter": 10,
            "lastIter": 0.03,
            "delta": 400,
            "lambda": 0.030,
            # "-cp_spacing": 5.0,
            "ncc": True,
            "gaussian_only": False,
            "svr_only": False,
            "no_intensity_matching": True,
            "no_sr": False,
            "no_robust_statistics": True,
            "reg_log": False,
        }

    def _extract_patches(self, image, target=False):
        """Uses MIRTK extract-image-region function to extract patches"""
        # extract-image-region <input> <output> [options] -patch <i> <j> <k> <nx> [<ny> [<nz>]]
        # normalise image intensity before splitting
        self._normalise_intensity(image)
        # defaults to full image if given dimension is zero
        if self._args.patchsize[0] == 0:
            self._args.patchsize[0] = image.img.shape[0]
            xlocs = [int(image.img.shape[0]/2)]
        else:
            xlocs = np.arange(int(self._args.patchsize[0]/2),
                              image.img.shape[0] - int(self._args.patchsize[0]/2),
                              self._args.patchstride[0])

        if self._args.patchsize[1] == 0:
            self._args.patchsize[1] = image.img.shape[1]
            ylocs =[int(image.img.shape[1] / 2)]
        else:
            ylocs = np.arange(int(self._args.patchsize[1]/2),
                              image.img.shape[1] - int(self._args.patchsize[1]/2),
                              self._args.patchstride[1])

        # TODO: handle when px defined but ps isnt

        # for now only patch in xy use full z depth
        zloc = int(image.img.shape[2]/2)
        zsize = image.img.shape[2]

        # define time axis
        tstring = ''
        tlocs = [0]
        if target:
            tlocs = np.arange(0,  np.max(self._states))

        for nt, ti in enumerate(tlocs):
            for nx, xi in enumerate(xlocs):
                for ny, yi in enumerate(ylocs):
                    patch_ind = ny + (nx * xlocs.__len__())
                    pixel_region = (xi, yi, zloc, self._args.patchsize[0], self._args.patchsize[1], zsize)
                    pixel_region = [str(i) for i in pixel_region]  # convert to string list
                    if target:
                        tstring = ' -Rt1 ' + str(ti) + ' -Rt2 ' + str(ti)

                    command_string = 'mirtk extract-image-region ' + \
                                     image.imagefilepath + ' ' + \
                                     self._write_paths.path_patch_img(patch_ind, '0', ww=nt, target=target) + \
                                     ' -patch ' + ' '.join(pixel_region) + \
                                     tstring
                    print(command_string)
                    subprocess.run(command_string.split())

    def _normalise_intensity(self, image, scale=['0', '3000']):
        """ Normalises intensity before patch splitting"""
        print('Normalising pixel values for %s' % image.imagefilepath)
        command_string = 'mirtk convert-image ' + image.imagefilepath + ' ' + image.imagefilepath + ' -rescale ' + ' '.join(scale)
        print(command_string)
        subprocess.run(command_string.split())

    def _extract_patches_internal(self, image, patch_size=None, patch_stride=None, target=False):
        """Extracts 2D patches with overlap and preserved geometry as NIfTI files"""
        if patch_size is None:
            patch_size = np.array([image.img.shape[0] / 2.1, image.img.shape[1] / 2.1, 0]).astype(int)

        if patch_stride is None:
            patch_stride = (patch_size * 0.3).astype(int)

        rect_list = self._patch_list_xy(image.img.shape, patch_size, patch_stride)
        self._npatches = rect_list.shape[0]
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

    def _set_resp_range(self):
        """defines which resp states are reconstructed (all | the most dense)"""
        if self._args.no_resp_recon:
            counts = np.bincount(self._states)  # how many slices in each state (including zero)
            mode_state = np.argmax(counts[1:]) + 1  # get most frequent bin +1 (excluding zero)
            self.resp_range = range(mode_state, mode_state+1)
            print('Reconstructing only state: %d' % mode_state.astype(int))
        else:
            self.resp_range = range(1, np.max(self._states) + 1)

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
