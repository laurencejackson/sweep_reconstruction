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

        if self._args.iterations > 1:
            resv = np.linspace(1.5*0.75, 0.75, self._args.iterations)
        else:
            resv = [0.75]

        # TODO read more opts from args
        opts = {'thickness': self._args.thickness,
                'ffd': self._args.free_form_deformation,
                'remote': self._args.remote,
                'resolution': resv[0]}

        self._process_patches('reconstructAngio', opts, 0)

        if self._args.iterations > 1:
            for ww in self.resp_range:
                for iteration in range(1, self._args.iterations):
                    self._target = ImageData(self._write_paths.path_combined_patches(ww, iteration-1))
                    self._extract_patches(self._target, target=True)
                    self._svr_options_init()
                    opts = {'thickness': self._args.thickness,
                            'ffd': self._args.free_form_deformation,
                            'remote': self._args.remote,
                            'resolution': resv[iteration]}
                    self._process_patches('reconstructAngio', opts, iteration)

    def _process_patches(self, function_path, opts, iteration):
        """Loop over patch directory structure and apply function"""

        # loop over resp states
        nstacks = 1  # likely to only work with one but I'll make this a variable anyway
        self._set_resp_range()
        print('Reconstruction iteration %d' % iteration)

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
            # os.rename('combined.nii.gz', self._write_paths.path_combined_patches(ww))

            if iteration == self._args.iterations - 1:
                os.rename('combined.nii.gz', self._write_paths.final_reconstruction(ww, iterations=self._args.iterations))
            else:
                os.rename('combined.nii.gz', self._write_paths.path_combined_patches(ww, iteration))

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
        graph = True
        if graph:
            target_space_path = self._target.imagefilepath
        else:
            target_space_path = self._write_paths.path_interpolated_3d_linear(1)

        command_string_1 = str('%s %s %f %d' % (function_path,
                                                target_space_path,
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
            "remote": False,

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
            "svr_only": True,
            "no_intensity_matching": True,
            "no_sr": False,
            "no_robust_statistics": False,
            "reg_log": False,
        }

    def _extract_patches(self, image, target=False):
        """Uses MIRTK extract-image-region function to extract patches"""
        # extract-image-region <input> <output> [options] -patch <i> <j> <k> <nx> [<ny> [<nz>]]
        # normalise image intensity before splitting

        normalise = True
        if normalise:
            self._normalise_intensity(image)

        # convert patch size to pixels
        self._patchsize_px = [0, 0]
        self._patchsize_px[0] = np.floor(self._args.patchsize[0] / image.nii.header['pixdim'][1]).astype(int)
        self._patchsize_px[1] = np.floor(self._args.patchsize[1] / image.nii.header['pixdim'][2]).astype(int)

        self._patchstride_px = [0, 0]
        self._patchstride_px[0] = np.floor(self._args.patchstride[0] / image.nii.header['pixdim'][1]).astype(int)
        self._patchstride_px[1] = np.floor(self._args.patchstride[1] / image.nii.header['pixdim'][2]).astype(int)

        # defaults to full image if given dimension is zero
        if self._patchsize_px[0] == 0:
            self._patchsize_px[0] = image.img.shape[0]
            xlocs = [int(image.img.shape[0]/2)]
        else:
            nstrides = 1 + np.ceil( (image.img.shape[0] - int(self._patchsize_px[0]/2) - int(self._patchsize_px[0]/2)) /
                                self._patchstride_px[0])
            xlocs = np.linspace(int(self._patchsize_px[0]/2),
                              image.img.shape[0] - int(self._patchsize_px[0]/2),
                              nstrides).astype(int)

        if self._patchsize_px[1] == 0:
            self._patchsize_px[1] = image.img.shape[1]
            ylocs =[int(image.img.shape[1] / 2)]
        else:
            nstrides = 1 + np.ceil((image.img.shape[1] - int(self._patchsize_px[1] / 2) - int(self._patchsize_px[1] / 2)) /
                               self._patchstride_px[1])
            ylocs = np.linspace(int(self._patchsize_px[1] / 2),
                                image.img.shape[1] - int(self._patchsize_px[1] / 2),
                                nstrides).astype(int)

        # for now only patch in xy use full z depth
        zlocs = np.array([int(image.img.shape[2]/2)])
        zsize = image.img.shape[2]

        self._npatches = xlocs.__len__() * ylocs.__len__() * zlocs.__len__()
        print('Creating %d patches' % self._npatches)

        # define time axis
        tstring = ''
        tlocs = [0]
        if target:
            tlocs = np.arange(0,  np.max(self._states))

        patch_ind = 0
        for nt, ti in enumerate(tlocs):
            for nz, zi in enumerate(zlocs):
                for nx, xi in enumerate(xlocs):
                    for ny, yi in enumerate(ylocs):
                        # patch_ind = ny + (nx * xlocs.__len__())
                        pixel_region = (xi, yi, zi, self._patchsize_px[0], self._patchsize_px[1], zsize)
                        pixel_region = [str(i) for i in pixel_region]  # convert to string list
                        if target:
                            tstring = ' -Rt1 ' + str(ti) + ' -Rt2 ' + str(ti)

                        command_string = 'mirtk extract-image-region ' + \
                                         image.imagefilepath + ' ' + \
                                         self._write_paths.path_patch_img(patch_ind, z=nz, ww=nt, target=target) + \
                                         ' -patch ' + ' '.join(pixel_region) + \
                                         tstring
                        print(command_string)
                        subprocess.run(command_string.split())
                        patch_ind += 1

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
            self.resp_range = range(1, np.max(self._states) + 1)  # resp range equal to tdim of target
            # self.resp_range = range(1, np.max(self._states) + 1)

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

    @staticmethod
    def _normalise_intensity(image, scale=['0', '1000']):
        """ Normalises intensity before patch splitting"""
        print('Normalising pixel values for %s' % image.imagefilepath)
        command_string = 'mirtk convert-image ' + image.imagefilepath + ' ' + image.imagefilepath + ' -rescale ' + ' '.join(scale)
        print(command_string)
        subprocess.run(command_string.split())
