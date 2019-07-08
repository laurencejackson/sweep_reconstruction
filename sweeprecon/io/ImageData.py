"""
Input parser for command line operation

Laurence Jackson, BME, KCL 2019
"""

import os
import nibabel as nib
import numpy as np


class ImageData(object):

    def __init__(self, file_path):
        """
        Initialise ImageData object
        :param file_path: path to NIfTI image to create object from
        """

        # for now works only with NIfTI images
        if not (file_path.endswith('.nii.gz') or file_path.endswith('.nii')):
            raise IOError('Invalid input file, must be in NIfTI format (.nii or .nii.gz)')

        self.nii = nib.load(file_path)
        self.img = self.nii.get_fdata()  # default _img data to the existing data object

    def get_data(self):
        """Returns image data as numpy array"""
        return self.img

    def reset_data(self):
        self.img = self.nii.get_fdata()

    def get_hdr(self):
        """Returns header information"""
        return self.nii.header

    def set_data(self, img):
        self.img = img

    def sort_4d_to_3d(self):
        """Sorts image data from 4D to 3D and writes new NIfTI"""

        n_dynamics = self.nii.header['dim'][4]

        if n_dynamics == 1:
            print('Image already 3D: no need to sort')
            return

        # reshape data
        img_reshape = np.reshape(self.img,
                                 (self.img.shape[0], self.img.shape[1], self.img.shape[2] * self.img.shape[3]))

        print('Converting data size from {0} to {1}'.format(self.img.shape, img_reshape.shape))

        # modify header meta keys to preserve NIfTI geometry
        self.nii.affine[:, 2] = self.nii.affine[:, 2] / n_dynamics
        self.nii.header["pixdim"][3] = self.nii.header["pixdim"][3] / n_dynamics

        # set reshaped data
        self.set_data(img_reshape)

    def write_nii(self, path):
        """
        Saves NIfTI image in working directory
        :param path: path to save NIfTI file to
        :return:
        """

        print('Saving ' + path)
        nii = nib.Nifti1Image(self.img, self.nii.affine, self.nii.header)
        nib.save(nii, path)

    def get_fs(self):
        """Estimates the sampling frequency in Hz from the NIfTI header"""
        if self.nii.header['pixdim'][4] == 0:
            print('No temporal pixel information - using default sampling frequency (1.0Hz)')
            fs = 1.0
        else:
            fs = 1 / self.nii.header['pixdim'][4]
        return fs

    def square_crop(self, rect=None, crop_z=None):
        """
        Crops image data in xyz and modifies header transform to preserve geometry
        :param rect: 2x2 matrix defining dx and dy
        :param crop_z: defines start and end slices of crop in z [start, end]
        :return:
        """

        if rect is None:
            print("Cropping region not defined - no crop performed")
            return

        # define xy cropping region
        dx = rect[1, 0] - rect[0, 0]
        dy = rect[1, 1] - rect[0, 1]

        img_mask = np.zeros(self.img.shape)
        img_mask[rect[0, 1]:rect[0, 1] + dy, rect[0, 0]:rect[0, 0] + dx, :] = 1

        # Find co-ordinates of pixels within pixel mask
        coords = np.array(np.where(img_mask))
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1

        if crop_z is not None:
            start[2] = crop_z[0]
            end[2] = crop_z[1]

        # Pad with one voxel to avoid re-sampling problems
        start = np.maximum(start - 1, 0)
        end = np.minimum(end + 1, self.img.shape[:3])
        slices = [slice(s, e) for s, e in zip(start, end)]
        self.set_data(self.img[tuple(slices)])

        new_origin_voxel = np.array([s.start for s in slices])
        new_origin = self.nii.affine[:3, 3] + self.nii.affine[:3, :3].dot(new_origin_voxel)

        self.nii.affine[:3, :3] = self.nii.affine[:3, :3]
        self.nii.affine[:3, 3] = new_origin

        self.nii.header['dim'][[1, 2, 3]] = self.img.shape

    def slice_positions(self):
        """returns vector of slice positions relative to origin [in mm]"""
        return np.linspace(0, (self.nii.header["pixdim"][3] *
                               self.img.shape[2]) + self.nii.header["pixdim"][3],
                           self.img.shape[2])
