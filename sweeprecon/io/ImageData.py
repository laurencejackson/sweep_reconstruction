"""
Input parser for command line operation
Laurence Jackson, BME, KCL 2019

"""

import nibabel as nib
import numpy as np
import os


class ImageData(object):

    def __init__(self, file_path):
        """
        Initialise ImageData object
        :param file_path: path to NIfTI image to work with
        """

        # Works only for NIfTI images for now
        if not file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
            raise IOError('Invalid input file, must be in NIfTI format (.nii or .nii.gz)')

        self._nii = nib.load(file_path)
        self._img = self._nii.get_fdata()  # default _img data to the existing data object

    def get_data(self):
        """Returns image data as numpy array"""
        return self._img

    def get_hdr(self):
        """Returns header information"""
        return self._nii.header

    def set_data(self, img):
        self._img = img

    def sort_4d_to_3d(self):
        """Sorts image data from 4D to 3D and writes new NIfTI"""

        n_dynamics = self._nii.header['dim'][4]

        if n_dynamics == 1:
            print('Image already 3D: no need to sort')
            pass

        # reshape data
        img_reshape = np.reshape(self._img, (self._img.shape[0], self._img.shape[1], self._img.shape[2] * self._img.shape[3]))

        # modify header meta keys to preserve NIfTI geometry
        self._nii.affine[:, 2] = self._nii.affine[:, 2] / n_dynamics
        self._nii.header['srow_x'][2] = self._nii.header['srow_x'][2] / n_dynamics
        self._nii.header['srow_y'][2] = self._nii.header['srow_y'][2] / n_dynamics
        self._nii.header['srow_z'][2] = self._nii.header['srow_z'][2] / n_dynamics
        self._nii.header['dim'][0] = 3
        self._nii.header['dim'][3] = img_reshape.shape[2]
        self._nii.header['dim'][4] = 1
        self._nii.header['xyzt_units'] = 2
        self._nii.header['pixdim'][3] = self._nii.header['pixdim'][3] / n_dynamics

        # time/slice used later to calculate sampling frequency
        self._nii.header['pixdim'][4] = self._nii.header['pixdim'][4] / self._img.shape[3]

        print('Converted data size from {0} to {1}'.format(self._img.shape, img_reshape.shape))
        self.set_data(img_reshape)

    def write_nii(self, filename, dirpath=None, prefix=None):
        """
        Saves NIfTI image in working directory
        :param dirpath: directory to save NIfTI file to
        :param filename: basename for file path
        :param prefix: prefix to add to file path
        :return:
        """

        if dirpath is None:
            dirpath = os.getcwd()

        save_name = os.path.join(dirpath, prefix + filename)

        print('Saving ' + prefix + filename)
        nii = nib.Nifti1Image(self._img, self._nii.affine, self._nii.header)
        nib.save(nii, save_name)
