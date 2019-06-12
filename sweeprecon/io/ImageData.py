"""
Input parser for command line operation
Laurence Jackson, BME, KCL 2019

"""

import sys
import SimpleITK as sitk
import numpy as np


class ImageData(object):

    def __init__(self, file_path):
        # Works only for Nifti images for now
        if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
            self._image = sitk.ReadImage(file_path)
        else:
            raise Exception('Invalid input file, must be in nifti format (.nii or .nii.gz)')

    def read_image(self):
        """Reads image file"""
        return self._image.Execute()

    def get_data(self):
        """Returns data as numpy array"""
        return sitk.GetArrayFromImage(self._image).transpose()

    def get_hdr(self):
        """Returns header information"""
        pass

    def write_data(self):
        """Writes new nifti image"""
        pass

    def modify_hdr(self):
        pass
