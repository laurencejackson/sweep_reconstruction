"""
List of definitions for various points in program - edit here to prevent renaming bugs

Laurence Jackson, BME, KCL, 2019
"""

import os


class WritePaths(object):

    def __init__(self, basename):

        self.basename = basename
        self._nii_ext = '.nii.gz'

        self.path_sorted = \
            os.path.join(os.getcwd(),    # cwd
                         'IMG_3D_' +     # prefix
                         self.basename   # basename
                         )

        self.path_cropped = \
            os.path.join(os.getcwd(),    # cwd
                         'IMG_3D_' +     # prefix
                         'cropped' +     # basename
                         self._nii_ext   # file ext
                         )

        self.path_initialised_contours = \
            os.path.join(os.getcwd(),  # cwd
                         'IMG_3D_' +  # prefix
                         'initialised_contours' +  # basename
                         self._nii_ext  # file ext
                         )

        self.path_refined_contours = \
            os.path.join(os.getcwd(),  # cwd
                         'IMG_3D_' +  # prefix
                         'refined_contours' +  # basename
                         self._nii_ext  # file ext
                         )