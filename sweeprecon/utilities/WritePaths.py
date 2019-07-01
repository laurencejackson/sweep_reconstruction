"""
List of file paths for various points in program - edit here to prevent bugs from renaming

Laurence Jackson, BME, KCL, 2019
"""

import os


class WritePaths(object):

    def __init__(self, args):

        self.basename = os.path.basename(args.input)
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
                         'contours_initialised' +  # basename
                         self._nii_ext  # file ext
                         )

        self.path_refined_contours = \
            os.path.join(os.getcwd(),  # cwd
                         'IMG_3D_' +  # prefix
                         'contours_refined' +  # basename
                         self._nii_ext  # file ext
                         )

        self.path_filtered_contours = \
            os.path.join(os.getcwd(),  # cwd
                         'IMG_3D_' +  # prefix
                         'contours_filtered' +  # basename
                         self._nii_ext  # file ext
                         )

        self.path_interpolated_4d = \
            os.path.join(os.getcwd(),  # cwd
                         'IMG_4D_interpolated_' +  # prefix
                         args.interpolator +  # basename
                         '_' +
                         self.basename
                         )
