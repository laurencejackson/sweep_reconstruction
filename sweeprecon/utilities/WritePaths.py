"""
List of file paths for various points in program - edit here to prevent bugs from renaming

Laurence Jackson, BME, KCL, 2019
"""

import os


class WritePaths(object):

    def __init__(self, args):

        self.basename = os.path.basename(args.input)

        self._nii_ext = '.nii.gz'
        self.state = 0
        self._args = args

        # folder names
        self._resp_vols_folder = '3D_respiration_volumes'

        # create output folders
        if not os.path.exists(self._resp_vols_folder):
            os.makedirs(self._resp_vols_folder)

    # path definition functions
    def path_sorted(self):
        return os.path.join(os.getcwd(),    # cwd
                             'IMG_3D_' +     # prefix
                            self.basename   # basename
                            )

    def path_cropped(self):
        return os.path.join(os.getcwd(),    # cwd
                            'IMG_3D_' +     # prefix
                            'cropped' +     # basename
                            self._nii_ext   # file ext
                            )

    def path_initialised_contours(self):
        return os.path.join(os.getcwd(),  # cwd
                            'IMG_3D_' +  # prefix
                            'contours_initialised' +  # basename
                            self._nii_ext  # file ext
                            )

    def path_refined_contours(self):
        return os.path.join(os.getcwd(),  # cwd
                            'IMG_3D_' +  # prefix
                            'contours_refined' +  # basename
                            self._nii_ext  # file ext
                            )

    def path_filtered_contours(self):
        return os.path.join(os.getcwd(),  # cwd
                            'IMG_3D_' +  # prefix
                            'contours_filtered' +  # basename
                            self._nii_ext  # file ext
                            )

    def path_interpolated_4d(self):
        return os.path.join(os.getcwd(),  # cwd
                            'IMG_4D_interpolated_' +
                            self._args.interpolator + '_' +
                            str(self._args.kernel_dims) + 'D_kernel_' +
                            self.basename
                            )

    def path_interpolated_4d_linear(self):
        return os.path.join(os.getcwd(),  # cwd
                            'IMG_4D_linear_' +
                            self._args.interpolator + '_' +
                            self.basename
                            )

    def path_interpolated_3d(self, resp_state):
        return os.path.join(os.getcwd(),  # cwd
                     self._resp_vols_folder,
                     'IMG_3D_resp_' +
                     str(resp_state) + '_' +
                     self.basename
                     )
