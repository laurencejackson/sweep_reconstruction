"""
List of file paths for various points in program - edit here to prevent bugs from renaming

Laurence Jackson, BME, KCL, 2019
"""

import os


class WritePaths(object):

    def __init__(self, args):

        self.basename = os.path.basename(args.input)

        # remove tags from basename
        prestrings = ('IMG_3D_', 'IMG_4D_')
        for substring in prestrings:
            self.basename = self.basename.replace(substring, '')

        self._nii_ext = '.nii.gz'
        self.state = 0
        self._args = args
        self.patch_dir_list = []

        # create output folders
        self._exclude_lists_folder = 'exclude_lists'
        self._resp_vols_linear_folder = '3D_respiration_volumes_linear'
        self._resp_vols_folder = '3D_respiration_volumes_' + str(args.interpolator)
        self._patches_folder = 'patches'
        self._recon_folder = 'RECON'


    # path definition functions
    def path_sorted(self, pre=''):
        return os.path.join(os.getcwd(),    # cwd
                            pre +
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
                            'contours_initialised' +
                            self._nii_ext  # file ext
                            )

    def path_refined_contours(self):
        return os.path.join(os.getcwd(),  # cwd
                            'IMG_3D_' +  # prefix
                            'contours_refined' +
                            self._nii_ext  # file ext
                            )

    def path_filtered_contours(self):
        return os.path.join(os.getcwd(),  # cwd
                            'IMG_3D_' +  # prefix
                            'contours_filtered' +
                            self._nii_ext  # file ext
                            )

    def path_interpolated_4d(self, pre=''):
        return os.path.join(os.getcwd(),  # cwd
                            pre +
                            'IMG_4D_interpolated_' +
                            self._args.interpolator + '_' +
                            self.basename
                            )

    def path_interpolated_4d_linear(self, pre=''):
        return os.path.join(os.getcwd(),  # cwd
                            pre +
                            'IMG_4D_linear_' +
                            self.basename
                            )

    def path_interpolated_3d_linear(self, resp_state):
        if not os.path.exists(self._resp_vols_linear_folder):
            os.makedirs(self._resp_vols_linear_folder)
        return os.path.join(os.getcwd(),  # cwd
                     self._resp_vols_linear_folder,
                     'IMG_3D_resp_' +
                     str(resp_state) + '_' +
                     self.basename
                     )

    def path_interpolated_3d(self, resp_state):
        if not os.path.exists(self._resp_vols_folder):
            os.makedirs(self._resp_vols_folder)
        return os.path.join(os.getcwd(),  # cwd
                     self._resp_vols_folder,
                     'IMG_3D_resp_' +
                     str(resp_state) + '_' +
                     self.basename
                     )

    def path_exclude_file(self, ww):
        if not os.path.exists(self._exclude_lists_folder):
            os.makedirs(self._exclude_lists_folder)
        return os.path.join(os.getcwd(),  # cwd
                            self._exclude_lists_folder,
                            'exclude_list_' +  # prefix
                            str(ww) +  # basename
                            '.txt'  # file ext
                            )

    def path_patch_img(self, xy, z, ww=0, target=False):

        if not os.path.exists(self._patches_folder):
            os.makedirs(self._patches_folder)
        patch_folder = 'IMG_3D_patch_xy' + str(xy) + '_z' + str(z)
        if not os.path.exists(os.path.join(os.getcwd(), self._patches_folder, patch_folder)):
            os.makedirs(os.path.join(os.getcwd(), self._patches_folder, patch_folder))

        if os.path.join(os.getcwd(), self._patches_folder, patch_folder) not in self.patch_dir_list:
            self.patch_dir_list.append(os.path.join(os.getcwd(), self._patches_folder, patch_folder))

        if target:
            resp_string = '_' + str(ww + 1)
            tgt_string = 'target_'
        else:
            resp_string = ''
            tgt_string = ''

        return os.path.join(os.getcwd(),  # cwd
                            self._patches_folder,
                            patch_folder,
                            tgt_string +
                            'IMG_3D_patch_xy' + str(xy) + '_z' + str(z) + resp_string +
                            self._nii_ext
                            )

    def path_patch_txt(self, xy, z, ww):
        if not os.path.exists(self._patches_folder):
            os.makedirs(self._patches_folder)
        patch_folder = 'IMG_3D_patch_xy' + str(xy) + '_z' + str(z)
        if not os.path.exists(os.path.join(os.getcwd(), self._patches_folder, patch_folder)):
            os.makedirs(os.path.join(os.getcwd(), self._patches_folder, patch_folder))
        return os.path.join(os.getcwd(),  # cwd
                            self._patches_folder,
                            patch_folder,
                            'IMG_3D_patch_xy' + str(xy) + '_z' + str(z) +
                            '_excludes_' +
                            str(ww) +
                            '.txt'
                            )

    def path_patch_recon(self, basename, ww):
        if not os.path.exists(self._recon_folder):
            os.makedirs(self._recon_folder)
        if not os.path.exists(os.path.join(os.getcwd(), self._recon_folder, basename)):
            os.makedirs(os.path.join(os.getcwd(), self._recon_folder, basename))
        return os.path.join(os.getcwd(),  # cwd
                            self._recon_folder,
                            basename,
                            'RECON_' +
                            basename +
                            str(ww) +
                            self._nii_ext
                            )

    def path_combined_patches(self, resp_state, pre=''):
        return os.path.join(os.getcwd(),  # cwd
                            self._recon_folder,
                            pre +
                            'IMG_3D_combined_' +
                            str(resp_state) +
                            self._nii_ext
                            )
