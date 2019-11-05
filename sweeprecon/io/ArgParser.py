"""
Input parser for command line operation

Laurence Jackson, BME, KCL 2019
"""

import argparse


class ArgParser(object):

    def __init__(self, description=None):

        # add program description
        kwargs = {}
        if description is not None:
            kwargs['description'] = description

        # initialise _parser object
        self._parser = argparse.ArgumentParser(**kwargs)

        # Create group for named required arguments listed separately
        self._parser_req = self._parser.add_argument_group('required arguments')

    @staticmethod
    def print_args(args):
        for arg in sorted(vars(args)):
            print('%s:   %s' % (arg, getattr(args, arg)))

    def parse_args(self):
        """Parse input arguments"""
        return self._parser.parse_args()

    def _add_argument(self, allvars):
        # Ignore self variable and extract name of argument to add
        allvars.pop('self')
        option_string = allvars.pop('option_string')

        # Add additional key word arguments
        kwargs = {}
        for key, value in allvars.items():
            kwargs[key] = value

        if kwargs['required']:
            # Add required argument to _parser
            self._parser_req.add_argument(*option_string, **kwargs)

        else:
            if 'default' in kwargs.keys():
                txt_default = " [default: %s]" % (str(kwargs['default']))
                kwargs['help'] += txt_default

            # Add optional argument to _parser
            self._parser.add_argument(*option_string, **kwargs)

    # ___________________________________________________________________
    # ____________________ List of input definitions ____________________

    def add_input_file(self,
                       option_string=("-i", "--input"),
                       help="path to input file",
                       required=True,
                       type=str
                       ):
        self._add_argument(dict(locals()))

    def add_target_file(self,
                       option_string=("-tt", "--target"),
                       help="path to target file",
                       required=False,
                       type=str
                       ):
        self._add_argument(dict(locals()))

    def add_slice_thickness(self,
                            option_string=("-z", "--thickness"),
                            metavar='',
                            help="thickness of acquired slice [mm]",
                            required=False,
                            default=2.5,
                            type=float
                            ):
        self._add_argument(dict(locals()))

    def add_crop_fraction(self,
                          option_string=("-cf", "--crop_fraction"),
                          metavar='',
                          help="fraction of image cropped to segment body area for respiration",
                          required=False,
                          default=0.4,
                          type=float
                          ):
        self._add_argument(dict(locals()))

    def add_ba_method(self,
                      option_string=("-ba", "--ba_method"),
                      metavar='',
                      help="method for segmentation body area (gac or cv)",
                      required=False,
                      default='gac',
                      type=str
                      ):
        self._add_argument(dict(locals()))

    def add_resp_method(self,
                      option_string=("-rs", "--resp_method"),
                      metavar='',
                      help="method for respiration correction (ba or graph)",
                      required=False,
                      default='graph',
                      type=str
                      ):
        self._add_argument(dict(locals()))

    def add_read_locs_matlab(self,
                      option_string=("-rlm", "--locs_matlab"),
                      metavar='',
                      help="read a mat file for locs matrix (filepath)",
                      required=False,
                      default=None,
                      type=str
                      ):
        self._add_argument(dict(locals()))

    def add_n_resp_states(self,
                          option_string=("-n", "--nstates"),
                          metavar='',
                          help="number of respiration states",
                          required=False,
                          default=4,
                          type=int
                          ):
        self._add_argument(dict(locals()))

    def add_interpolator(self,
                         option_string=("-x", "--interpolator"),
                         metavar='',
                         help="choose interpolater to use [options: 'fast_linear' or 'rbf']",
                         required=False,
                         default='fast_linear'
                         ):
        self._add_argument(dict(locals()))

    def add_kernel_dims(self,
                        option_string=("-k", "--kernel_dims"),
                        metavar='',
                        help="Size of interpolation kernel (recommended values 2->4) [default: 2]",
                        required=False,
                        type=int,
                        default=1
                        ):
        self._add_argument(dict(locals()))

    def add_n_threads(self,
                      option_string=("-t", "--n_threads"),
                      metavar='',
                      help="number of processor threads to use [default: max(1, N_cpu - 1)]",
                      required=False,
                      type=int,
                      default=0
                      ):
        self._add_argument(dict(locals()))

    def add_recon_iterations(self,
                      option_string=("-it", "--iterations"),
                      metavar='',
                      help="number of reconstruction iterations (loops over full +patch recon)",
                      required=False,
                      type=int,
                      default=1
                      ):
        self._add_argument(dict(locals()))

    def add_patch_size(self,
                       option_string=("-px", "--patchsize"),
                       metavar='',
                       help="size of patches used for reconstruction in mm [default: use full image]",
                       nargs=2,
                       required=False,
                       type=int,
                       default=[0, 0]
                       ):
        self._add_argument(dict(locals()))

    def add_patch_stride(self,
                         option_string=("-ps", "--patchstride"),
                         metavar='',
                         help="stride of patches used for reconstruction in mm [default: use full image]",
                         nargs=2,
                         required=False,
                         type=int,
                         default=[0, 0]
                         ):
        self._add_argument(dict(locals()))

    def add_flag_redo(self,
                      option_string=("-r", "--redo"),
                      action='store_true',
                      help="redo all steps with given arguments",
                      required=False,
                      default=False,
                      ):
        self._add_argument(dict(locals()))

    def add_flag_no_resp_recon(self,
                               option_string=("-nrr", "--no_resp_recon"),
                               action='store_true',
                               help="reconstruct only the most dense respiration state",
                               required=False,
                               default=False,
                               ):
        self._add_argument(dict(locals()))

    def add_flag_no_auto_crop(self,
                               option_string=("-nac", "--no_auto_crop"),
                               action='store_true',
                               help="default resp crop to center of image",
                               required=False,
                               default=False,
                               ):
        self._add_argument(dict(locals()))

    def add_flag_ffd_recon(self,
                           option_string=("-ffd", "--free_form_deformation"),
                           action='store_true',
                           help="use free form deformation for reconstruction",
                           required=False,
                           default=False,
                           ):
        self._add_argument(dict(locals()))

    def add_flag_remote_recon(self,
                           option_string=("-remote", "--remote"),
                           action='store_true',
                           help="use remote flag in reconstruction",
                           required=False,
                           default=False,
                           ):
        self._add_argument(dict(locals()))

    def add_flag_disable_resp_crop(self,
                                   option_string=("-c", "--disable_crop"),
                                   action='store_false',
                                   help="disable automatic cropping of data to respiration regions",
                                   required=False,
                                   default=False,
                                   ):
        self._add_argument(dict(locals()))

    def add_flag_frangi(self,
                        option_string=("-fr", "--frangi"),
                        action='store_true',
                        help="output frangi filtered versions of images",
                        required=False,
                        default=False,
                        ):
        self._add_argument(dict(locals()))
