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

        # Create group for named required arguments so thesargparse.ArgumentParsere are listed separately
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
                       metavar='',
                       help="path to input file",
                       required=True,
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

    def add_flag_redo(self,
                      option_string=("-r", "--redo"),
                      action='store_true',
                      help="redo all steps with given arguments",
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
