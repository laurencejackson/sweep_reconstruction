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

        # Create group for named required arguments so these are listed separately
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
                       required=True
                       ):
        self._add_argument(dict(locals()))

    def add_slice_thickness(self,
                            option_string=("-t", "--thickness"),
                            metavar='',
                            help="thickness of acquired slice [mm]",
                            required=False,
                            default=3
                            ):
        self._add_argument(dict(locals()))

    def add_n_resp_states(self,
                          option_string=("-n", "--nstates"),
                          metavar='',
                          help="number of respiration states",
                          required=False,
                          default=4
                          ):
        self._add_argument(dict(locals()))

    def add_redo_flag(self,
                      option_string=("-r", "--redo"),
                      action='store_true',
                      help="redo all steps with given arguments",
                      required=False,
                      default=False
                      ):
        self._add_argument(dict(locals()))

    def add_resp_crop(self,
                      option_string=("-c", "--crop_resp"),
                      action='store_true',
                      help="automatically crop data to respiration regions, can improve respiration estimates",
                      required=False,
                      default=False
                      ):
        self._add_argument(dict(locals()))
