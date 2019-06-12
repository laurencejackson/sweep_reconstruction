"""
Input parser for command line operation
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
        self._parser_req = self._parser.add_argument_group('Required arguments')

    def get_parser(self):
        return self._parser

    def parse_args(self):
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
                       required=True
                       ):
        self._add_argument(dict(locals()))

    def add_slice_thickness(self,
                            option_string=("-t", "--thickness"),
                            help="thickness of acquired slice [mm]",
                            required=False,
                            default=3
                            ):
        self._add_argument(dict(locals()))

    def add_n_resp_states(self,
                         option_string=("-n", "--nstates"),
                         help="number of respiration states",
                         required=False,
                         default=4
                         ):
        self._add_argument(dict(locals()))

    def add_redo_flag(self,
                      option_string=("-r", "--redo"),
                      help="redo all steps with given arguments",
                      required=False,
                      default=False
                      ):
        self._add_argument(dict(locals()))
