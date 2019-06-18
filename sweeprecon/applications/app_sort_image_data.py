"""
Application module
sorts image data from 4D slice+dynamics to 3D with by interleaving the dynamics with the slice positions

Laurence Jackson, BME, KCL, 2019
"""

import os

from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.io.ImageData import ImageData


def app_sort_image_data(args=None):
    """
    Sorts image data from 4D slice+dynamics to 3D with by interleaving the dynamics with the slice positions
    :param args: arguments parsed through command line ( run with [-h] to see required parameters)
    :return:
    """

    print('\n________________________ Sorting image data ________________________\n')

    # Check if function if being run as part of pipeline or by itself
    if args is None:

        # parse arguments
        input_vars = ArgParser(description="Sorts image data from 4D slice+dynamics to 3D with by interleaving the "
                                           "dynamics with the slice positions")

        # required
        input_vars.add_input_file(required=True)

        # optional
        input_vars.add_flag_redo(required=False)

        # parse
        args = input_vars.parse_args()

    # local file output vars
    basename = os.path.basename(args.input)
    prefix = 'IMG_3D_'
    dirpath = os.getcwd()

    # read image data if not already done and redo not flagged
    if os.path.isfile(os.path.join(dirpath, prefix + basename)) and not args.redo:
        print('Data already sorted.')
        return

    image = ImageData(args.input)

    # sort image
    image.sort_4d_to_3d()

    # save output
    image.write_nii(basename, prefix=prefix)

    # Done
    print('Sorting data complete')

    return
