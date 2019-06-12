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
    Sort image data
    :return:
    """

    # Check if function if being run as part of pipeline or by itself
    if not args:

        # parse arguments
        input_vars = ArgParser(description="Sorts image data such that the temporal dimension is interleaved "
                                           "with the slice direction creating a 3D output")

        # required
        input_vars.add_input_file(required=True)

        # optional
        input_vars.add_redo_flag()

        # parse
        args = input_vars.parse_args()

    # read image data if not already done and redo not flagged
    image = ImageData(args.input)
    img = image.get_data()

    # save output

    # create rescat object

    return
