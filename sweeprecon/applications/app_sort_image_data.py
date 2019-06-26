"""
Application module
sorts image data from 4D slice+dynamics to 3D with by interleaving the dynamics with the slice positions

Laurence Jackson, BME, KCL, 2019
"""

import os

from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.io.ImageData import ImageData
from sweeprecon.utilities.LogData import LogData
from sweeprecon.utilities.WritePaths import WritePaths


def app_sort_image_data(pipeline=False):
    """
    Sorts image data from 4D slice+dynamics to 3D with by interleaving the dynamics with the slice positions
    :param pipeline: bool describing whether function is running in isolation or as pipeline
    :return:
    """

    print('\n________________________ Sorting image data ________________________\n')
    logger = LogData()
    # Check if function if being run as part of pipeline or by itself
    if not pipeline:

        # parse arguments
        input_vars = ArgParser(description="Sorts image data from 4D slice+dynamics to 3D with by interleaving the "
                                           "dynamics with the slice positions")

        # required
        input_vars.add_input_file(required=True)

        # optional
        input_vars.add_flag_redo(required=False)

        # parse
        args = input_vars.parse_args()
        # save args to logger
        logger.set_key('args', args)

    # otherwise is running as pipeline from __main__
    else:
        # load LogData
        logger.load_log_file()
        args = logger.log.args

    # initialise write paths
    write_paths = WritePaths(os.path.basename(args.input))

    # logging
    logger = LogData()
    logger.set_key('input_data_raw', args.input)

    # read image data if not already done and redo not flagged
    if os.path.isfile(write_paths.path_sorted) and not args.redo:
        print('Data already sorted.')
        return

    image = ImageData(args.input)

    # sort image
    image.sort_4d_to_3d()

    # save output
    image.write_nii(write_paths.path_sorted)

    # record output
    logger.set_key('input_data_sorted', write_paths.path_sorted)
    logger.set_key('geo_slice_locations', image.slice_positions())
    logger.set_key('args', args)

    # log complete
    logger.set_key('flag_sorted', True)
    logger.save_log_file()

    # Done
    print('Sorting data complete')

    return
