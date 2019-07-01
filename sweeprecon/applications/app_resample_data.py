"""
Application module
Resamples sorted and classified data into 3D/4D respiration resolved volumes

Laurence Jackson, BME, KCL, 2019
"""

import os
import sys

from sweeprecon.ResampleData import ResampleData
from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.io.ImageData import ImageData
from sweeprecon.utilities.LogData import LogData

from sweeprecon.utilities.WritePaths import WritePaths


def app_resample_data(pipeline=False):
    """
    Resamples sorted and classified data into 3D/4D respiration resolved volumes
    :param pipeline: bool describing whether function is running in isolation or as pipeline
    :return:
    """

    print("\n__________________________ Re-sampling Data ________________________\n")
    logger = LogData()
    # Check if function if being run as part of pipeline or by itself
    if not pipeline:
        # parse arguments
        input_vars = ArgParser(description="Estimates a respiration signal from a sequence of dynamic 2D images")

        # required
        input_vars.add_input_file(required=True)

        # optional
        input_vars.add_flag_redo(required=False)
        input_vars.add_interpolater(required=False)

        # parse
        args = input_vars.parse_args()
        write_paths = WritePaths(os.path.basename(args.input))
        image = ImageData(args.input)

        # save args to logger
        logger.set_key('args', args)

    # otherwise is running as pipeline from __main__
    else:
        # load LogData
        logger.load_log_file()
        args = logger.log.args
        write_paths = WritePaths(os.path.basename(args.input))
        image = ImageData(write_paths.path_sorted)

    if not logger.log.flag_estimated_respiration or not logger.log.flag_sorted:
        print('Missing requirements: please run full pipeline through __main__')
        sys.exit()

    # set up re-sampler
    resampler = ResampleData(image,
                             logger.log.resp_states,
                             logger.log.geo_slice_locations,
                             write_paths,
                             args.interpolater
                             )

    # run re-sampling
    resampler.run()

    # save output
    logger.set_key('flag_resampled', True)
    logger.save_log_file()

    # Done
    print('Resampling complete')
