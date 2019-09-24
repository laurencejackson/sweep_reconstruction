"""
Application module
Reconstructs data using patch based SVR

Laurence Jackson, BME, KCL, 2019
"""

import sys

from sweeprecon.Reconstruction import Reconstruction
from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.io.ImageData import ImageData
from sweeprecon.utilities.LogData import LogData
from sweeprecon.utilities.WritePaths import WritePaths


def app_reconstruct_data(pipeline=False):
    """
    Reconstructs data using patch based SVR
    :param pipeline: bool describing whether function is running in isolation or as pipeline
    :return:
    """

    print("\n________________________ Reconstructing Data _______________________\n")
    logger = LogData()
    # Check if function if being run as part of pipeline or by itself
    if not pipeline:
        # parse arguments
        input_vars = ArgParser(description="Reconstructs data using SVR")

        # required
        input_vars.add_input_file(required=True)
        input_vars.add_target_file(required=True)

        # optional
        input_vars.add_flag_redo(required=False)
        input_vars.add_slice_thickness(required=False)
        input_vars.add_n_threads(required=False)
        input_vars.add_interpolator(required=False)
        input_vars.add_flag_no_resp_recon(required=False)
        input_vars.add_flag_frangi(required=False)
        input_vars.add_flag_ffd_recon(required=False)
        input_vars.add_patch_size(required=False)
        input_vars.add_patch_stride(required=False)

        # parse
        args = input_vars.parse_args()
        write_paths = WritePaths(args)
        image = ImageData(args.input)
        target = ImageData(args.target)

        # save args to logger
        logger.set_key('args', args)

    # otherwise is running as pipeline from __main__
    else:
        # load LogData
        logger.load_log_file()
        args = logger.log.args
        write_paths = WritePaths(args)
        image = ImageData(write_paths.path_sorted())
        target = ImageData(logger.log.target)

    if not logger.log.flag_estimated_respiration or not logger.log.flag_sorted or not logger.log.flag_resampled:
        print('Missing requirements: please run full pipeline through __main__')
        sys.exit()

    # set up re-sampler
    reconstructor = Reconstruction(image,
                                   target,
                                   write_paths,
                                   logger.log.resp_states,
                                   args
                                   )

    # run re-sampling
    reconstructor.run()
