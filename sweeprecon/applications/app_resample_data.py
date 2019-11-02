"""
Application module
Resamples sorted and classified data into 3D/4D respiration resolved volumes

Laurence Jackson, BME, KCL, 2019
"""

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
        input_vars.add_interpolator(required=False)
        input_vars.add_kernel_dims(required=False)
        input_vars.add_n_threads(required=False)
        input_vars.add_flag_frangi(required=False)

        # parse
        args = input_vars.parse_args()
        write_paths = WritePaths(args)
        image = ImageData(args.input)

        # save args to logger
        logger.set_key('args', args)

    # otherwise is running as pipeline from __main__
    else:
        # load LogData
        logger.load_log_file()
        args = logger.log.args
        write_paths = WritePaths(args)
        image = ImageData(write_paths.path_sorted())

    if not logger.log.flag_estimated_respiration or not logger.log.flag_sorted:
        print('Missing requirements: please run full pipeline through __main__')
        sys.exit()

    if args.resp_method == 'graph':
        resampler = ResampleData(image,
                                 logger.log.graph_locs,
                                 logger.log.geo_slice_locations,
                                 write_paths,
                                 args,
                                 kernel_dims=args.kernel_dims,
                                 n_threads=args.n_threads
                                 )

    elif args.resp_method =='ba':
        # set up re-sampler
        resampler = ResampleData(image,
                                 logger.log.resp_states,
                                 logger.log.geo_slice_locations,
                                 write_paths,
                                 args,
                                 kernel_dims=args.kernel_dims,
                                 n_threads=args.n_threads
                                 )

    # run re-sampling
    resampler.run()

    # define target volume for reconstruction
    if args.interpolator is not 'fast_linear':
        print('setting fast_linear interpolation to target')
        logger.set_key('target', write_paths.path_interpolated_4d())
    else:
        print('setting ' + args.interpolator + ' interpolation to target')
        logger.set_key('target', write_paths.path_interpolated_4d_linear())

    # save output
    logger.set_key('flag_resampled', True)

    logger.save_log_file()

    # Done
    print('Resampling complete')
