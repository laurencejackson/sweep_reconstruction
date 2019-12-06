"""
Application module
Resamples sorted and classified data into 3D/4D respiration resolved volumes

Laurence Jackson, BME, KCL, 2019
"""

import scipy.io as sio
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
        input_vars.add_resp_method(required=False)
        input_vars.add_read_locs_matlab(required=False)

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

    if args.resp_method == 'graph' or logger.log.args.resp_method == 'graph':
        # set up re-sampler
        if args.locs_matlab is not None:
            logger.log.graph_locs = sio.loadmat(args.locs_matlab)['locs']
            logger.log.px_py = sio.loadmat(args.locs_matlab)['pxpy']
        resample_struct = (logger.log.locs, logger.log.px_py)
    else:
        resample_struct = logger.log.resp_states

    # create resampler
    resampler = ResampleData(image,
                             resample_struct,
                             image.slice_positions(),
                             write_paths,
                             args,
                             kernel_dims=args.kernel_dims,
                             n_threads=args.n_threads
                             )

    # run re-sampling
    resampler.run()

    # Define target for next pipeline reg step
    logger.set_key('target', resampler.target_out)

    # save output
    logger.set_key('flag_resampled', True)

    logger.save_log_file()

    # Done
    print('Resampling complete')
