"""
Application module
Estimates a respiration signal from a sequence of dynamic 2D images

Laurence Jackson, BME, KCL, 2019
"""

import os

from sweeprecon.EstimateRespiration import EstimateRespiration
from sweeprecon.ClassifyRespiration import ClassifyRespiration
from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.io.ImageData import ImageData
from sweeprecon.utilities.LogData import LogData
from sweeprecon.utilities.PlotFigures import plot_respiration_summary
from sweeprecon.utilities.WritePaths import WritePaths


def app_estimate_respiration(pipeline=False):
    """
    Estimates a respiration signal from a sequence of dynamic 2D images
    :param pipeline: bool describing whether function is running in isolation or as pipeline
    :return:
    """

    print("\n_______________________ Estimating respiration _____________________\n")
    logger = LogData()
    # Check if function if being run as part of pipeline or by itself
    if not pipeline:
        # parse arguments
        input_vars = ArgParser(description="Estimates a respiration signal from a sequence of dynamic 2D images")

        # required
        input_vars.add_input_file(required=True)

        # optional
        input_vars.add_flag_redo(required=False)
        input_vars.add_flag_disable_resp_crop(required=False)
        input_vars.add_n_resp_states(required=False)
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

    # make sure input argument is 3D and check if already done or redo flagged
    # TODO

    # Estimate respiration
    resp = EstimateRespiration(image,
                               write_paths,
                               method='body_area',  # currently only body_area but space for other methods,
                               disable_crop_data=args.disable_crop,
                               )
    resp.run()

    # classify resp states
    classifier = ClassifyRespiration(resp.resp_trace)
    classifier.classify_states(args.nstates)

    # Plot and save summary of respiration
    plot_respiration_summary(resp.resp_raw, resp.resp_trend, resp.resp_trace, classifier.index)

    # record output
    logger.set_key('resp_raw', resp.resp_raw)
    logger.set_key('resp_trend', resp.resp_trend)
    logger.set_key('resp_trace', resp.resp_trace)
    logger.set_key('resp_states', classifier.index)

    # log complete
    logger.set_key('flag_estimated_respiration', True)
    logger.save_log_file()

    # Done
    print('Estimating respiration complete')

    return
