"""
Main application
Runs the full reconstruction pipeline

Laurence Jackson, BME, KCL, 2019
"""


from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.applications.app_sort_image_data import app_sort_image_data
from sweeprecon.applications.app_estimate_respiration import app_estimate_respiration
from sweeprecon.applications.app_resample_data import app_resample_data
from sweeprecon.utilities.LogData import LogData


def main():
    """
    Main application function for running the full reconstruction pipeline
    :return:
    """

    # _________________________ Parsing arguments ________________________

    input_vars = ArgParser(description="Reconstruct respiration resolved dynamic SWEEP MRI data")

    # required
    input_vars.add_input_file(required=True)

    # optional
    input_vars.add_slice_thickness(required=False)
    input_vars.add_n_resp_states(required=False)
    input_vars.add_flag_redo(required=False)
    input_vars.add_flag_disable_resp_crop(required=False)
    input_vars.add_interpolator(required=False)
    input_vars.add_kernel_dims(required=False)
    input_vars.add_n_threads(required=False)

    # parse
    args = input_vars.parse_args()
    input_vars.print_args(args)

    # store args in logger
    logger = LogData()
    logger.set_key('args', args)
    logger.save_log_file()

    # ________________________ Sorting image data ________________________
    app_sort_image_data(pipeline=True)

    # ________________________ Pre-processing data _______________________
    # TODO outlier exclusion

    # _______________________ Estimating respiration _____________________
    app_estimate_respiration(pipeline=True)

    # ______________________ Re-sampling image data ______________________
    app_resample_data(pipeline=True)

    return


if __name__ == '__main__':
    main()
