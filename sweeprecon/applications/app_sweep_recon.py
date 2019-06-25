"""
Main application
Runs the reconstruction pipeline

Laurence Jackson, BME, KCL, 2019
"""


from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.applications.app_sort_image_data import app_sort_image_data
from sweeprecon.applications.app_estimate_respiration import app_estimate_respiration

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

    # parse
    args = input_vars.parse_args()
    input_vars.print_args(args)

    # store args in logger
    logger = LogData()
    logger.args = args
    logger.save_log_file()

    # ________________________ Sorting image data ________________________
    app_sort_image_data(pipeline=True)

    # ________________________ Pre-processing data _______________________
    # TODO outlier exclusion

    # _______________________ Estimating respiration _____________________
    app_estimate_respiration(pipeline=True)

    # ___________________ Classify respiration states ____________________

    print('\n______________________ Re-sampling image data ______________________\n')
    print('\n________________________ Splitting patches _________________________\n')
    print('\n___________________ Performing DSVR registration ___________________\n')
    print('\n_________________________ Recombining data _________________________\n')
    print('\n__________________________ Saving output ___________________________\n')

    return


if __name__ == '__main__':
    main()
