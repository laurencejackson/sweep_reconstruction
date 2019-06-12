"""
Main application
Runs the reconstruction pipeline

Laurence Jackson, BME, KCL, 2019
"""


from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.applications.app_sort_image_data import app_sort_image_data


def main():
    """
    Main application function for running the full reconstruction pipeline
    :return:
    """

    print('\n_________________________ Parsing arguments ________________________\n')

    input_vars = ArgParser(description="Reconstruct respiration resolved dynamic SWEEP MRI data")

    # required
    input_vars.add_input_file(required=True)

    # optional
    input_vars.add_slice_thickness()
    input_vars.add_n_resp_states()
    input_vars.add_redo_flag()

    # parse
    args = input_vars.parse_args()
    input_vars.print_args(args)

    print('\nParsing complete.\n')

    # ________________________ Sorting image data ________________________
    app_sort_image_data(args)
    print('got to here')

    print('\n________________________ Pre-processing data _______________________\n')
    print('\n_______________________ Estimating respiration _____________________\n')
    print('\n_______________________ Respiration binning ________________________\n')
    print('\n______________________ Re-sampling image data ______________________\n')
    print('\n________________________ Splitting patches _________________________\n')
    print('\n___________________ Performing DSVR registration ___________________\n')
    print('\n_________________________ Recombining data _________________________\n')
    print('\n__________________________ Saving output ___________________________\n')

    return


if __name__ == '__main__':
    main()
