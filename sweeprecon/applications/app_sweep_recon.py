"""
Main application function for running the reconstruction pipeline

"""

from sweeprecon.io.ArgParser import ArgParser


def main():
    """
    Main application function for running the reconstruction pipeline
    :return:
    """

    # parse command line input arguments
    input_vars = ArgParser(description="Reconstruct respiration resolved dynamic SWEEP MRI data")

    # required
    input_vars.add_input_file(required=True)

    # optional
    input_vars.add_slice_thickness(required=False)
    input_vars.add_n_resp_states(required=False)
    input_vars.add_redo_flag(required=False)

    input_vars.parse_args()

    return


if __name__ == '__main__':
    main()
