"""
Application module
Estimates a respiration signal from a sequence of dynamic 2D images

Laurence Jackson, BME, KCL, 2019
"""


from sweeprecon.io.ArgParser import ArgParser


def app_estimate_respiration(args=None):
    """
    Estimates a respiration signal from a sequence of dynamic 2D images
    :param args: arguments parsed through command line ( run with [-h] to see required parameters)
    :return:
    """

    print("\n_______________________ Estimating respiration _____________________\n")

    # Check if function if being run as part of pipeline or by itself
    if args is None:
        # parse arguments
        input_vars = ArgParser(description="Estimates a respiration signal from a sequence of dynamic 2D images")

        # required
        input_vars.add_input_file(required=True)

        # optional
        input_vars.add_redo_flag(required=False)

        # parse
        args = input_vars.parse_args()

    # Estimate respiration

    # Done
    print('Estimating respiration complete')

    return
