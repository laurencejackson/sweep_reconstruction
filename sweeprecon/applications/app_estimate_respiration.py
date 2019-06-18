"""
Application module
Estimates a respiration signal from a sequence of dynamic 2D images

Laurence Jackson, BME, KCL, 2019
"""


from sweeprecon.io.ArgParser import ArgParser
from sweeprecon.EstimateRespiration import EstimateRespiration
from sweeprecon.io.ImageData import ImageData


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
        input_vars.add_flag_redo(required=False)
        input_vars.add_flag_disable_resp_crop(required=False)

        # parse
        args = input_vars.parse_args()

        # load image
        image = ImageData(args.input)

    else:
        # if running through pipeline, make sure input argument is 3D
        image = ImageData(args.input)
        # args.input = 'IMG_3D_'  # Check input file is 3D

    # check if already done or redo flagged
    # TODO

    # Estimate respiration
    resp = EstimateRespiration(image,
                               method='body_area',  # currently only body_area but space for other methods
                               disable_crop_data=args.disable_crop)
    resp.run()

    # Plot and save summary of respiration

    # Done
    print('Estimating respiration complete')

    return
