"""
Application module
Resamples sorted and classified data into 3D/4D respiration resolved volumes

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


def app_resample_data(pipeline=False):
    """
    Resamples sorted and classified data into 3D/4D respiration resolved volumes
    :param pipeline: bool describing whether function is running in isolation or as pipeline
    :return:
    """
