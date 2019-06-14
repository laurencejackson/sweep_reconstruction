"""
Runs the application 'sort_image_data'. Can be as part of app_sweep_recon or run independently

Laurence Jackson, BME, KCL, 2019
"""

import sys

from sweeprecon.applications.app_sort_image_data import app_sort_image_data

if __name__ == "__main__":
    app_sort_image_data()
    sys.exit()
