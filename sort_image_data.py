"""
Runs the application 'sort_image_data' can be as part of app_sweep_recon or run independently

Laurence Jackson, BME, KCL, 2019
"""

import sys

from sweeprecon.applications.app_sort_image_data import app_sort_image_data

if __name__ == "__main__":
    print('\n________________________ Sorting image data ________________________\n')
    app_sort_image_data()
    sys.exit()
