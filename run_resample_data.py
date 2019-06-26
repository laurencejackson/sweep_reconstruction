"""
Runs the application 'resample_data'. Can be as part of app_sweep_recon or run independently

Laurence Jackson, BME, KCL, 2019
"""

import sys

from sweeprecon.applications.app_resample_data import app_resample_data

if __name__ == "__main__":
    app_resample_data()
    sys.exit()
