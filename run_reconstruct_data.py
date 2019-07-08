"""
Runs the application 'reconstruct_data'. Can be as part of app_sweep_recon or run independently

Laurence Jackson, BME, KCL, 2019
"""

import sys

from sweeprecon.applications.app_reconstruct_data import app_reconstruct_data

if __name__ == "__main__":
    app_reconstruct_data()
    sys.exit()
