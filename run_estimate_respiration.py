"""
Runs the application 'estimate_respiration'. Can be as part of app_sweep_recon or run independently

Laurence Jackson, BME, KCL, 2019
"""

import sys

from sweeprecon.applications.app_estimate_respiration import app_estimate_respiration

if __name__ == "__main__":
    app_estimate_respiration()
    sys.exit()
