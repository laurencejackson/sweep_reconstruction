"""
Runs full reconstruction pipeline

Laurence Jackson, BME, KCL, 2019
"""

import sys

from sweeprecon.applications.app_sweep_recon import main

if __name__ == "__main__":
    print('Running full reconstruction...')
    main()
    sys.exit()
