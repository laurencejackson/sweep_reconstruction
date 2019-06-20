"""
Class storing information about reconstruction stages

Laurence Jackson, BME, KCL, 2019
"""

import os
import pickle


class LogData(object):

    def __init__(self):
        """ Checks if LogData exists in working directory and either loads existing log or creates a new one"""
        if os.path.isfile('LogData'):
            self.load_log_file()
        else:
            """Initialise fields"""
            # input info
            self.log.input_data_raw = None
            self.log.input_data_sorted = None
            self.log.input_data_cropped = None
            self.log.input_data_filtered = None

            # Geometry info
            self.log.geo_slice_locations = None

            # Respiration info
            self.log.resp_raw = None
            self.log.resp_trend = None
            self.log.resp_trace = None

    def set_key(self, key, value):
        self.log.__setattr__(key, value)

    def save_log_file(self, path=None):
        """Saves LogData to path"""
        if path is None:
            path = os.path.join(os.getcwd(), 'LogData')

        # write serialised object
        pickle.dump(self.log, open(path, "wb"))

    def load_log_file(self):
        """Loads existing LogData to memory"""
        print('Retrieving log data')
        path = os.path.join(os.getcwd(), 'LogData')

        # open file and load data
        with open(path, 'rb') as pickle_file:
            self.log = pickle.load(pickle_file)

    def print_log(self):
        # TODO
        pass

    def write_log_to_text(self):
        # TODO
        pass
