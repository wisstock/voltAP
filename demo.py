 #!/usr/bin/env python3
""" Copyright © 2021 Borys Olifirov
AP detector test functions.

"""

import sys
import os
import logging

import yaml

import numpy as np
import pandas as pd
import scipy
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt

import pyabf


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)


data_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data')
res_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'results')
if not os.path.exists(res_path):
    os.makedirs(res_path)

def ABFpars(path):
    """ Read methadata YAML file with records siffixes
    and feature (time after aplication) value.
    Return list of pyABF instances.

    """
    # reading methadata YAML file
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.yml') or file.endswith('.yaml'):
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    input_suffix = yaml.safe_load(f)
                logging.info(f'Methadata file {file} uploaded!')

    # reading ABF files according to suffixes list from YAML file
    record_list = []
    for root, dirs, files in os.walk(path):
            for file in files:
                file_suffix = file.split('.')[0]
                file_suffix = file_suffix.split('_')[-1]
                if file_suffix in input_suffix.keys():
                    file_path = os.path.join(root, file)
                    record_list.append(spikeDet(file_path, file, input_suffix[file_suffix]))
    return (record_list)


class spikeDet():
    def __init__(self, record_path, record_name, app_time):
        self.record = pyabf.ABF(record_path)
        self.record_name = record_name.split('.')[0]

        # add application time attribute
        self.app_time = app_time

        # create Y no gap record from individual sweeps
        self.sweepY_no_gap = np.array([])
        for i in range(0,self.record.sweepCount):
            self.record.setSweep(i)
            np.append(self.sweepY_no_gap, [val for val in self.record.sweepY])

        # create X timeline for no gap record
        self.sweepX_no_gap = np.arange(len(self.sweepY_no_gap))*self.record.dataSecPerPoint

        logging.info(f'File {self.record_name} uploaded!')

    def no_gap_spike_detect(self, spike_h=0, l_lim=20, r_lim=50):
        """ Simple spike detection by peak feature and spike interval extraction.
        spike_h - spike amlitude threshold (mV)
        l_lim - left limit of spike region (ms)
        r_lim - right limit of spike region (ms)
        write_spike_interval - create dictionary with time interval for each detected spike

        """
        self.spike_peaks, self.spike_prop = signal.find_peaks(self.sweepY_no_gap, height=spike_h)  # index of spike peak
        self.spike_time = self.sweepX_no_gap[self.spike_peaks]  # time of spike peak

        logging.info(f'In {self.record.sweepCount} sweeps finded {len(self.spike_peaks)} peaks, thr. = {spike_h}mV')

        self.spike_interval = {spike_p:[spike_p - (l_lim/1e3/self.record.dataSecPerPoint),
                                        spike_p + (r_lim/1e3/self.record.dataSecPerPoint)]
                               for spike_p in self.spike_peaks}

    def der(self):
        """ 1st derivate.
        vector - input data (sweepY)
        dt - discretization frequency (in sec)

        """
        point_der = lambda x_2k, x_1k, x_k1, x_k2, t: (x_2k - 8*x_1k + 8*x_k1 - x_k2)/(12*t)

        vector = self.sweepY_no_gap
        time_vector = self.sweepX_no_gap
        dt = self.record.dataSecPerPoint

        self.der = [point_der(vector[i-2], vector[i-1], vector[i+1], vector[i+2], dt)
                    for i in range(2,len(vector)-2)]
        self.der_adj_vector = vector[2:-2]
        self.der_adj_time_vector = time_vector[2:-2]

    def der2(vector, dt):
        """ 2nd derivate.
        vector - input data (sweepY)
        dt - discretization frequency (in sec)

        """
        point_der2 = lambda x_2k, x_1k, x_k, x_k1, x_k2, t: (-x_2k + 16*x_1k - 30*x_k + 16*x_k1 - x_k2)/(12 * t*t)
        return [point_der2(vector[i-2], vector[i-1], vector[i], vector[i+1], vector[i+2], dt)
                for i in range(2,len(vector)-2)], vector[2:-2]


reg_list = ABFpars(data_path)
num = 0
reg = reg_list[0]
reg.no_gap_spike_detect()

# small_sweep = abf_list[num].sweepY[107500:108500]
# sweepDer, sweepRes = der(small_sweep, 1/abf_list[num].dataRate)
# sweepDer2, sweepRes2 = der2(small_sweep, 1/abf_list[num].dataRate)

plt.figure(figsize=(8, 5))
# plt.xlim(left=107500, right=108500)

plt.plot(reg.sweepX_no_gap, reg.sweepY_no_gap)

plt.show()