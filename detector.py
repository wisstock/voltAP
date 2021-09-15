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
                    abf_record = pyabf.ABF(file_path)

                    # CUSTOM ATTRIBUTE!
                    # add application time attribute
                    setattr(abf_record, 'appTime',
                            input_suffix[file_suffix])

                    # create Y no gap record from individual sweeps
                    no_gap_Y = []
                    for i in range(0,abf_record.sweepCount):
                        abf_record.setSweep(i)
                        [no_gap_Y.append(val) for val in abf_record.sweepY]
                    setattr(abf_record, 'sweepY_no_gap', np.array(no_gap_Y))

                    # create X timeline for no gap record
                    setattr(abf_record, 'sweepX_no_gap',
                            np.arange(len(abf_record.sweepY_no_gap))*abf_record.dataSecPerPoint)

                    record_list.append(abf_record)
                    logging.info(f'File {file} uploaded!')
    return (record_list)


def spike_detect(record, spike_h=0, spike_w=2,spike_d=10, l_lim=20, r_lim=50):
    """ Simple spike detection by peak feature and spike interval extraction.
    record - pyABF instance 
    spike_h - spike amlitude threshold (mV)
    spike_d - minimal distance between spikes (ms)
    l_lim - left limit of spike region (ms)
    r_lim - right limit of spike region (ms)
    write_spike_interval - create dictionary with time interval for each detected spike

    """
    spike_peaks, spike_prop = signal.find_peaks(record.sweepY_no_gap,
                                                height=spike_h,
                                                width=spike_w,
                                                distance=spike_d/1e3/record.dataSecPerPoint)  # index of spike peak
    # spike_time = record.sweepX_no_gap[spike_peaks]  # time of spike peak

    spike_interval = {spike_p:[spike_p - (l_lim/1e3/record.dataSecPerPoint),
                               spike_p + (r_lim/1e3/record.dataSecPerPoint)]
                      for spike_p in spike_peaks}

    logging.info(f'In {record.sweepCount} sweeps finded {len(spike_peaks)} peaks, thr. = {spike_h}mV')
    return spike_peaks, spike_interval

def otsu_baseline(record=False, vector=False):
    if record:
        counts, bin_centers = np.histogram(record.sweepY_no_gap, bins=256)
    else:
        counts, bin_centers = np.histogram(vector, bins=256)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers[:-1]) / weight1
    mean2 = (np.cumsum((counts * bin_centers[:-1])[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    otsu = bin_centers[idx]
    logging.info(f'Otsu threshold = {otsu}mV')
    return otsu


def der(vector, dt):
    """ 1st derivate.
    vector - input data (sweepY)
    dt - discretization frequency (in sec)
    """
    point_der = lambda x_2k, x_1k, x_k1, x_k2, t: (x_2k - 8*x_1k + 8*x_k1 - x_k2)/(12*t)
    return [point_der(vector[i-2], vector[i-1], vector[i+1], vector[i+2], dt)
            for i in range(2,len(vector)-2)], vector[2:-2]

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
reg = reg_list[num]


ot = otsu_baseline(reg)
otsuY = np.copy(reg.sweepY_no_gap)
otsuY[otsuY > ot] = ot

oot = otsu_baseline(vector=otsuY)
otsuY[otsuY > oot] = oot



reg_spike, reg_interval = spike_detect(reg)



# small_sweep = abf_list[abf_num].sweepY[107500:108500]

# sweepDer, sweepRes = der(small_sweep, 1/abf_list[abf_num].dataRate)
# sweepDer2, sweepRes2 = der2(small_sweep, 1/abf_list[abf_num].dataRate)

# norm = lambda x: [i / max(x) for i in x]

plt.figure(figsize=(8, 5))

for i in range(0, reg.sweepCount):
    reg.setSweep(i)
    sweep_der, sweep_adj = der(reg.sweepY, reg.dataSecPerPoint)
    plt.plot(sweep_adj, sweep_der, alpha=.5)

# plt.plot(reg.sweepX_no_gap, reg.sweepY_no_gap)
# plt.plot(reg.sweepX_no_gap, otsuY, ls=':')
# plt.plot(reg.sweepX_no_gap[reg_spike], reg.sweepY_no_gap[reg_spike], 'x')
# plt.axhline(ot, color='k', ls='--')
# # plt.plot(reg.sweepX_no_gap, reg.sweepY_no_gap)

# plt.hist(reg_hist)

plt.show()