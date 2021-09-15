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
                    setattr(abf_record, 'fileName', file.split('.')[0])

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

def spike_detect(record, spike_h=0, spike_w=2, spike_d=10, l_lim=False, r_lim=False, lim_adj=15):
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
    
    # minimal spike distance estimation
    if not l_lim:
        l_diff = np.absolute(np.append(spike_peaks, 0) - np.insert(spike_peaks, 0, 0))
        l_lim = min(l_diff[1:-2])
        if l_lim > 1000:  # 50 ms
            l_lim = 400  # 20 ms
            logging.info('Left limit too large, l_lim = 20ms')
        else:
            l_lim = l_lim - lim_adj
            logging.info(f'l_lim = {round(l_lim*record.dataSecPerPoint*1e3, 1)}ms')

        r_diff = np.absolute(np.insert(spike_peaks, 0, 0) - np.append(spike_peaks, 0))
        r_lim = min(r_diff[1:-2])
        if r_lim > 1000:  # 50 ms
            r_lim = 400  # 20 ms
            logging.info('Right limit too large, r_lim = 50ms')
        else:
            r_lim = r_lim - lim_adj
            logging.info(f'r_lim = {round(r_lim*record.dataSecPerPoint*1e3, 1)}ms')

    spike_interval = {spike_p:[int(spike_p - l_lim),
                               int(spike_p + r_lim)]
                      for spike_p in spike_peaks}

    logging.info(f'In {record.sweepCount} sweeps {len(spike_peaks)} peaks, thr. = {spike_h}mV')
    return spike_peaks, spike_interval

def spike_extract(vector, intervals):
    """ Extract individual spikes by ibdex 
    """
    spike_array = []
    [spike_array.append(vector[intervals[peak_key][0]:intervals[peak_key][1]])
     for peak_key in intervals.keys()]

    return np.array(spike_array)

def otsu_baseline(record=False, vector=False, rep=1):
    """ Otsu thresholding for baseline extraction
    """
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
    return [vector[2:-2],
            np.array([point_der(vector[i-2], vector[i-1], vector[i+1], vector[i+2], dt)
                      for i in range(2,len(vector)-2)])]

def der2(vector, dt):
    """ 2nd derivate.
    vector - input data (sweepY)
    dt - discretization frequency (in sec)
    """
    point_der2 = lambda x_2k, x_1k, x_k, x_k1, x_k2, t: (-x_2k + 16*x_1k - 30*x_k + 16*x_k1 - x_k2)/(12 * t*t)
    return [vector[2:-2],
            np.array([point_der2(vector[i-2], vector[i-1], vector[i], vector[i+1], vector[i+2], dt)
                     for i in range(2,len(vector)-2)])]

# def der3(vector, dt)

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

data_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data')
res_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'results')
if not os.path.exists(res_path):
    os.makedirs(res_path)



reg_list = ABFpars(data_path)
num = 0
reg = reg_list[num]

for reg in reg_list:
    logging.info(f'Registration {reg.fileName} in progress')

    # spike detection and extraction
    reg_spike_peak, reg_spike_interval = spike_detect(reg, spike_h=-10)
    spike_array = spike_extract(reg.sweepY_no_gap, reg_spike_interval)

    # derivate section
    der_array = [der(i, reg.dataSecPerPoint) for i in spike_array]
    der2_array = [der2(i, reg.dataSecPerPoint) for i in spike_array]
    time_line = np.arange(len(der_array[0][0]))*reg.dataSecPerPoint  # time axis for derivate data (sec)

    
    full_no_gap_sweep, full_no_gap_der = der(reg.sweepY_no_gap, reg.dataSecPerPoint)


    # plot section
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'{reg.fileName}, {reg.appTime}')

    ax0 = fig.add_subplot(311)
    ax0.set_title('Full record')
    ax0.set_xlabel('t (sec)')
    ax0.set_ylabel('V (mV)')
    ax0.plot(reg.sweepX_no_gap[2:-2], full_no_gap_sweep)

    ax00 = fig.add_subplot(312)
    ax00.set_title('Full record dV/dt')
    ax00.set_xlabel('t (sec)')
    ax00.set_ylabel('dV/dt (V/sec)')
    ax00.plot(reg.sweepX_no_gap[2:-2], full_no_gap_der/1e3)

    ax1 = fig.add_subplot(337)
    ax1.set_title('V ~ t')
    ax1.set_xlabel('t (sec)')
    ax1.set_ylabel('V (mV)')

    ax2 = fig.add_subplot(338)
    ax2.set_title('dV/dt ~ V')
    ax2.set_xlabel('V (mV)')
    ax2.set_ylabel('dV/dt (V/sec)')

    ax3 = fig.add_subplot(339)
    ax3.set_title('dV2/dt2 ~ V')
    ax3.set_xlabel('V (mV)')
    ax3.set_ylabel('dV2/dt2 (mV/sec2)')

    for plot_num in range(0, len(der_array)):
        der_plot = der_array[plot_num]
        der2_plot = der2_array[plot_num]
        ax1.plot(time_line, der_plot[0], alpha=.5)
        ax2.plot(der_plot[0], der_plot[1]/1e3, alpha=.5)
        ax3.plot(der2_plot[0], der2_plot[1], alpha=.5)

    plt.tight_layout()
    plt.savefig(f'{res_path}/{reg.fileName}_ctrl_img.png')
    plt.close('all')

    logging.info('Ctrl img saved\n')



# ot = otsu_baseline(reg)
# otsuY = np.copy(reg.sweepY_no_gap)
# otsuY[otsuY > ot] = ot

# oot = otsu_baseline(vector=otsuY)
# otsuY[otsuY > oot] = oot


# plt.figure(figsize=(8, 5))

# for plot_num in range(0, len(der_array)):
#     der_plot = der_array[plot_num]
#     der2_plot = der2_array[plot_num]
#     time_line = np.arange(len(der_plot[0]))*reg.dataSecPerPoint
#     plt.plot(der_plot[1], der2_plot[1]/der_plot[1], alpha=.5)
#     # plt.plot(der_plot[0], der2_plot[1]/max(der2_plot[1]), alpha=.5, ls='--')

# # # plt.plot(reg.sweepX_no_gap, reg.sweepY_no_gap)
# # # plt.plot(reg.sweepX_no_gap, otsuY, ls=':')
# # # plt.plot(reg.sweepX_no_gap[reg_spike], reg.sweepY_no_gap[reg_spike], 'x')
# # # plt.axhline(ot, color='k', ls='--')
# # # # plt.plot(reg.sweepX_no_gap, reg.sweepY_no_gap)

# # # plt.hist(reg_hist)

# plt.show()