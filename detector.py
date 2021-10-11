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
from scipy import integrate

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
                    # add file name
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

    logging.info(f'{len(spike_peaks)} peaks in {record.sweepCount} sweeps, thr. = {spike_h}mV')
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
    return np.array([point_der(vector[i-2], vector[i-1], vector[i+1], vector[i+2], dt)
                      for i in range(3,len(vector)-3)])

def der2(vector, dt):
    """ 2nd derivate.
    vector - input data (sweepY)
    dt - discretization frequency (in sec)
    """
    point_der2 = lambda x_2k, x_1k, x_k, x_k1, x_k2, t: (-x_2k + 16*x_1k - 30*x_k + 16*x_k1 - x_k2)/(12 * t*t)
    return np.array([point_der2(vector[i-2], vector[i-1], vector[i], vector[i+1], vector[i+2], dt)
                  for i in range(3,len(vector)-3)])

def der3(vector, dt):
    """ 3d derivate.
    vector - input data (sweepY)
    dt - discretization frequency (in sec)

    """
    point_der3 = lambda x_3k, x_2k, x_1k, x_k1, x_k2, x_k3, t: (x_3k - 8*x_2k + 13*x_1k - 13*x_k1 + 8*x_k2 - x_k3)/(12 * t*t*t)
    return np.array([point_der3(vector[i-3], vector[i-2], vector[i-1], vector[i+1], vector[i+2], vector[i+3], dt)
                     for i in range(3,len(vector)-3)])

def g_t(der_array, der2_array, noise_win=20, noise_tolerance=1):
    """ Calculate der2/der
    noise_win - number of element for noise sd calculation
    noise_tolerance - number of noise SD for positive value only extraction, another element equallies to zero
    
    """
    der_ratio_list = []
    der_ratio_max = []
    for i in range(0, len(der_array)):
        der_noise = np.std(der_array[i][:noise_win])
        der_positive_mask = np.where(der_array[i] < der_noise*noise_tolerance)

        der_ratio = der2_array[i] / der_array[i]
        der_ratio[der_positive_mask] = 0

        der_ratio_list.append(der_ratio)
        der_ratio_max.append(np.where(der_ratio == np.max(der_ratio)))

    return np.array(der_ratio_list), np.array(der_ratio_max)

def h_t(der, der2, der3, noise_win=20, noise_tolerance=1):
    der_noise = np.std(der_array[:noise_win])
    logging.info(f'dV/dt noise SD={round(der_noise, 2)}')
    der_positive_mask = np.where(der_array < der_noise*noise_tolerance)

    h_list = []
    for i in range(0, len(der_array)):
        h_list.append((der3[i]*der[i] - der2[i]**2)/(der[i]**3))
    h_list = np.array(h_list)
    h_list[der_positive_mask] = 0 
    return h_list



FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

np.seterr(divide='ignore', invalid='ignore')

data_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'data')
res_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'results')
if not os.path.exists(res_path):
    os.makedirs(res_path)

reg_list = ABFpars(data_path)
# num = 0
# reg = reg_list[num]
demo = False
save_csv = True

# df init
df = pd.DataFrame(columns=['file',      # file name
                           'app_time',  # application time
                           'v_max',     # AP max amplitude
                           't_max',     # AP maw time
                           'v_th',      # threshold voltage
                           't_th',      # threshold time
                           'power'])    # AP power


# loop over input registrations
for reg in reg_list:
    logging.info(f'Registration {reg.fileName} in progress')

    # loop over sweeps
    # for i in range(0, reg.sweepCount):

    # spike detection and extraction
    reg_spike_peak, reg_spike_interval = spike_detect(reg, spike_h=-10)
    spike_array = spike_extract(reg.sweepY_no_gap, reg_spike_interval)

    # croped array 
    voltage_array = np.array([i[3:-3] for i in spike_array])  # voltage axis, resize to der size
    time_line = np.arange(np.shape(voltage_array)[1])*reg.dataSecPerPoint  # time axis for derivate data (sec)

    # derivate section
    logging.info('1st derivete calc in progress')
    der_array = [der(i, reg.dataSecPerPoint) for i in spike_array]
    logging.info('2nd derivete calc in progress')
    der2_array = [der2(i, reg.dataSecPerPoint) for i in spike_array]
    logging.info('3d derivete calc in progress')
    der3_array = [der3(i, reg.dataSecPerPoint) for i in spike_array]

    # threshold calc
    g_t_array, g_t_max = g_t(der_array, der2_array,
                             noise_win=100, noise_tolerance=15)  # 15 noise SD, realy?!

    # loop over APs and df writing
    v_th_list = []  # list of absolute AP threshold values
    ap_pow_list = []  # list of AP power values
    for i in range(0, len(voltage_array)):
        
        # extract AP max amplitude
        v_max = max(voltage_array[i])
        v_max_i = voltage_array[i] == v_max
        t_max = time_line[v_max_i][0]

        # extract Vth
        th_i = g_t_max[i][0][0]
        logging.info(f'Threshold index {th_i}')
        v_th = round(float(voltage_array[i][th_i]), 3)
        t_th = float(time_line[th_i])
        v_th_list.append(v_th)

        # AP power calc
        ap_pow = int(integrate.cumtrapz(der_array[i], voltage_array[i], initial=0)[-1])
        ap_pow_list.append(ap_pow)

        df = df.append(pd.Series([reg.fileName,  # file name
                                  reg.appTime,   # application time
                                  v_max,         # AP max amplitude
                                  t_max,         # AP maw time
                                  v_th,          # threshold voltage
                                  t_th,          # threshold time
                                  ap_pow],       # AP power
                                 index=df.columns),
                       ignore_index=True)

    print(v_th_list)
    print(ap_pow_list)

    if demo:
        # DER plot section
        i = 0
        test_int = integrate.cumtrapz(der_array[i], voltage_array[i], initial=0)
        print(test_int[-1])
        
        # for i in range(0, len(voltage_array)):

        plt.figure(figsize=(8, 5)) 
        # vector ~ time
        # plt.plot(time_line, voltage_array[i], alpha=.5)
        # plt.plot(time_line, (der_array[i]/np.max(der_array[i])*20), alpha=.5, ls='--')
        # plt.plot(time_line, der2_array[i]/np.max(der2_array[i]), alpha=.25, ls='--')
        # plt.plot(time_line, h_t_array[i]/np.max(h_t_array[i]), alpha=.5, ls=':')
        # plt.plot(time_line, (g_t_array[i]/np.max(g_t_array[i])*5), alpha=.5, ls=':')
        # plt.axvline(x=time_line[g_t_max[i]])

        # # der ~ voltage
        plt.plot(voltage_array[i], der_array[i]/max(der_array[i]), alpha=.5, ls='-')
        plt.plot(voltage_array[i], test_int/max(test_int), alpha=.5, ls=':')
        # plt.plot(voltage_array[i]/np.max(voltage_array[i]), der2_array[i]/np.max(der2_array[i]), alpha=.5, ls='--')
        # plt.plot(voltage_array[i]/np.max(voltage_array[i]), der3_array[i]/np.max(der3_array[i]), alpha=.5, ls=':')
        plt.show()
    else: 
        # CTRL plot section

        # plot section
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f'{reg.fileName}, {reg.appTime}')

        ax0 = fig.add_subplot(211)
        ax0.set_title('Full record')
        ax0.set_xlabel('t (sec)')
        ax0.set_ylabel('V (mV)')
        ax0.plot(reg.sweepX_no_gap, reg.sweepY_no_gap)

        ax1 = fig.add_subplot(234)
        ax1.set_title('V ~ t')
        ax1.set_xlabel('t (sec)')
        ax1.set_ylabel('V (mV)')

        ax2 = fig.add_subplot(235)
        ax2.set_title('dV/dt ~ V')
        ax2.set_xlabel('V (mV)')
        ax2.set_ylabel('dV/dt (V/sec)')

        ax3 = fig.add_subplot(236)
        ax3.set_title('dV2/dt2 ~ V')
        ax3.set_xlabel('V (mV)')
        ax3.set_ylabel('dV2/dt2 (mV/sec2)')

        for i in range(0, len(der_array)):
            ax1.plot(time_line, voltage_array[i], alpha=.5)
            ax2.plot(voltage_array[i], der_array[i]/1e3, alpha=.5)
            ax3.plot(voltage_array[i], der2_array[i], alpha=.5)

        plt.tight_layout()
        plt.savefig(f'{res_path}/{reg.fileName}_ctrl_img.png')
        plt.close('all')

        logging.info('Ctrl img saved\n')

if save_csv:
  df.to_csv(f'{res_path}/results.csv', index=False)
  logging.info('CSV file saved')