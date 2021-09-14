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
                    setattr(abf_record, 'sweepY_no_gap', no_gap_Y)

                    # create X timeline for no gap record
                    setattr(abf_record, 'sweepX_no_gap',
                            np.arange(len(abf_record.sweepY_no_gap))*abf_record.dataSecPerPoint)

                    record_list.append(abf_record)
                    logging.info(f'File {file} uploaded!')
    return (record_list)


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


abf_list = ABFpars(data_path)
num = 0
abf_list[num].setSweep(0)


# small_sweep = abf_list[abf_num].sweepY[107500:108500]

# sweepDer, sweepRes = der(small_sweep, 1/abf_list[abf_num].dataRate)
# sweepDer2, sweepRes2 = der2(small_sweep, 1/abf_list[abf_num].dataRate)

# norm = lambda x: [i / max(x) for i in x]

plt.figure(figsize=(8, 5))
# plt.xlim(left=107500, right=108500)
plt.plot(abf_list[num].sweepX_no_gap, abf_list[num].sweepY_no_gap)
plt.plot(abf_list[num].sweepX, abf_list[num].sweepY)
# for i in range(0, abf_list[abf_num].sweepCount):
#     abf_list[abf_num].setSweep(i)
#     plt.plot(abf_list[abf_num].sweepX, abf_list[abf_num].sweepY, alpha=.5, label="sweep %d" % (i))
plt.show()