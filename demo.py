 #!/usr/bin/env python3
""" Copyright © 2021 Borys Olifirov
AP detector test functions.

"""

import sys
import os
import logging

import numpy as np
import pandas as pd
import scipy

import matplotlib
import matplotlib.pyplot as plt

import pyabf


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)


data_path = os.path.join(sys.path[0], 'data')
res_path = os.path.join(sys.path[0], 'results')
if not os.path.exists(res_path):
    os.makedirs(res_path)

demo_file = '2020_11_24_0016.abf'
print(sys.path[0])
for root, dirs, files in os.walk(data_path):
        for file in files:
            print(file)
            if file == demo_file:
                file_path = os.path.join(root, file)
                abf = pyabf.ABF(file_path)
                logging.info(f'File {file} uploaded!')
print('b')