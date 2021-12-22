#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:48:45 2021

@author: mariashamonova
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from matplotlib import pyplot as plt

from tensorflow import keras 
from tensorflow.keras.layers import Dense