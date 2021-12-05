#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 19:27:34 2021

@author: mariashamonova
"""

import numpy as np

def reshapeFrame(frame, N):  
    h, w = np.array(frame).shape
    sz = np.array(frame).itemsize
    bh, bw = N, N
    shape = (int(h/bh), int(w/bw), bh, bw)
    strides = sz*np.array([w*bh, bw, w, 1])

    X = np.lib.stride_tricks.as_strided(
        frame, shape=shape, strides=strides)
    return X, shape