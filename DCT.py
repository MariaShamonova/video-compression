#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:29:30 2021

@author: mariashamonova
"""

import numpy as np
import math
from reshapeFrame import reshapeFrame


def ZigZagTransform(block):
    zigzag = []

    for index in range(1, len(block) + 1):
        slice = [i[:index] for i in block[:index]]

        diag = [slice[i][len(slice)-i-1] for i in range(len(slice))]

        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    for index in reversed(range(1, len(block))):
        slice = [i[:index] for i in block[:index]]

        diag = [block[len(block) - index + i][len(block) - i - 1]
                for i in range(len(slice))]

        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    return zigzag


def DCT(frame, N, QP):
    print('DCT')
    rows, columns = frame.shape

    X, shape = reshapeFrame(frame, N)

    A = np.zeros(shape)
    Y = np.zeros(shape)

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            cosx = math.cos(int(((2*j+1)*i*math.pi)/2*N))
            if (i == 0):
                A[i][j] = math.sqrt(1/N) * cosx
            else:
                A[i][j] = math.sqrt(2/N) * cosx

            # Quantization
            Y[i][j] = QP * \
                np.round(A[i][j].dot(X[i][j]).dot(A[i][j].transpose()) / QP, 0)
            
            Y[i][j] = ZigZagTransform(Y[i][j])

    return Y
