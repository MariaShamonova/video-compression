#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:29:30 2021

@author: mariashamonova
"""

import math
import numpy as np
import numpy as geek

from array import array
from matplotlib import pyplot as plt

from reshapeFrame import reshapeFrame
from entropyEncoding import entropyEncoding
from ZigZagTransform import ZigZagTransform
from Quantization import Quantization


def separatePair(seq):
    transformSeq = []
    while (seq[len(seq) - 1] == 0):
        seq.pop(len(seq) - 1)

    count_zero = 0

    for i in range(0, len(seq)):

        if (seq[i] != 0):
            isLatestElement = 'ЕОВ' if i == (len(seq) - 1) else 0
            transformSeq.extend([count_zero, seq[i], isLatestElement])

            count_zero = 0
        else:
            count_zero += 1

    return transformSeq


def getMatrixA(N):

    A = [[np.round(math.sqrt((1 if (i == 0) else 2)/N) * math.cos(((2*j+1)*i*math.pi)/(2*N)), 3)
          for j in range(0, N)] for i in range(0, N)]
    return A


def IDCT(Y, N):
    A = getMatrixA(N)

    return np.array(A).transpose().dot(Y).dot(np.array(A))


def FDCT(X, N):
    A = getMatrixA(N)

    return np.array(A).dot(X).dot(np.array(A).transpose())


def DCT(frame, N, QP):
    print('DCT')

    rows, columns = frame.shape

    X, shape = reshapeFrame(frame, N)
   
    Y = [[[] for c in range(shape[1])] for r in range(shape[0])]
    all_dct_elements = []

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):

            Y[i][j] = ZigZagTransform(
                Quantization(FDCT(X[i][j], N)), N)

            Y[i][j] = separatePair(Y[i][j])

            # if i == 0 and j == 0:
            #     print('Y sep: ',Y[i][j])
            # if (i == 0 and j == 0):
            #     dc_coeff = Y[i][j][1]
            # elif (i != 0  and j == 0):
            #     dc_coeff = Y[i][j][1] - Y[i - 1][j][1]
            # else:
            #     dc_coeff = Y[i][j][1] - Y[i][j - 1][1]

            all_dct_elements.append(abs(Y[i][j][1]))
            all_dct_elements.extend([abs(Y[i][j][index])
                                    for index in range(1, len(Y[i][j]) - 1)])

    return all_dct_elements, Y
