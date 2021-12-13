#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 13:29:30 2021

@author: mariashamonova
"""

import numpy as np
import math
from reshapeFrame import reshapeFrame
from entropyEncoding import entropyEncoding
from array import array
import numpy as geek
from matplotlib import pyplot as plt

matrix_quantization_Y = [[16, 11, 10, 16, 2, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [4, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 35, 55, 64, 81, 104, 113, 92],
                         [49, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 108, 99]]


def ZigZagTransform(block, N):
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


def separatePair(seq):
    transformSeq = []
    while (seq[len(seq) - 1] == 0):
        seq.pop(len(seq) - 1)

    count_zero = 0

    for i in range(0, len(seq)):
        if (seq[i] != 0):

            transformSeq.extend([count_zero, seq[i]])
            count_zero = 0
        else:
            count_zero += 1

    return transformSeq


def DCT(frame, N, QP):
    print('DCT')
    rows, columns = frame.shape

    X, shape = reshapeFrame(frame, N)
    A = np.zeros((N, N))
    quantization_coeff = np.zeros(shape)

    Y = [[[] for c in range(shape[1])] for r in range(shape[0])]
    all_dct_elements = []
    dc_coeff = 0
    cc_coeff = []

    for i in range(0, shape[0]):

        for j in range(0, shape[1]):

            A = [[math.sqrt((1 if (r == 0) else 2)/N) * math.cos(int(((2*c+1)*r*math.pi)/2*N))
                  for c in range(N)] for r in range(N)]

            tempY = np.array(A).dot(X[i][j]).dot(np.array(A).transpose())

            quantization_coeff = np.round(np.array(tempY).dot(
                np.linalg.inv(matrix_quantization_Y))).astype(int)

            Y[i][j] = ZigZagTransform(
                quantization_coeff, N)  # sequence of values
            
            
            Y[i][j] = separatePair(Y[i][j])
            
           
            all_dct_elements.append(abs(Y[i][j][0]) if (
                j == 0) else abs(Y[i][j][1] - Y[i][j - 1][1]))
            all_dct_elements.extend([abs(Y[i][j][index])
                                    for index in range(1, len(Y[i][j]))])
           
    
    return all_dct_elements, Y
