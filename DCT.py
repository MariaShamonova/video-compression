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

    zigzag = np.array(zigzag).reshape(int(N*N/2), 2)
    zigzag = encodingSeriasValue(zigzag)
    return zigzag


def encodingSeriasValue(block):
    r, c = np.array(block).shape
    pairs = [[[] for c in range(c)] for r in range(r)]

    for index in reversed(range(0, len(block))):

        if (block[index][0] == float(0) and block[index][1] == float(0)):
            pairs.remove(pairs[index])
        else:

            pairs = [[block[i][0], block[i][1], 0] for i in range(0, index)]

            pairs.append([block[index][0], block[index][1], 1])

            return pairs

    return []


def DCT(frame, N, QP):
    print('DCT')
    rows, columns = frame.shape

    X, shape = reshapeFrame(frame, N)
    A = np.zeros((N, N))
    quantization_coeff = np.zeros(shape)

    Y = [[[] for c in range(shape[1])] for r in range(shape[0])]

    for i in range(0, shape[0]):

        for j in range(0, shape[1]):

            A = [[math.sqrt((1 if (r == 0) else 2)/N) * math.cos(int(((2*c+1)*r*math.pi)/2*N))
                  for c in range(N)] for r in range(N)]

            # Quantization
            # quantization_coeff = QP * \
            #     np.round(np.array(A).dot(X[i][j]).dot(
            #         np.array(A).transpose()) / QP, 2)
            
            tempY = np.array(A).dot(X[i][j]).dot(np.array(A).transpose())
            
            if (i == 2 and j == 4):
                print("A: "+str(A))
                print("X: "+str(X[i][j]))
                print("Y: "+str(tempY))
            quantization_coeff = np.round(np.array(tempY).dot(np.linalg.inv(matrix_quantization_Y)))
            # Encoding series-value
            Y[i][j] = ZigZagTransform(quantization_coeff, N)
            

    
    return Y
