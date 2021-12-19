#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:20:25 2021

@author: mariashamonova
"""

import time
import numpy as np
from reshapeFrame import reshapeFrame
from ZigZagTransform import ZigZagTransformInverse
from Quantization import Dequantization
from DCT import IDCT


def getValues(bit_stream, codewars, N, shape):
    values = []
    bites = ''

    codewars = dict((v, k) for k, v in codewars.items())

    countValues = 0
    i = 0

    rows = int(shape[0] / 8)
    columns = int(shape[1] / 8)

    r = 0
    c = 0

    blocks = [[[] for i in range(columns)] for j in range(rows)]

    start_time = time.time()
    while i <= len(bit_stream):
        try:
            bites += bit_stream[i]
            value = codewars[bites]

            if (countValues == 0):

                values.extend([0 for z in range(int(value))])

                countValues = 1
            elif (countValues == 1):
                values.append(
                    int(('' if int(bit_stream[i + 1]) else '-') + value))  # Значение

                if (bit_stream[i + 2] == '1'):

                    values.extend([0 for z in range(int(N*N - len(values)))])

                    blocks[r][c] = values
                    r = (r + 1 if r < rows - 1 else 0)
                    c = (c + 1 if c < columns - 1 else 0)
                    values = []
                countValues = 0
                i += 2
            bites = ''

        except Exception:

            pass

        i += 1
    print("--- %s seconds ---" % (time.time() - start_time))

    return blocks


def decoder(bit_stream, codewars, shape, N):

    blocks = getValues(bit_stream, codewars, N, shape)

    r = len(blocks)
    c = len(blocks[0])

    Y = [[[] for c in range(int(shape[1] / N))]
         for r in range(int(shape[0] / N))]

    for i in range(r):
        for j in range(c):

            Y[i][j] = ZigZagTransformInverse(blocks[i][j], 8)

            Y[i][j] = Dequantization(Y[i][j])

            Y[i][j] = IDCT(Y[i][j], N)

    return Y
