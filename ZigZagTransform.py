#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:28:18 2021

@author: mariashamonova
"""
import numpy as np


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


def ZigZagTransformInverse(zigzag, N):
    blocks_8x8 = np.zeros((N, N))

    for index in range(1, 9):
        slice = [i[:index] for i in blocks_8x8[:index]]
        i = (0 if index % 2 == 0 else index - 1)
        j = (0 if index % 2 == 1 else index - 1)
        for ind in range(len(slice)):
            blocks_8x8[i][j] = zigzag[0]
            zigzag.pop(0)
            i = i + (1 if index % 2 == 0 else -1)
            j = j + (1 if index % 2 == 1 else -1)

    for index in reversed(range(1, 8)):

        slice = [i[:index] for i in blocks_8x8[:index]]
        i = (N - len(slice) if index % 2 == 0 else N - 1)
        j = (N - len(slice) if index % 2 == 1 else N - 1)

        for ind in range(len(slice)):
            blocks_8x8[i][j] = int(zigzag[0])
            zigzag.pop(0)
            i = i + (1 if index % 2 == 0 else -1)
            j = j + (1 if index % 2 == 1 else -1)

    return blocks_8x8.astype(int)
