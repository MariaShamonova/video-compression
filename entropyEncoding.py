#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:03:33 2021

@author: mariashamonova
"""
import numpy as np
import math
import collections
import queue
from algorithmHaffman import algorithmHaffman

def separatePair(seq):
    pairs = []

    while (seq[len(seq) - 1] == 0):
        seq.pop(len(seq) - 1)

    count_zero = 0

    for i in range(0, len(seq)):
        if (seq[i] != 0):
            pairs.append((count_zero, seq[i]))
            count_zero = 0
        else:
            count_zero += 1

    return pairs


def mergeTwoNodes(data):
    new_node = ((data[0][0], data[1][0]), data[0][1] +
                data[1][1], data[0][2] + data[1][2])
    data.pop(0)
    data.pop(0)
    data.insert(0, new_node)

    return data


def getProbabilities(block):
    pairs = [344, 34, 45, 0, 0, 1, 0, 0, 0, 45, 0, 0, 1, 32, 2, 344, 34, 45, 0, 0, 3, 7, 3, 5, 34, 45, 0, 0, 1, 32,
             2, 344, 34, 45, 0, 0, 1, 3, 6, 2, 34, 45, 3, 6, 8, 32, 2, 344, 34, 45, 0, 0, 1, 32, 2, 344, 34, 45, 0, 0, 1, 32, 2, ]

    collection = collections.Counter(pairs)
    unique_numbers = sorted(collection.items(), key=lambda item: item[1])
    total_count = len(block)
    probabilities = []
    # print(collection)

    for ind, item in unique_numbers:

        p = item/total_count
        probabilities.append((p, str(ind)))

   
    print(algorithmHaffman(probabilities))

    return pairs


def entropyEncoding(block):
    block = [234, 34, 4, 3, 6, 7, 0, 0, 0, 34, 4, 3, 0, 0, 0, 0, 234, 34, 4, 3, 6, 7, 0, 0, 0, 34,
             4, 3, 6, 0, 8, 5, 0, 0, 3, 54, 0, 0, 6, 9, 34, 4, 3, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # array of 64 elements
    # block = separatePair(block)
    getProbabilities(block)

    return block
