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
from algorithmHaffman import HuffmanTree


def getProbabilities(block):
    collection = collections.Counter(block)
   
    unique_numbers = sorted(collection.items(), key=lambda item: item[1])
    
    total_count = len(block)
    # probabilities = []
    probabilities = dict()
    
    for ind, item in unique_numbers:
        
        p = item/total_count
        probabilities[str(ind)] = p
    
    return probabilities


def entropyEncoding(blocks):
   
    probability = getProbabilities(blocks)
    # codewars = algorithmHaffman(probability)

    tree = HuffmanTree(probability)
    codewars = tree.get_code()
   
    # print(codewars)
    # transformToBitStream(codewars, blocks)

    return codewars
