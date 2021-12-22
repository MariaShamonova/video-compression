#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:18:03 2021

@author: mariashamonova
"""


def transformToBitStream(codewars, Y):
    bit_stream = ''

    r = len(Y)
    c = len(Y[0])

    for i in range(r):
        for j in range(c):
            ind = 0

            while ind < len(Y[i][j]) - 1:
                item = Y[i][j][ind]
                try:
                    bit_stream += codewars[str(abs(item))]
                
                    if (ind % 3 == 1):
                        bit_stream += ('0' if item < 0 else '1')

                        bit_stream += ('0' if Y[i][j][ind + 1] == 0 else '1')
                        ind += 1

                except Exception:
                    print("Word: " + str(abs(item)) + ' not exist')
                
                ind += 1

    return bit_stream
