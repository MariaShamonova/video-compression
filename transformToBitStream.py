#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:18:03 2021

@author: mariashamonova
"""

import numpy as np

def transformToBitStream(codewars, Y):
    bit_stream = ''


    r, c = np.array(Y).shape
  

    for i in range(r):
        for j in range(c):
               for  ind, item in enumerate(Y[i][j]):
                    try:
                        bit_stream += codewars[str(abs(item))] + ('0' if item < 0 else '1')
                    except Exception:
                        print("Word: " + str(item) + ' not exist')
            
        
                
    print(bit_stream)        
    return bit_stream