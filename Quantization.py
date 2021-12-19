#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 19:18:34 2021

@author: mariashamonova
"""
import numpy as np

matrix_quantization_Y = [[16, 11, 10, 16, 2, 40, 51, 61],
                         [12, 12, 14, 19, 26, 58, 60, 55],
                         [4, 13, 16, 24, 40, 57, 69, 56],
                         [14, 17, 22, 29, 51, 87, 80, 62],
                         [18, 22, 37, 56, 68, 109, 103, 77],
                         [24, 35, 55, 64, 81, 104, 113, 92],
                         [49, 64, 78, 87, 103, 121, 120, 101],
                         [72, 92, 95, 98, 112, 100, 108, 99]]


def Quantization(Y):
    quantization_coeff = np.round(
        (np.divide(np.array(Y), matrix_quantization_Y))).astype(int)
    return quantization_coeff


def Dequantization(Y):
    r_coeff = np.multiply(np.array(Y), matrix_quantization_Y)
    return r_coeff


# if __name__ == "__main__":

#     temp = [
#         [1, 2, 3, 4, 5, 6, 7, 8],
#         [9, 10, 11, 12, 13, 14, 15, 16],
#         [17, 18, 19, 20, 21, 22, 23, 24],
#         [25, 26, 27, 28, 29, 30, 31, 32],
#         [33, 34, 35, 36, 37, 38, 39, 40],
#         [41, 42, 43, 44, 45, 46, 47, 48],
#         [49, 50, 51, 52, 53, 54, 55, 56],
#         [57, 58, 59, 60, 61, 62, 63, 64]]

