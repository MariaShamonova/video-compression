# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from reshapeFrame import reshapeFrame
from encoder import Encoder
from decoder import Decoder

p_frame_thresh = 300000
N = 8

def degreeOfSimilarity(block1, block2):
    # Евклидовая норма разницы двух матриц
    diff = block1 - block2
    np.linalg.norm(diff, ord=None, axis=None, keepdims=False)

def splitIntoDisjointBlocks(block):
    result = block.reshape(3, 4, 4)
    return result


def searchForKeyframes(curr_frame, prev_frame):
    diff = cv2.absdiff(curr_frame, prev_frame)
    non_zero_count = np.count_nonzero(diff)
    if (non_zero_count > p_frame_thresh):
        print("Got P-Frame")
        prev_frame = curr_frame
        return "P"
    return "I"
    print('search key frames')


def encode(frame):
    print('encode')

    encoder = Encoder()
    frame_y = encoder.transform_rgb_to_y(frame)

    X, shape = reshapeFrame(frame_y, N)

    Y = [[[] for c in range(shape[1])] for r in range(shape[0])]
    all_dct_elements = []

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            dct_coeff = encoder.dct(X[i][j], N)
            quantinization_coeff = encoder.quantization(dct_coeff)
            sequence_coeff = encoder.zig_zag_transform(quantinization_coeff)

            series_value_coeff = encoder.separate_pair(sequence_coeff)
            Y[i][j] = series_value_coeff
            all_dct_elements.append(abs(Y[i][j][1]))
            all_dct_elements.extend([abs(Y[i][j][index])
                                     for index in range(1, len(Y[i][j]) - 1)])

    codewars = encoder.entropy_encoding(all_dct_elements)
    bit_stream = encoder.transform_to_bit_stream(codewars, Y)

    return bit_stream, codewars, frame_y


def decode(bit_stream, codewars, shape):
    decoder = Decoder()
    blocks = decoder.entropy_decoder(bit_stream, codewars, N, shape)

    r = len(blocks)
    c = len(blocks[0])

    Y = [[[] for c in range(int(shape[1] / N))]
         for r in range(int(shape[0] / N))]

    for i in range(r):
        for j in range(c):
            Y[i][j] = decoder.inverse_zig_zag_transform(blocks[i][j], 8)

            Y[i][j] = decoder.dequantization(Y[i][j])

            Y[i][j] = decoder.idct(Y[i][j], N)

    Y = decoder.concat_blocks(Y)
    return Y

if __name__ == "__main__":
    cap = cv2.VideoCapture('video.mp4')

    i = 0
    frames = []

    # while(cap.isOpened()):
    while(i < 1):
    
        ret, frame = cap.read()
        
        frame = cv2.imread('frame_1.jpeg', cv2.COLOR_RGB2BGR)

        bit_stream, codewars, frame_y = encode(frame)
        decode_frame_y = decode(bit_stream, codewars, frame_y.shape)


        if ret == False:
            break

        # frames.append(frame_y)

        i += 1

    cap.release()

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(frame_y, cmap=plt.get_cmap(name='gray'))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(decode_frame_y, cmap=plt.get_cmap(name='gray'))
    plt.show()





