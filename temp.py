# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from DCT import DCT
from reshapeFrame import reshapeFrame
from entropyEncoding import entropyEncoding
from transformToBitStream import transformToBitStream

p_frame_thresh = 300000


def degreeOfSimilarity(block1, block2):
    # Евклидовая норма разницы двух матриц
    diff = block1 - block2
    np.linalg.norm(diff, ord=None, axis=None, keepdims=False)
    print(np)


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


def encodeFrame(frame):
    print('encode')


def setChromaticSamples(frame_y, frame):
    r, c = int(np.array(frame).shape[0]/4), int(np.array(frame).shape[1]/4)
    blocks = np.zeros((r, c, 2))

    blocks[0, 1] = [1, 2]
    for i in range(0, r):
        for j in range(0, c):
            x = 2*i + 1
            y = 2*j+1
            blocks[i][j] = [0.713 * (frame[x][y][0] - frame_y[x])
                            [y], 0.564 * (frame[x][y][2] - frame_y[x][y])]
    return blocks



if __name__ == "__main__":

    cap = cv2.VideoCapture('video.mp4')

    i = 0
    frames = []

    while(i < 5):
        # while(cap.isOpened()):

        ret, frame = cap.read()

        def frame_gray(rgb): return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        frame_y = frame_gray(frame)
        frame_4x4, shape_4x4 = reshapeFrame(frame_y, 4)

        frame_chromatic = setChromaticSamples(
            frame_y, frame)

        plt.figure(figsize=(10, 10))

        plt.imshow(frame_y, cmap=plt.get_cmap(name='gray'))
        if ret == False:
            break
        # cv2.imwrite('kang'+str(i)+'.jpg', frame)
        frames.append(frame_y)
        i += 1

    cap.release()

    # all_dct_elements = []
    abs_dct_elements, Y = DCT(frames[0], 8, 100)
    
    codewars = entropyEncoding(abs_dct_elements)
  
    # print(codewars)
    transformToBitStream(codewars, Y)






