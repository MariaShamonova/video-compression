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
from decoder import decoder

p_frame_thresh = 300000


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


def concatBlocks(blocks):
    blocks = np.array(blocks)
    frame = []
    
    r = blocks.shape[0]

    for i in range(r):
        frame.append(np.concatenate((blocks[i]),axis=1))
   
    for i in range(0, r):
        if (r > 1):
            frame.append(np.concatenate((frame[i]),axis=2))
            
    return frame

def frame_gray(rgb): return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

if __name__ == "__main__":

    cap = cv2.VideoCapture('video.mp4')

    i = 0
    frames = []

    # while(cap.isOpened()):
    while(i < 1):
    
        ret, frame = cap.read()
        
        frame = cv2.imread('frame_1.jpeg', cv2.COLOR_RGB2BGR)
        
        frame_y = frame_gray(frame)


       
        frame_chromatic = setChromaticSamples(
            frame_y, frame)

        plt.figure(figsize=(10, 10))
  
        plt.imshow(frame_y, cmap=plt.get_cmap(name='gray'))
        
        if ret == False:
            break

        frames.append(frame_y)

        i += 1

    cap.release()

    abs_dct_elements, Y = DCT(frames[0], 8, 100)
    
    codewars = entropyEncoding(abs_dct_elements)
  

    bit_stream = transformToBitStream(codewars, Y)
    
    print('decoder')
    frame = decoder(bit_stream, codewars, frames[0].shape, 8)
    
    frame = concatBlocks(frame)
   
    plt.imshow(frame[0], cmap=plt.get_cmap(name='gray'))






