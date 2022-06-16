#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:28:57 2021

@author: mariashamonova
"""

import collections
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from constants import BLOCK_SIZE


class Node(object):
    def __init__(self, name=None, value=None):
        self.name = name
        self.value = value
        self.lchild = None
        self.rchild = None


class HuffmanTree(object):
    def __init__(self, char_Weights):
        self.Leaf = [Node(k, v) for k, v in char_Weights.items()]
        while len(self.Leaf) != 1:
            self.Leaf.sort(key=lambda node: node.value, reverse=True)
            n = Node(value=(self.Leaf[-1].value + self.Leaf[-2].value))
            n.lchild = self.Leaf.pop(-1)
            n.rchild = self.Leaf.pop(-1)
            self.Leaf.append(n)
        self.root = self.Leaf[0]
        self.Buffer = list(range(1500))
        self.Dict = dict()

    def Hu_generate(self, tree, length):
        node = tree
        code = ""
        if not node:
            return
        elif node.name:
            for i in range(length):
                code += str(self.Buffer[i])
            self.Dict[node.name] = code
            return

        self.Buffer[length] = 0
        self.Hu_generate(node.lchild, length + 1)
        self.Buffer[length] = 1
        self.Hu_generate(node.rchild, length + 1)

    def get_code(self):
        self.Hu_generate(self.root, 0)
        return self.Dict


def assign_code(nodes, label, result, prefix=""):
    childs = nodes[label]
    tree = {}

    if len(childs) == 2:
        tree["0"] = assign_code(nodes, childs[0], result, prefix + "0")
        tree["1"] = assign_code(nodes, childs[1], result, prefix + "1")
        return tree
    else:
        result[label] = prefix
        return label


def Huffman_code(_vals):
    vals = _vals.copy()

    nodes = {}

    for n in vals.keys():
        nodes[n] = []

    while len(vals) > 1:
        s_vals = sorted(vals.items(), key=lambda x: x[1])
        a1 = s_vals[0][0]
        a2 = s_vals[1][0]
        vals[a1 + a2] = vals.pop(a1) + vals.pop(a2)
        nodes[a1 + a2] = [a1, a2]

    code = {}
    root = a1 + a2

    tree = assign_code(nodes, root, code)

    return code, tree


def algorithm_Haffman(freq):
    vals = {l: v for (v, l) in freq}

    code, tree = Huffman_code(vals)
    return code


def get_probabilities(block):
    collection = collections.Counter(block)

    unique_numbers = sorted(collection.items(), key=lambda item: item[1])

    total_count = len(block)
    probabilities = dict()

    for ind, item in unique_numbers:
        p = item / total_count
        probabilities[str(ind)] = p

    return probabilities


def round_num(num):
    while num % 8 != 0:
        num += 1
    return num


def append_zeros(frame):
    width, height = frame.shape
    mask = np.zeros((round_num(width), round_num(height)))
    mask = mask.astype("uint8")
    mask[:width, :height] = np.array(frame)
    return mask


def concat_blocks(blocks):
    blocks = np.array(blocks)
    blocks = np.concatenate(blocks, axis=1)
    blocks = np.concatenate(blocks, axis=1)
    return blocks


def get_values(bit_stream, codewars, N, shape):
    values = []
    bites = ""

    codewars = dict((v, k) for k, v in codewars.items())

    count_values = 0
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

            if count_values == 0:

                values.extend([0 for z in range(int(value))])

                count_values = 1
            elif count_values == 1:
                values.append(int(("" if int(bit_stream[i + 1]) else "-") + value))

                if bit_stream[i + 2] == "1":

                    values.extend([0 for z in range(int(N * N - len(values)))])

                    blocks[r][c] = values
                    r = r + 1 if r < rows - 1 else 0
                    c = c + 1 if c < columns - 1 else 0
                    values = []
                count_values = 0
                i += 2
            bites = ""

        except Exception:
            pass

        i += 1
    print("--- %s seconds ---" % (time.time() - start_time))


def draw_motion_vectors(image, motion_vector):
    backtorgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    color = (0, 255, 0)
    thickness = 1
    rows, columns, coordinates = motion_vector.shape

    for i in range(rows):
        for j in range(columns):
            x_00 = i * BLOCK_SIZE
            y_00 = j * BLOCK_SIZE
            center = BLOCK_SIZE // 2
            start_point = (x_00 + center, y_00 + center)
            end_point = (
                x_00 + motion_vector[i][j][0] * BLOCK_SIZE + center,
                y_00 + motion_vector[i][j][1] * BLOCK_SIZE + center,
            )
            if start_point != end_point:

                image = cv2.line(backtorgb, start_point, end_point, color, thickness)

    plt.imshow(image)
    plt.show()


def reshape_frame(frame, N):
    h, w = np.array(frame).shape
    sz = np.array(frame).itemsize
    bh, bw = N, N
    shape = (int(h / bh), int(w / bw), bh, bw)
    strides = sz * np.array([w * bh, bw, w, 1])

    X = np.lib.stride_tricks.as_strided(frame, shape=shape, strides=strides)

    return X


def compression_ratio(img_in, img_sam):
    return img_in / img_sam


def compute_psnr(img_in, img_sam):
    assert img_in.shape == img_sam.shape

    mse = np.mean((img_in / 255.0 - img_sam / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 20 * np.log10(1 / np.sqrt(mse))


def compute_mse(frame, reconstructed_frame):
    assert frame.shape == reconstructed_frame.shape

    err = np.sum((frame.astype("float") - reconstructed_frame.astype("float")) ** 2)
    err /= float(frame.shape[0] * frame.shape[1])

    return err


def show_image(image1, image2):

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Original")
    ax.imshow(image1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Decoded image")
    ax2.imshow(image2)
    plt.show()


def append_num(num):
    while num % 8 != 0:
        num += 1
    return num
