#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:28:57 2021

@author: mariashamonova
"""

import collections
import time

import cv2
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
                values.append(
                    int(("" if int(bit_stream[i + 1]) else "-") + value)
                )  # Значение

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
    color = (0, 255, 0)
    thickness = 2
    height, width, index = image.shape
    width_num = width // BLOCK_SIZE
    height_num = height // BLOCK_SIZE

    for i in range(height_num):
        for j in range(width_num):

            start_point = (int(i), int(j))
            end_point = (int(motion_vector[i][j] + i), int(motion_vector[i][j] + j))
            if start_point != end_point:
                image = cv2.arrowedLine(image, start_point, end_point, color, thickness)

    print(image.shape)

    return image


def reshape_frame(frame, N):
    h, w = np.array(frame).shape
    sz = np.array(frame).itemsize
    bh, bw = N, N
    shape = (int(h / bh), int(w / bw), bh, bw)
    strides = sz * np.array([w * bh, bw, w, 1])

    X = np.lib.stride_tricks.as_strided(frame, shape=shape, strides=strides)
    print(shape)
    print(X.shape)
    return X
