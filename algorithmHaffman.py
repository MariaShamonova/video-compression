#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:28:57 2021

@author: mariashamonova
"""


def assign_code(nodes, label, result, prefix=''):
    childs = nodes[label]
    tree = {}

    if len(childs) == 2:
        tree['0'] = assign_code(nodes, childs[0], result, prefix+'0')
        tree['1'] = assign_code(nodes, childs[1], result, prefix+'1')
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
        vals[a1+a2] = vals.pop(a1) + vals.pop(a2)
        nodes[a1+a2] = [a1, a2]

    code = {}
    root = a1+a2
    tree = {}

    tree = assign_code(nodes, root, code)

    return code, tree


def algorithmHaffman(freq):

    vals = {l: v for (v, l) in freq}

    code, tree = Huffman_code(vals)

    # text = '4363'  # text to encode
    
    # encoded = ''.join([code[text]])
    # print('Encoded text:', encoded)

    # decoded = []
    # i = 0
    # while i < len(encoded):  # decoding using the binary graph
    #     ch = encoded[i]
    #     act = tree[ch]
    #     while not isinstance(act, str):
    #         i += 1
    #         ch = encoded[i]
    #         act = act[ch]
    #     decoded.append(act)
    #     i += 1

    # print('Decoded text:', ''.join(decoded))
    return code


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
        code = ''
        if (not node):
            return
        elif node.name:
            for i in range(length):
                code += str(self.Buffer[i])
            self.Dict[node.name] = code
            # print('\n')
            return

        self.Buffer[length] = 0
        self.Hu_generate(node.lchild, length + 1)
        self.Buffer[length] = 1
        self.Hu_generate(node.rchild, length + 1)

    def get_code(self):
        self.Hu_generate(self.root, 0)
        return self.Dict
