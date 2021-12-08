#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:28:57 2021

@author: mariashamonova
"""

def assign_code(nodes, label, result, prefix = ''):    
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
    for n in vals.keys(): # leafs initialization
        nodes[n] = []

    while len(vals) > 1: # binary tree creation
        s_vals = sorted(vals.items(), key=lambda x:x[1]) 
        a1 = s_vals[0][0]
        a2 = s_vals[1][0]
        vals[a1+a2] = vals.pop(a1) + vals.pop(a2)
        nodes[a1+a2] = [a1, a2]        
    code = {}
    root = a1+a2
    tree = {}
    tree = assign_code(nodes, root, code)   # assignment of the code for the given binary tree      
    return code, tree

def algorithmHaffman(freq):
    

    # freq = [
    # (8.167, 'a'), (1.492, 'b'), (2.782, 'c'), (4.253, 'd'),
    # (12.702, 'e'),(2.228, 'f'), (2.015, 'g'), (6.094, 'h'),
    # (6.966, 'i'), (0.153, 'j'), (0.747, 'k'), (4.025, 'l'),
    # (2.406, 'm'), (6.749, 'n'), (7.507, 'o'), (1.929, 'p'), 
    # (0.095, 'q'), (5.987, 'r'), (6.327, 's'), (9.056, 't'), 
    # (2.758, 'u'), (1.037, 'v'), (2.365, 'w'), (0.150, 'x'),
    # (1.974, 'y'), (0.074, 'z') ]    
    vals = {l:v for (v,l) in freq}
    code, tree = Huffman_code(vals)
    print(freq)
    # print(code)
    text = '344' # text to encode
    print(code['3'])
    encoded = ''.join([code[t] for t in text]) #потму что в текущем алгоритме он ищет это число побуквенно, а у меня цифра
    print('Encoded text:',encoded)
    
   
    
    decoded = []
    i = 0
    while i < len(encoded): # decoding using the binary graph
        ch = encoded[i]  
        act = tree[ch]
        while not isinstance(act, str):
            i += 1
            ch = encoded[i]  
            act = act[ch]        
        decoded.append(act)          
        i += 1
    
    print('Decoded text:',''.join(decoded))
    return code
    
  
   