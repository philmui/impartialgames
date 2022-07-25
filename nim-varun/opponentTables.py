#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 14:30:30 2022

@author: varun
"""

def pair_nim_sum(a, b):
    x = bin(a)[2:][::-1]  # converts to binary and reverses the order of the digits
    y = bin(b)[2:][::-1]  # ^

    z = ''

    for i in range(0, min(len(x), len(y))):
        if x[i] == y[i]:
            z += '0'
        else:
            z += '1'
    if (len(x) > len(y)):
        for i in range(len(y), len(x)):
            z += x[i]
    else:
        for i in range(len(x), len(y)):
            z += y[i]

    z = z[::-1]

    return int(z, 2)
    

def nim_sum(arr):
    ret = arr[0]
    for i in range(1, len(arr)):
        ret = pair_nim_sum(arr[i], ret)
    return ret

def optimalTable(template):
    ret = template.copy()   
    for s in ret:
        for a in ret[s]:
            i = a[0]
            x = a[1]
            k = s[:i] + (s[i] - x) + s[i+1:]          
            if nim_sum(k) == 0:
                ret[s][a] = 1.0
    return ret

def malOptimalTable(template):
    opt = optimalTable(template)
    ret = {} 
    for key in opt:
        ret[key] = {}
        for key2 in opt[key]:
            if opt[key][key2] == 1.0:
                ret[key][key2] = 0.0
            else:
                ret[key][key2] = 1.0   
    return ret

def randTable(template):
    return template.copy()