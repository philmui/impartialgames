#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 14:30:30 2022

@author: varun
"""

def nim_sum(arr):
    ret = 0
    for i in arr:
        ret ^= i
    return ret

def optimalTable(template):
    ret = template.copy()   
    for s in ret:
        for a in ret[s]:
            i = a[0]
            x = a[1]
            k = tuple(list(s[:i]) + [s[i] - x] + list(s[i+1:]))     
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
    ret = template.copy()   
    for s in ret:
        for a in ret[s]:
            ret[s][a] = 1.0
    return ret