#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:46:20 2022

@author: varun
"""

import qLearningNim
from matplotlib import pyplot as plt

def getStratPlot(strat_error, moves_per_pt):
    k = len(strat_error)
    num = moves_per_pt
    x = int((k + num - 1)/num)
    avgs = []

    for i in range(0, x):
        avg = 0.0
        for j in range(0, num):
            if (i * num + j < k):
                avg += strat_error[i * num + j]
        avg = avg/float(num)
        avgs.append(avg)
        
    plt.plot(avgs)
    plt.show()

def getWinsPlot(wins, games_per_pt):
    k = len(wins)
    num = games_per_pt
    x = int((k + num - 1)/num)
    avgs = []

    for i in range(0, x):
        avg = 0.0
        for j in range(0, num):
            if (i * num + j < k):
                avg += wins[i * num + j]
        avg = avg/float(num)
        avgs.append(avg)
        
    plt.plot(avgs)
    plt.show()

DISCOUNT_RATE = 1
LEARNING_RATE = 0.45

q1_strat_error = []
q2_strat_error = []
wins = []

game = qLearningNim.Nim(10, 5)

q1 = qLearningNim.QAgent(game, DISCOUNT_RATE, LEARNING_RATE, 1)

q2 = qLearningNim.QAgent(game, DISCOUNT_RATE, LEARNING_RATE, 1)

opt = qLearningNim.Opponent(game, 'opt')

#qLearningNim.playQvQ(100000, game, q1, q2, wins, q1_strat_error, q2_strat_error)

#getStratPlot(q1_strat_error, 1000)
#getStratPlot(q2_strat_error, 1000)
#getWinsPlot(wins, 100)

qLearningNim.playQvOpp(100000, game, q1, opt, wins, q1_strat_error)

getStratPlot(q1_strat_error, 1000)
getWinsPlot(wins, 100)




