#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:44:48 2022

@author: varun
"""

import nim_v2
import opponentTables
from matplotlib import pyplot as plt

LEARNING_RATE = 0.45

def train2Players(game, agent1, agent2, episodes, thresh_stop_exploring):
    for ep in range(0, episodes):        
        if agent1.isQ():    
            agent1.setEpsilon(0.99 * agent1.getEpsilon())
        if agent2.isQ():
            agent2.setEpsilon(0.99 * agent2.getEpsilon())
        
        if ep + 1 == thresh_stop_exploring:
            if agent1.isQ():    
                agent1.setEpsilon(0.0)
            if agent2.isQ():
                agent2.setEpsilon(0.0)
        
        nim_v2.playGame(game, agent1, agent2)

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

game = nim_v2.nim(10, 3)

q = nim_v2.q_agent(game, LEARNING_RATE)

opt = nim_v2.opp_agent(game)
opt.setTable(opponentTables.optimalTable(game.getTemplate()))

train2Players(game, q, opt, 100000, 50000)

getStratPlot(q.getStratError(), 1000)
getWinsPlot(q.getWins(), 1000)


