#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:44:48 2022

@author: varun
"""

import nim_v2
import opponentTables
from matplotlib import pyplot as plt
import random

LEARNING_RATE = 0.45 # optimal learning rate for q agents

def train2Players(game, agent1, agent2, episodes, thresh_start_exploiting, thresh_stop_exploring):
    for ep in range(0, episodes):        
        if agent1.isQ() and ep + 1 >= thresh_start_exploiting:    
            agent1.setEpsilon(0.99 * agent1.getEpsilon())
        if agent2.isQ() and ep + 1 >= thresh_start_exploiting:
            agent2.setEpsilon(0.99 * agent2.getEpsilon())
        
        if ep + 1 == thresh_stop_exploring:
            if agent1.isQ():    
                agent1.setEpsilon(0.0)
            if agent2.isQ():
                agent2.setEpsilon(0.0)
        
        nim_v2.playGame(game, agent1, agent2)

def trainOneQ(game, q_agent, rand_agent, opt_agent, mal_agent, percent_rand, percent_opt, percent_mal, episodes, thresh_start_exploiting, thresh_stop_exploring):
    for ep in range(0, episodes):
        if ep + 1 >= thresh_start_exploiting:    
            q_agent.setEpsilon(0.99 * q_agent.getEpsilon())
        
        if ep + 1 == thresh_stop_exploring:
            q_agent.setEpsilon(0.0)
        
        x = random.random()
        if x < percent_rand:
            nim_v2.playGame(game, q_agent, rand_agent)
        elif x < percent_rand + percent_opt:
            nim_v2.playGame(game, q_agent, opt_agent)
        else:
            nim_v2.playGame(game, q_agent, mal_agent)
    

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

game = nim_v2.nim(3, 3) # a game starting with three piles and three stones in each pile

q = nim_v2.q_agent(game, LEARNING_RATE) # a q agent which can play in "game" and has learning rate 0.45
q2 = nim_v2.q_agent(game, LEARNING_RATE) # another such q agent

# an optimal agent which does not learn
opt = nim_v2.opp_agent(game) 
opt.setTable(opponentTables.optimalTable(game.getTemplate())) 

# a random agent which does not learn
rand = nim_v2.opp_agent(game)
rand.setTable(opponentTables.randTable(game.getTemplate()))

# a mal-optimal agent which does not learn
mal = nim_v2.opp_agent(game)
mal.setTable(opponentTables.malOptimalTable(game.getTemplate()))

# the q agent(s) stop(s) exploring 100% of the time after 20000 games 
# the q agent(s) start(s) exploiting 100% of the time after 60000 games
train2Players(game, q, opt, 100000, 20000, 60000) # q plays opt in game 100000 times
q.setTable(game.getTemplate()) # resets q's q table
train2Players(game, q, q2, 100000, 20000, 60000) # q plays q2 in game 100000

q.setTable(game.getTemplate()) # resets q's q table

# q plays in environment with 20% random agents, 50% optimal agents, 30% mal-optimal agents 
# the last three parameters are the same from train2Players
trainOneQ(game, q, rand, opt, mal, 0.2, 0.5, 0.3, 100000, 20000, 60000)

# graphing q's strategy rate and win rate over time
getStratPlot(q.getStratError(), 1000) 
getWinsPlot(q.getWins(), 1000)

# graphing q2's strategy rate and win rate over time
getStratPlot(q2.getStratError(), 1000)
getWinsPlot(q2.getWins(), 1000)


