#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 14:31:41 2022

@author: varun
"""

import random

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

class nim:
    def __init__(self, initial_stones_per_pile, piles):
        self.n = piles
        self.m = initial_stones_per_pile
        self.state = [self.m] * self.n
        self.player = 1
        self.template = {}
        
    def getState(self):
        return self.state
        
    def getPossActions(self, state):
        ret = []
        for i in range(0, self.n):
            for j in range (1, state[i] + 1):
                ret.append([i, j])
        return ret
    
    def playMove(self, action):
        self.state[action[0]] -= action[1]
    
    def gameOver(self):
        for i in range(0, self.n):
            if self.state[i] != 0:
                return False
        return True
    
    def getPlayer(self):
        return self.player
    
    def switchPlayer(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1
    
    def recursion(self, ind, s):
        if ind == 0:
            for a in self.getPossActions(s):
                self.template[tuple(s)][tuple(a)] = 0.0
            return

        for i in range(0, self.m + 1):
            s.append[i]    
            self.recursion(ind-1, s)
        
    def getTemplate(self):
        self.recursion(self.n, [])       
        return self.template

    def reset(self):
        self.state = [self.m] * self.n
        self.player = 1
        

class q_agent:
    def __init__(self, game, learning_rate):
        self.game = game
        self.qtable = self.game.getTemplate().copy()
        self.alpha = learning_rate
        self.epsilon = 1.0
        self.wins = []
        self.strat_error = []
    
    def setTable(self, qtable):
        self.qtable = qtable
    
    def getTable(self):
        return self.qtable

    def bestAction(self, state):
        best_actions = []
        max_val = 0
        
        for action in self.game.getPossActions(state):
            if self.qtable[tuple(state)][tuple(action)] > max_val:
                best_actions.clear()
                best_actions.append(action)
                max_val = self.qtable[tuple(state)][tuple(action)]
            
            elif self.qtable[tuple(state)][tuple(action)] == max_val:
                best_actions.append(action)
        
        k = len(best_actions)
        ind = random.randint(0, k-1)      
        return best_actions[ind]
    
    def getAction(self, state):  
        x = random.random()
        
        if x < self.epsilon:
            poss_actions = self.game.getPossActions(state) 
            k = len(poss_actions)
            ind = random.randint(0, k-1)
            return poss_actions[ind]
        else:
            return self.bestAction(state)
    
    def updateTable(self, state, action, new_state, reward):
        old = self.qtable[tuple(state)][tuple(action)]
        new = reward + self.qtable[tuple(new_state)][tuple(self.bestAction(new_state))]
        self.qtable[tuple(state)][tuple(action)] = old + self.alpha * (new - old)
    
    def setEpsilon(self, new_exploration_rate):
        self.epsilon = new_exploration_rate
    
    def getEpsilon(self):
        return self.epsilon
    
    def isQ(self):
        return True
    
    def addWin(self, win):
        if win:
            self.wins.append(1.0)
        else:
            self.wins.append(0.0)
    
    def getWins(self):
        return self.wins
    
    def addStratError(self, state, action):
        i = action[0]
        x = action[1]
        if nim_sum(state) == 0:
            self.strat_error.append(1.0)
        elif nim_sum(state[:i] + (state[i] - x) + state[i+1:]) == 0:
            self.strat_error.append(1.0)
        else:
            self.strat_error.append(0.0)
    
    def getStratError(self):
        return self.strat_error

class opp_agent:
    def __init__(self, game):
        self.game = game
        self.qtable = self.game.getTemplate().copy()
    
    def setTable(self, qtable):
        self.qtable = qtable
    
    def getTable(self):
        return self.qtable
    
    def getAction(self, state):
        best_actions = []
        max_val = 0
        for action in self.game.getPossActions(state):
            if self.qtable[tuple(state)][tuple(action)] > max_val:
                best_actions.clear()
                best_actions.append(action)
                max_val = self.qtable[tuple(state)][tuple(action)]
            elif self.qtable[tuple(state)][tuple(action)] == max_val:
                best_actions.append(action)
        
        k = len(best_actions)
        ind = random.randint(0, k-1)      
        return best_actions[ind]
    
    def isQ(self):
        return False

def playGame(game, agent1, agent2):
    swap_starter = random.random()
    if swap_starter < 0.5:
        game.switchPlayer()
    
    agent = agent1
    old_state = -1
    old_action = -1
    while True:
        if game.getPlayer == 1:
            agent = agent1
        else:
            agent = agent2
        
        state = game.getState()
        action = agent.getAction(state)
        game.playMove(action)    
        new_state = game.getState()
        
        if agent.isQ():
            agent.addStratError(state, action)
        
        if game.gameOver():
            break
        
        if agent.isQ():
            agent.updateTable(state, action, new_state, 0)
        
        game.switchPlayer()
        old_state = state
        old_action = action
    
    if game.getPlayer == 1:
        if agent1.isQ():
            agent1.updateTable(state, action, new_state, 1)
            agent1.addWin(True)
        if agent2.isQ():
            agent2.updateTable(old_state, old_action, state, -1)
            agent2.addWin(False)
    else:
        if agent2.isQ():
            agent2.updateTable(state, action, new_state, 1)
            agent2.addWin(True)
        if agent1.isQ():
            agent1.updateTable(old_state, old_action, state, -1)
            agent1.addWin(False)