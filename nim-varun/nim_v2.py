#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 14:31:41 2022

@author: varun
"""

import random

def nim_sum(arr):
    ret = 0
    for i in arr:
        ret ^= i
    return ret

class nim:
    def __init__(self, initial_stones_per_pile, piles):
        self.n = piles
        self.m = initial_stones_per_pile
        self.state = []
        self.state = [self.m] * self.n
        self.player = 1
        self.template = {}
        
    def getState(self):
        return self.state.copy()
        
    def getPossActions(self, state):
        ret = []
        for i in range(0, self.n):
            for j in range (1, state[i] + 1):
                ret.append([i, j])
        return ret
    
    def getNewActions(self, state, table):
        ret = []
        for i in range(0, self.n):
            for j in range (1, state[i] + 1):
                if table[tuple(state)][tuple([i, j])] == 0.0:
                    ret.append([i, j])
        return ret
    
    def playMove(self, action):
        self.state[action[0]] -= action[1]
    
    def gameOver(self):
        return self.state == [0] * self.n
    
    def getPlayer(self):
        return self.player
    
    def switchPlayer(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1
    
    def recursion(self, ind, s):
        if ind == 0:
            self.template[tuple(s)] = {}
            for a in self.getPossActions(s):
                self.template[tuple(s)][tuple(a)] = 0.0
            return

        for i in range(0, self.m + 1):
            t = s.copy()
            t.append(i)    
            self.recursion(ind-1, t)
        
    def getTemplate(self):
        for i in range(0, self.m + 1):
            arr = []
            arr.append(i)
            self.recursion(self.n - 1, arr)       
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
        max_val = -2.0
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
            #print("explore")
            poss_actions = self.game.getNewActions(state, self.qtable)
            if len(poss_actions) == 0:
                poss_actions = self.game.getPossActions(state) 
            k = len(poss_actions)
            ind = random.randint(0, k-1)
            #print(self.qtable[tuple(state)][tuple(poss_actions[ind])])
            return poss_actions[ind]
        else:
            #print("exploit")
            return self.bestAction(state)
    
    def updateTable(self, state, action, new_state, reward):
        old = self.qtable[tuple(state)][tuple(action)]
        new = reward
        if new_state != [0] * self.game.n:
            new += self.qtable[tuple(new_state)][tuple(self.bestAction(new_state))]
        self.qtable[tuple(state)][tuple(action)] = old + self.alpha * (new - old)
        #print("updating: state = {0}, action = {1}, new state = {2}".format(state, action, new_state))
        #print("          old = {0}, q[ns][na] = {1}, reward = {2}, new = {3}".format(old, new - reward, reward, self.qtable[tuple(state)][tuple(action)]))
    
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
    
    def addStratError(self, state, action, new_state):
        if nim_sum(new_state) == 0:
            self.strat_error.append(1.0)
        elif nim_sum(state) != 0:
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
    old_agent = -1
    old_state = -1
    old_action = -1
    while True:
        if game.getPlayer() == 1:
            agent = agent1
        else:
            agent = agent2
        
        state = game.getState()       
        action = agent.getAction(state)
        game.playMove(action)    
        new_state = game.getState()
        #print("player {0} made move {1} from state {2} to state {3}".format(game.getPlayer(), action, state, new_state))
        
        if agent.isQ():
            agent.addStratError(state, action, new_state)
        
        if game.gameOver():
            break
        
        if old_agent != -1 and old_agent.isQ():
            old_agent.updateTable(old_state, old_action, new_state, 0)
        
        game.switchPlayer()
        old_state = state
        old_action = action
        old_agent = agent
    
    if game.getPlayer() == 1:
        #print("gg: player 1 won")
        if agent1.isQ():
            agent1.updateTable(state, action, new_state, 1)
            agent1.addWin(True)
        if agent2.isQ():
            agent2.updateTable(old_state, old_action, new_state, -1)
            agent2.addWin(False)
    else:
        #print("gg: player 2 won")
        if agent2.isQ():
            agent2.updateTable(state, action, new_state, 1)
            agent2.addWin(True)
        if agent1.isQ():
            agent1.updateTable(old_state, old_action, new_state, -1)
            agent1.addWin(False)
    game.reset()