#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:26:30 2022

@author: varun
"""

import random

def big(arr):
    if len(arr) == 0:
        return -1
    return max(arr)

class Nim:
    def __init__(self, n, max_remove): # n = number of stones in each pile at the start of the game
        self.n = n
        self.state = [n, n, n]
        self.max_remove = max_remove  # max number of stones that can be removed from a stone
        self.game_over = False

        self.values = [] # q_table

        """
        initializing q_table, 5 dimensions: [a, b, c, a_act, b_act]
        a represents the number of stones in the first pile
        b represents the number of stones in the second pile
        c represents the number of stones in the third pile
        a_act represents the pile from which the stone is removed
        b_act represents the number of stones removed from a_act pile
        """
        for i in range(0, n + 1):
            arrj = []
            for j in range(0, n + 1):
                arrk = []
                for k in range(0, n + 1):
                    a = min(self.max_remove, i)
                    b = min(self.max_remove, j)
                    c = min(self.max_remove, k)
                    arr = []
                    arr1 = []
                    arr2 = []
                    arr3 = []
                    for l in range(0, a):
                        arr1.append(0.0) 
                    for l in range(0, b):
                        arr2.append(0.0) 
                    for l in range(0, c):
                        arr3.append(0.0) 
                    arr.append(arr1)
                    arr.append(arr2)
                    arr.append(arr3)
                    arrk.append(arr)
                arrj.append(arrk)
            self.values.append(arrj)

    def playMove(self, action): # plays the specified action (removes some stones from a pile)
        if self.state == [0, 0, 0]:
            self.game_over = True
            return

        # updating the state
        self.state[action[0]] -= action[1]
        #print("action = {0}: {1}".format(action, self.state))

    def currState(self): # returns the current state
        return self.state.copy()
    
    def reset(self): # resets game to the starting state
        self.state = [self.n, self.n, self.n]

    def getPossibleMoves(self, state, poss_actions): # returns the possible moves from a given state
        if state == [0, 0, 0]:
            return [[-1, -1]]

        poss_actions = []

        a = state[0]
        b = state[1]
        c = state[2]

        # getting possible actions, basically the number of stones each pile has because you can't take more than that
        a_acts = len(self.values[a][b][c][0])
        b_acts = len(self.values[a][b][c][1])
        c_acts = len(self.values[a][b][c][2])


        # putting every possible action in the list
        for i in range(0, a_acts):
            poss_actions.append([0, i + 1])
        for i in range(0, b_acts):
            poss_actions.append([1, i + 1])
        for i in range(0, c_acts):
            poss_actions.append([2, i + 1])

        return poss_actions

    def nim_sum(self, a, b): # returns a xor b 
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

    def getOptMoves(self, state, poss_actions): # returns the optimal moves from a given state
        # Mathematical way of finding the optimal moves, not q_agent learning
        a = state[0] % (self.max_remove + 1)
        b = state[1] % (self.max_remove + 1)
        c = state[2] % (self.max_remove + 1)
        
        #print(f"opt says [{a}, {b}, {c}]")
        
        poss_actions = []
        
        if state[0] == 0 and state[1] == 0 and state[2] == 0:
            return [-1, -1]
        
        if a == 0 and b == 0 and c == 0:
            return self.getPossibleMoves(state, poss_actions)
        
        bc = self.nim_sum(b, c)
        ns = self.nim_sum(a, bc)
        #print(f"nimsum: {ns}")
        if ns == 0:
            return self.getPossibleMoves(state, poss_actions)
        else:
            if self.nim_sum(ns, a) <= a:
                poss_actions.append( [0, a - self.nim_sum(ns, a)] )
            
            if self.nim_sum(ns, b) <= b:
                poss_actions.append( [1, b - self.nim_sum(ns, b)] )
            
            if self.nim_sum(ns, c) <= c:
                poss_actions.append( [2, c - self.nim_sum(ns, c)] )
            
            #print(f"poss_actions: {poss_actions}")
            return poss_actions

class QAgent: # q - learning agent
    def __init__(self, game, disc_rate, learn_rate, exp_rate):
        self.values = game.values.copy()
        self.disc_rate = disc_rate
        self.learn_rate = learn_rate
        self.exp_rate = exp_rate
        self.game = game

    def getAction(self, state): # chooses an action from the given state based on exploration vs exploitation
        #print("q says: {}".format(state))

        if random.random() < self.exp_rate:  # exp_rate is the probability of it exploring and choosing a random action
            acts = self.game.getPossibleMoves(state, self.values)
            i = random.randint(0, len(acts) - 1)
            ret = acts[i]
            #print("explore {}".format(ret))
            return ret

        if state == [0, 0, 0]:
            #print("exploit {}".format([-1, -1]))
            return [-1, -1]

        a = state[0]
        b = state[1]
        c = state[2]
        ret = [-1, -1]
        max_val = -1
        for i in range(0, 3):  # otherwise, it will choose the action with the highest value, which exploits the existing strategy
            for j in range(0, len(self.values[a][b][c][i])):
                if max_val < self.values[a][b][c][i][j]:
                    max_val = self.values[a][b][c][i][j]
                    ret = [i, j]
        #print("exploit {}".format(ret))
        return ret

    def updateValues(self, state, action, new_state, game_over, reward): # updates the q-table (learning happens here)
        a = state[0]
        b = state[1]
        c = state[2]
        d = action[0]
        e = action[1] - 1
        x = new_state[0]
        y = new_state[1]
        z = new_state[2]
        #print(a, end = ' ')
        #print(b, end = ' ')
        #print(c, end = ' ')
        #print(d, end = ' ')
        #print(e, end = ' ')
        #print(x, end = ' ')
        #print(y, end = ' ')
        #print(z)

        # Bellman equation for updating the q-table: q(s, a) = q(s, a) + alpha * (reward + gamma * max(q(s', a')) - q(s, a))
        # gamma is the discount rate, alpha is the learning rate, and reward is the reward received from the action taken
        # the max(q(s', a')) is the max value of the q-table predicted by the q-agent for the next possible move from the new state
        if (game_over):
            self.values[a][b][c][d][e] = (1.0 - self.learn_rate) * self.values[a][b][c][d][e] + self.learn_rate * reward
        else:
            self.values[a][b][c][d][e] = (1.0 - self.learn_rate) * self.values[a][b][c][d][e] + \
                                         self.learn_rate * (reward + self.disc_rate * max(big(self.values[x][y][z][0]),
                                                                                          big(self.values[x][y][z][1]),
                                                                                          big(self.values[x][y][z][2])))

    def updateExpRate(self, new_exp_rate):
        self.exp_rate = new_exp_rate


class Opponent: # a few different opponents 
    def __init__(self, game, opponent_type): # opponent_type specifies what kind of agent
        self.poss_actions = []
        self.type = opponent_type
        self.game = game

    def getOptMove(self, state): # returns an optimal move from the given state
        self.poss_actions = self.game.getOptMoves(state, self.poss_actions)
        i = random.randrange(0, len(self.poss_actions))
        return self.poss_actions[i]

    def getRandMove(self, state): # returns a random move from the given state
        actions = self.getPossibleMoves(state, self.values, self.poss_actions)
        ind = random.randrange(0, len(actions))
        return actions[ind]

    def getMalRandMove(self, state):  # returns a random non-optimal move from the given state
        bad_actions = self.game.getOptMoves(state, self.poss_actions)

        if state == [0, 0, 0]:
            return [-1, -1]

        self.poss_actions = []

        a = state[0]
        b = state[1]
        c = state[2]

        a_acts = len(self.game.values[a][b][c][0])
        b_acts = len(self.game.values[a][b][c][1])
        c_acts = len(self.game.values[a][b][c][2])

        for i in range(0, a_acts):
            if [0, i + 1] not in bad_actions:
                self.poss_actions.append([0, i + 1])
        for i in range(0, b_acts):
            if [1, i + 1] not in bad_actions:
                self.poss_actions.append([1, i + 1])
        for i in range(0, c_acts):
            if [2, i + 1] not in bad_actions:
                self.poss_actions.append([2, i + 1])

        x = random.randrange(0, len(self.poss_actions))
        return self.poss_actions[x]

    def getAction(self, state): # chooses the best action from the given state based on the type of agent
        #print('opp says: {}'.format(state))

        if self.type == 'opt':
            ret = self.getOptMove(state)
        if self.type == 'rand':
            ret = self.getRandMove(state)
        if self.type == 'malrand':
            ret = self.getMalRandMove(state)
        if self.type == 'mix':  # can use a probability distribution instead of picking the type
            rand = random.random()
            if (rand < 0.5):
                ret = self.getRandMove(state)
            else:
                ret = self.getRandMove(state)

        #print('action {}'.format(ret))
        return ret


def playQ(eps, game, q1, strat_error): # q-learning agent plays on its own [eps = # of episodes, game = Nim(), q1 = QAgent()]
    # strat_error: binary array: 1 if the q-agent played an optimal move, 0 if the q-agent did not

    for i in range(0, eps):
        #print('------------------')
        reward = 0
        while True:
            state = game.currState()
            #print("state: {}".format(state))
            action = q1.getAction(state)
            game.playMove(action)

            # error calculation
            good_actions = []
            good_actions = game.getOptMoves(state, good_actions)
            if action in good_actions:
                strat_error.append(1)
            else:
                strat_error.append(0)

            new_state = game.currState()
            #print("new state: {}".format(new_state))

            if new_state == [0, 0, 0]:
                reward = 1
                #print('game over')
                break

            q1.updateValues(state, action, new_state, False, reward)
   
        q1.updateValues(state, action, new_state, True, reward)

        if (eps == 10000):
            q1.updateExpRate(0.5)
        if (eps == 20000):
            q1.updateExpRate(0.25)
        if (eps == 50000):
            q1.updateExpRate(0)
        
        game.reset()


def playQvQ(eps, game, q1, q2, wins, q1_strat_error, q2_strat_error): # q-learning agent plays q-learning agent
    # wins: append 1 if q1 wins, append 2 if q2 wins
    # q1_strat_error: same as line 245 for q-agent 1
    # q2_strat_error: same as line 245 for q-agent 2

    for i in range(0, eps):
        #print('------------------')
        reward_q1 = 0
        reward_q2 = 0
        while True:
            state = game.currState()
            #print("state: {}".format(state))
            action = q1.getAction(state)
            game.playMove(action)

            # error calculation
            good_actions = []
            good_actions = game.getOptMoves(state, good_actions)
            if action in good_actions:
                q1_strat_error.append(1)
            else:
                q1_strat_error.append(0)

            new_state = game.currState()
            #print("new state: {}".format(new_state))

            if new_state == [0, 0, 0]:
                reward_q1 = 1
                reward_q2 = -1
                wins.append(1)
                #print('game over: q1 won')
                break

            q1.updateValues(state, action, new_state, False, reward_q1)

            state = new_state
            #print("state: {}".format(state))
            action = q2.getAction(state)
            game.playMove(action)

            # error calculation
            good_actions2 = []
            good_actions2 = game.getOptMoves(state, good_actions2)
            if action in good_actions2:
                q2_strat_error.append(1)
            else:
                q2_strat_error.append(0)

            new_state = game.currState()
            #print("new state: {}".format(new_state))

            if new_state == [0, 0, 0]:
                reward_q1 = -1
                reward_q2 = 1
                wins.append(2)
                #print('game over: q2 won')
                break

            q2.updateValues(state, action, new_state, False, reward_q2)

        q1.updateValues(state, action, new_state, True, reward_q1)
        q2.updateValues(state, action, new_state, True, reward_q2)
        
        if (eps == 10000):
            q1.updateExpRate(0.5)
            q2.updateExpRate(0.5)
        if (eps == 20000):
            q1.updateExpRate(0.25)
            q2.updateExpRate(0.25)
        if (eps == 50000):
            q1.updateExpRate(0)
            q2.updateExpRate(0)

        game.reset()

def playQvOpp(eps, game, q1, opp, wins, strat_error): # q-learning agent plays opponent
    # wins: append 1 if q1 wins, append -1 if opp wins
    # strat_error: same as line 245

    for i in range(0, eps):
        #print('------------------')
        #print(i)
        reward = 0
        while True:
            state = game.currState()
            #print("state: {}".format(state))
            action = q1.getAction(state)
            game.playMove(action)

            # error calculation
            good_actions = []
            good_actions = game.getOptMoves(state, good_actions)
            if action in good_actions:
                strat_error.append(1)
            else:
                strat_error.append(0)

            new_state = game.currState()
            #print("new state: {}".format(new_state))

            if new_state == [0, 0, 0]:
                reward = 1
                wins.append(1)
                #print('game over: q1 won')
                break

            q1.updateValues(state, action, new_state, False, reward)

            state = new_state
            #print("state: {}".format(state))
            action = opp.getAction(state)
            game.playMove(action)

            new_state = game.currState()
            #print("new state: {}".format(new_state))

            if new_state == [0, 0, 0]:
                reward = -1
                wins.append(-1)
                #print('game over: q1 lost')
                break

        q1.updateValues(state, action, new_state, True, reward)
        
        if (eps == 10000):
            q1.updateExpRate(0.5)
        if (eps == 20000):
            q1.updateExpRate(0.25)
        if (eps == 50000):
            q1.updateExpRate(0)
        
        game.reset()

