#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:26:30 2022

@author: varun
"""

import random


def nim_sum(a, b):
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


def getOptMoves(state, values, poss_actions):
    if state == [0, 0, 0]:
        return [-1, -1]
    # finish this
    return -1


def getPossibleMoves(state, values,
                     poss_actions):  # poss_actions is a param so that a new array is not created for every call of the function
    if state == [0, 0, 0]:
        return [[-1, -1]]

    poss_actions = []

    a = state[0]
    b = state[1]
    c = state[2]

    a_acts = len(values[a][b][c][0])
    b_acts = len(values[a][b][c][1])
    c_acts = len(values[a][b][c][2])

    for i in range(0, a_acts):
        poss_actions.append([0, i + 1])
    for i in range(0, b_acts):
        poss_actions.append([1, i + 1])
    for i in range(0, c_acts):
        poss_actions.append([2, i + 1])

    return poss_actions


def getRandMove(state, values, poss_actions):
    actions = getPossibleMoves(state, values, poss_actions)
    ind = random.randrange(0, len(actions) - 1)
    return actions[ind]


class Nim:
    def __init__(self, n):
        self.n = n
        self.state = [n, n, n]
        self.max_remove = n  # this assumes a player can remove any number of stones in a turn
        self.game_over = False

        self.values = [[[[[]] * 3] * (n + 1)] * (n + 1)] * (n + 1)
        print(len(self.values))
        print(len(self.values[0]))
        print(len(self.values[0][0]))
        print(len(self.values[0][0][0]))
        for i in range(0, n + 1):
            for j in range(0, n + 1):
                for k in range(0, n + 1):
                    for l in range(0, 3):
                        if l == 0:
                            self.values[i][j][k][l] = [0] * min(self.max_remove, i)
                        elif l == 1:
                            self.values[i][j][k][l] = [0] * min(self.max_remove, j)
                        else:
                            self.values[i][j][k][l] = [0] * min(self.max_remove, k)

    def playMove(self, action):
        if self.state == [0, 0, 0]:
            self.game_over = True
            return

        print("action = {}".format(action))
        self.state[action[0]] -= action[1]

    def currState(self):
        print(self.state)
        return self.state


class QAgent:
    def __init__(self, game, disc_rate, learn_rate, exp_rate):
        self.values = game.values.copy()
        self.disc_rate = disc_rate
        self.learn_rate = learn_rate
        self.exp_rate = exp_rate

    def getAction(self, state):
        print("q says: {}".format(state))

        if random.random() < self.exp_rate:  # exploration
            ret = getRandMove(state, self.values)
            print("explore {}".format(ret))
            return ret

        if state == [0, 0, 0]:
            print("exploit {}".format([-1, -1]))
            return [-1, -1]

        a = state[0]
        b = state[1]
        c = state[2]
        ret = [-1, -1]
        max_val = -1
        for i in range(0, 3):  # exploitation
            for j in range(0, len(self.values[a][b][c][i])):
                if max_val < self.values[a][b][c][i][j]:
                    max_val = self.values[a][b][c][i][j]
                    ret = [i, j]
        print("exploit {}".format(ret))
        return ret

    def updateValues(self, state, action, new_state, game_over, reward):
        a = state[0]
        b = state[1]
        c = state[2]
        d = action[0]
        e = action[1]
        x = new_state[0]
        y = new_state[1]
        z = new_state[2]
        if (game_over):
            self.values[a][b][c][d][e] = (1 - self.learn_rate) * self.values[a][b][c][d][e] + self.learn_rate * reward
        else:
            self.values[a][b][c][d][e] = (1 - self.learn_rate) * self.values[a][b][c][d][e] + \
                                         self.learn_rate * (reward + self.disc_rate * max(max(self.values[x][y][z][0]),
                                                                                          max(self.values[x][y][z][1]),
                                                                                          max(self.values[x][y][z][2])))

    def updateExpRate(self, new_exp_rate):
        self.exp_rate = new_exp_rate


class Opponent:
    def _init_(self, game, opponent_type):
        self.values = game.values.copy()
        self.poss_actions = []
        self.type = opponent_type

    def getOptMove(self, state):
        self.poss_actions = getOptMoves(state, self.values, self.poss_actions)
        i = random.randrange(0, len(self.poss_actions) - 1)
        return self.poss_actions[i]

    def getRandMove(self, state):
        action = getRandMove(state, self.values, self.poss_actions)
        return action

    def getMalRandMove(self, state):  # random non-optimal move
        bad_actions = getOptMoves(state, self.values, self.poss_actions)

        if state == [0, 0, 0]:
            return [-1, -1]

        self.poss_actions = []

        a = state[0]
        b = state[1]
        c = state[2]

        a_acts = len(self.values[a][b][c][0])
        b_acts = len(self.values[a][b][c][1])
        c_acts = len(self.values[a][b][c][2])

        for i in range(0, a_acts):
            if [0, i + 1] not in bad_actions:
                self.poss_actions.append([0, i + 1])
        for i in range(0, b_acts):
            if [1, i + 1] not in bad_actions:
                self.poss_actions.append([1, i + 1])
        for i in range(0, c_acts):
            if [2, i + 1] not in bad_actions:
                self.poss_actions.append([2, i + 1])

        x = random.randrange(0, len(self.poss_actions) - 1)
        return self.poss_actions[x]

    def getAction(self, state):
        print('opp says: {}'.format(state))

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

        print('action {}'.format(ret))
        return ret


def playQ(eps, game, q1):
    strat_error = []

    for i in range(0, eps):
        reward = 0
        while True:
            state = game.currState()
            action = q1.getAction(state)
            game.playMove(action)

            # error calculation
            good_actions = []
            good_actions = getOptMoves(state, game.values, good_actions)
            if action in good_actions:
                strat_error.append(1)
            else:
                strat_error.append(0)

            new_state = game.currState()

            if new_state == [0, 0, 0]:
                reward = 1
                print('game over')
                break

            q1.updateValues(state, action, new_state, False, reward)

        q1.updateValues(state, action, new_state, True, reward)

        # update exp rate


def playQvQ(eps, game, q1, q2):
    wins = []  # 1 if q1 wins, 2 if q2 wins
    q1_strat_error = []
    q2_strat_error = []

    for i in range(0, eps):
        reward_q1 = 0
        reward_q2 = 0
        while True:
            state = game.currState()
            action = q1.getAction(state)
            game.playMove(action)

            # error calculation
            good_actions = []
            good_actions = getOptMoves(state, game.values, good_actions)
            if action in good_actions:
                q1_strat_error.append(1)
            else:
                q1_strat_error.append(0)

            new_state = game.currState()

            if new_state == [0, 0, 0]:
                reward_q1 = 1
                reward_q2 = -1
                wins.append(1)
                print('game over: q1 won')
                break

            q1.updateValues(state, action, new_state, False, reward_q1)

            state = new_state
            action = q2.getAction(state)
            game.playMove(action)

            # error calculation
            good_actions = []
            good_actions = getOptMoves(state, game.values, good_actions)
            if action in good_actions:
                q2_strat_error.append(1)
            else:
                q2_strat_error.append(0)

            new_state = game.currState()

            if new_state == [0, 0, 0]:
                reward_q1 = -1
                reward_q2 = 1
                wins.append(2)
                print('game over: q2 won')
                break

            q2.updateValues(state, action, new_state, False, reward_q2)

        q1.updateValues(state, action, new_state, True, reward_q1)
        q2.updateValues(state, action, new_state, True, reward_q2)

        # update exp rate for both


def playQvOpp(eps, game, q1, opp):
    wins = []  # 1 if q1 wins, -1 if opp wins
    strat_error = []

    for i in range(0, eps):
        reward = 0
        while True:
            state = game.currState()
            action = q1.getAction(state)
            game.playMove(action)

            # error calculation
            good_actions = []
            good_actions = getOptMoves(state, game.values, good_actions)
            if action in good_actions:
                strat_error.append(1)
            else:
                strat_error.append(0)

            new_state = game.currState()

            if new_state == [0, 0, 0]:
                reward = 1
                wins.append(1)
                print('game over: q1 won')
                break

            q1.updateValues(state, action, new_state, False, reward)

            state = new_state
            action = opp.getAction(state)
            game.playMove(action)

            new_state = game.currState()

            if new_state == [0, 0, 0]:
                reward = -1
                wins.append(-1)
                print('game over: q1 lost')
                break

        q1.updateValues(state, action, new_state, True, reward)

        # update exp rate

