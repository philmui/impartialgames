from functools import partial

from nim_env import NimEnv
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def default():
    return 0.0


class QAgent:
    def __init__(self, name, discount_rate, learning_rate, epsilon, side):
        # SIDE: 0-Optimal, 1-Mal
        self.side = side
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.nim_env = None
        self.name = name
        self.episode = 0
        self.wins_x = []
        self.wins_y = []
        self.accuracy_x = []
        self.accuracy_y = []
        self.wins = []
        self.accuracy = []
        self.q_table = defaultdict(partial(defaultdict, default))

    def get_action(self, state, play=False):
        actions = self.nim_env.get_possible_actions(state)
        if np.random.rand() < self.epsilon and not play:
            action = actions[np.random.randint(0, len(actions))]
        else:
            max_val = -np.inf
            max_action = []
            for action in actions:

                if self.q_table[str(state)][str(action)] > max_val:
                    max_val = self.q_table[str(state)][str(action)]
                    max_action = [action]
                elif self.q_table[str(state)][str(action)] == max_val:
                    max_action.append(action)

            action = max_action[np.random.randint(0, len(max_action))]

        opt_act = self.nim_env.get_optimal_action(state)

        self.accuracy.append(1 if action in opt_act else 0)

        return action

    def update_q_table(self, state, action, reward, next_state, flagged=False):
        q_predict = self.q_table[str(state)][str(action)]

        beta = 0.2 if flagged else 1

        if np.sum(next_state) == 0:
            if reward < 0:
                self.wins.append(0)
            else:
                self.wins.append(1)
            q_target = beta * reward
        else:
            # q_target = beta * reward + self.discount_rate * (0 if len(self.q_table[str(next_state)]) == 0 else max(self.q_table[str(next_state)].values()))
            q_target = beta * reward \
                       + self.discount_rate * (0 if len(self.q_table[str(next_state)]) == 0 else max(self.q_table[str(next_state)].values())) \
                       #- q_predict


        self.q_table[str(state)][str(action)] = (1 - self.learning_rate) * q_predict + self.learning_rate * q_target

    def get_q_table(self):
        return self.q_table

    def set_env(self, env):
        self.nim_env = env

    def get_side(self):
        return self.side

    def get_name(self):
        return self.name

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def add_points(self):
        self.episode += 1
        self.update_accurate()
        self.update_win()

    def add_win(self):
        self.wins.append(1)

    def update_win(self):
        if self.episode % 500 == 0:
            self.wins_x.append(self.episode)
            self.wins_y.append(np.mean(self.wins))
            self.wins = []

    def update_accurate(self):
        if self.episode % 500 == 0:
            self.accuracy_x.append(self.episode)
            self.accuracy_y.append(np.mean(self.accuracy))
            self.accuracy = []

    def plot(self, title):
        # print(self.wins)
        plt.plot(self.accuracy_x, self.accuracy_y, label='Accuracy Rate', color='red')
        plt.plot(self.wins_x, self.wins_y, label='Win Rate', color='blue')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.title(title)
        plt.show()
