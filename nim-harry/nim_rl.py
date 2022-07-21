from functools import partial

from nim_env import NimEnv
from collections import defaultdict
import pandas as pd
import numpy as np


def default():
    return 0.0

class QAgent:
    def __init__(self, exp_rate, discount_rate, learning_rate, epsilon, nim_env):
        self.exp_rate = exp_rate
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.nim_env = nim_env
        self.q_table = defaultdict(partial(defaultdict, default))

    def get_action(self, state):
        actions = self.nim_env.get_possible_actions(state)
        if np.random.rand() < self.epsilon:
            return actions[np.random.randint(0, len(actions))]
        else:
            max_val = -np.inf
            max_action = []
            for action in actions:

                if self.q_table[str(state)][str(action)] > max_val:
                    max_val = self.q_table[str(state)][str(action)]
                    max_action = [action]
                elif self.q_table[str(state)][str(action)] == max_val:
                    max_action.append(action)

            return max_action[np.random.randint(0, len(max_action))]

    def update_q_table(self, state, action, reward, next_state):
        # print(type(self.q_table[str(state)][str(action)]) is float)
        # print(self.q_table[str(next_state)][str(action)])
        # print(np.max(self.q_table[str(next_state)]))

        # print(reward)
        if np.sum(next_state) == 0:
            self.q_table[str(state)][str(action)] = (1 - self.learning_rate) * self.q_table[str(state)][str(action)] + self.learning_rate * reward
        else:
            self.q_table[str(state)][str(action)] = (1 - self.learning_rate) * self.q_table[str(state)][str(action)] + self.learning_rate * (reward + self.discount_rate * max(self.q_table[str(next_state)].values()))

        print(self.q_table[str(state)][str(action)])

    def get_q_table(self):
        return self.q_table
