from functools import partial

from nim_env import NimEnv
from collections import defaultdict
import pandas as pd
import numpy as np


def default():
    return 0.0

class QAgent:
    def __init__(self, discount_rate, learning_rate, epsilon, nim_env):
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.nim_env = nim_env
        self.q_table = defaultdict(partial(defaultdict, default))

    def get_action(self, state, play=False):
        actions = self.nim_env.get_possible_actions(state)
        if np.random.rand() < self.epsilon and not play:
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
        q_predict = self.q_table[str(state)][str(action)]
        if np.sum(next_state) == 0:
            q_target = reward
        else:
            q_target = reward + self.discount_rate * (0 if len(self.q_table[str(next_state)]) == 0 else max(self.q_table[str(next_state)].values()))

        self.q_table[str(state)][str(action)] += self.learning_rate * (q_target - q_predict)
        # print(self.q_table[str(state)][str(action)])

    def get_q_table(self):
        return self.q_table
