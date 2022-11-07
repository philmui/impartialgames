from functools import partial

from nim_env import NimEnv
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def default():
    return 0.0


class QAgent:
    def __init__(self, name, discount_rate, learning_rate, epsilon):
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

    def get_action(self, state, play=False, lookup=False):
        actions = self.nim_env.get_possible_actions(state)
        if np.random.rand() < self.epsilon and not play and not lookup:
            action = actions[np.random.randint(0, len(actions))]
        else:
            max_val = -np.inf
            max_action = []
            '''
            if len(actions)==0:
                print("where are my actions")
            print(state)
            '''

            for action in actions:
                #print(self.q_table[str(state)][str(action)])
                if self.q_table[str(state)][str(action)] > max_val:
                    max_val = self.q_table[str(state)][str(action)]
                    max_action = [action]
                elif self.q_table[str(state)][str(action)] == max_val:
                    max_action.append(action)

            #print(len(max_action))
            action = max_action[np.random.randint(0, len(max_action))]

        opt_act = self.nim_env.get_optimal_action(state)

        self.accuracy.append(1 if action in opt_act else 0)

        return action

    def update_q_table(self, state, action, reward, next_state, env, flagged=False, acc_check=False):
    #def update_q_table(self, state, action, reward, next_state, flagged=False):
        #acc_check=False

        q_predict = self.q_table[str(state)][str(action)]
        acc = self.get_acc_check(env) if acc_check else False
        #acc=False
        #if acc:
        #    print("Acc is true, reward is ",reward)
        multiplier = 0.1 if acc else 1
        #multipler = 0.2 if acc_check else 1
        if np.sum(next_state) == 0:
            if reward < 0:
                self.wins.append(0)
            else:
                self.wins.append(1)
            q_target = reward
        else:
            q_target = reward + self.discount_rate * (0 if len(self.q_table[str(next_state)]) == 0 else max(self.q_table[str(next_state)].values()))
        #if (not acc_check) or (acc_check and self.get_acc_check(state, action, ((1 - self.learning_rate) * q_predict + self.learning_rate * q_target))):
        self.q_table[str(state)][str(action)] = (1 - (self.learning_rate * multiplier)) * q_predict + (self.learning_rate * multiplier) * q_target

    def get_acc_check(self, env): #, state, action, new_val):
        '''
        current_val=self.q_table[str(state)][str(action)]
        if (new_val>0 and current_val<0) or (new_val<0 and current_val>0):
            return False
        else:
            return True
        '''
        recent_acts=env.get_recent_actions()
        recent_states=env.get_recent_states()
        '''if recent_states[1]==[0,0,0]:
            return True'''
        if recent_states[0]!=[0,0,0]:
            my_act=self.get_action(recent_states[0], lookup=True)
            if my_act!=recent_acts[1]:
                return True
        return False



    def get_q_table(self):
        return self.q_table

    def set_env(self, env):
        self.nim_env = env

    def get_name(self):
        return self.name

    def get_epsilon(self):
        return self.epsilon

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
            #print("Updating wins, episode=", self.episode)

    def update_accurate(self):
        if self.episode % 500 == 0:

#        if self.episode % 50 == 0:
            self.accuracy_x.append(self.episode)
            self.accuracy_y.append(np.mean(self.accuracy))
            self.accuracy = []

    def plot(self, title,name):
        # print(self.wins)
        '''
        print("accuracy_x: ",self.accuracy_x)
        print("accuracy_y: ",self.accuracy_y)
        print("wins_x: ",self.wins_x)
        print("wins_y: ",self.wins_y)
       '''
        figure(figsize=(8, 6), dpi=100)

        plt.plot(self.accuracy_x, self.accuracy_y, label='Accuracy Rate', color='red')
        plt.plot(self.wins_x, self.wins_y, label='Win Rate', color='blue')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.title(title)
        #plt.show()
        plt.savefig('MaryamGraphs/v' + name)
