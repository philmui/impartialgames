import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions=[1, 2, 3], learning_rate=0.01, reward_decay=0.9, e_greedy=0.8):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            if int(observation) >= 3:
                state_action = self.q_table.loc[observation, :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                if int(observation) == 1:
                    action = 1
                elif int(observation) == 2:
                    state_action = self.q_table.loc[observation, :]
                    if state_action[1] > state_action[2]:
                        action = 1
                    elif state_action[2] > state_action[1]:
                        action = 2
                    else:
                        action = np.random.choice([1, 2])
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s)
        self.check_state_exist(s_)
        # if s == '1':
        #     print(s + ' ' + str(a))
        q_predict = self.q_table.loc[s, a]

        # print(r)
        if s_ != 'terminal':
            # print('sdfkajf')
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
            # print(q_target)
        else:
            # print('hi')
            q_target = r
        self.q_table.loc[s, a] = (1 - self.lr) * q_predict + self.lr * q_target

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )


"""
def get_env_feedback(S, A, q_table):
    take_amt = A % 10
    A /= 10
    pile_to_take = A - 1

    if S[pile_to_take] > 0:
        S[pile_to_take] -= take_amt
        if sum(S) == 0:
            R = 10
            S = 'terminal'
            return S, R

    _, opponent = choose_action(S, q_table)

    take_amt = opponent % 10
    opponent /= 10
    pile_to_take = opponent - 1

    if S[pile_to_take] > 0:
        S[pile_to_take] -= take_amt
        if sum(S) == 0:
            R = -10
            S = 'terminal'
            return S, R

    return S, 0
"""
