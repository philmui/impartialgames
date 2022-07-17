import time

import numpy as np


class Nim:
    def __init__(self):
        self.action_space = [1, 2, 3]
        self.n_actions = len(self.action_space)
        self.state = 13

    def step(self, A):
        take_amt = A

        R = 0
        done = False
        if self.state - take_amt >= 0:
            self.state -= take_amt
            if self.state == 0:
                R = 1
                self.state = 'terminal'
                done = True
                return self.state, R, done

        A = 3
        if self.state <= 3:
            A = self.state
        elif self.state == 5 or self.state == 9:
            A = 1
        elif self.state == 6 or self.state == 10:
            A = 2
        else:
            A = np.random.choice([1, 2, 3])

        take_amt = A

        R = 0
        done = False
        if self.state - take_amt >= 0:
            self.state -= take_amt
            if self.state == 0:
                R = -1
                self.state = 'terminal'
                done = True
                return self.state, R, done

        return self.state, R, done

    def reset(self):
        self.state = 13
        return self.state

    def render(self):
        interaction = str(self.state)
        # print('\r{}'.format(interaction), end='')
        # time.sleep(1)



        # _, opponent = choose_action(S, q_table)
        #
        # take_amt = opponent % 10
        # opponent /= 10
        # pile_to_take = opponent - 1
        #
        # if S[pile_to_take] > 0:
        #     S[pile_to_take] -= take_amt
        #     if sum(S) == 0:
        #         R = -10
        #         S = 'terminal'
        #         return S, R

