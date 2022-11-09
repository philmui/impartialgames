import numpy as np
import random

"""
Nim Enviroment for specified amount of piles.
"""


class Nim:
    def __init__(self, n, stones):
        self.n = n
        self.max_remove = max(stones)
        self.stones = stones
        self.state = stones.copy()
        self.game_over = False
        self.action_space = n * self.max_remove

    def is_legal(self, action):
        """
        Checks if action is legal.

        Action is an integer between 1 and n * max_remove
        State is a list of n integers

        Pile 0, remove 1 stone. Action is 1
        Pile 2, remove 3 stone. Action is 202
        """

        pile = action // self.max_remove
        remove = action % self.max_remove + 1

        if self.state[pile] < remove:
            return False

        return True

    def reset(self):
        self.state = self.stones.copy()
        self.game_over = False
        return np.array(self.state)

    def update(self, action):
        pile = action // self.max_remove
        remove = action % self.max_remove + 1

        self.state[pile] -= remove

        if sum(self.state) == 0:
            self.game_over = True

    def step(self, action):
        xor_sum = self.get_xor_sum(self.state)
        self.update(action)
        xor_sum_final = self.get_xor_sum(self.state)

        if self.game_over:
            return np.array(self.state), 1, True

        opt_act = random.choice(self.get_optimal_action(np.array(self.state)))
        self.update(opt_act)

        if self.game_over:
            return np.array(self.state), -1, True

        reward = 0
        if xor_sum != 0:
            if xor_sum_final == 0:
                reward = 1
            else:
                reward = -1

        return np.array(self.state), reward, False

    def get_xor_sum(self, state):
        xor = 0
        for i in range(self.n):
            xor ^= state[i] % (self.max_remove + 1)

        return xor

    def get_state(self):
        return np.array(self.state.copy())

    def get_possible_actions(self, state):
        state = state.reshape(self.n, )
        pos = []
        for i in range(self.n):
            for j in range(1, self.max_remove + 1):
                if state[i] >= j:
                    pos.append(i * self.max_remove + j - 1)

        return pos

    def get_optimal_action(self, state):
        state = state.reshape(self.n, )
        xor = 0
        for i in range(self.n):
            xor ^= state[i] % (self.max_remove + 1)

        if xor == 0:
            return self.get_possible_actions(state)

        poss_actions = []

        for i in range(self.n):
            a = state[i] % (self.max_remove + 1)
            ns = xor ^ a
            if ns < a:
                poss_actions.append(i * self.max_remove + (a - ns) - 1)

        return poss_actions

    def get_mal_random_action(self, state):
        opt_actions = self.get_optimal_action(state)

        poss_actions = self.get_possible_actions(state)

        ret = []
        for poss_action in poss_actions:
            if poss_action not in opt_actions:
                ret.append(poss_action)

        return ret if len(ret) != 0 else poss_actions

    def is_game_over(self):
        return self.game_over


"""
    def step(self, action, opts=[1, 0, 0], print_=False):
        state = self.get_state()
        opt_choice = self.get_optimal_action(self.state)
        next_state = self.update(action)

        if action in opt_choice:
            self.error.append(1)
        else:
            if print_:
                print(state)
                print(action)
                print(opt_choice)
                print(action in opt_choice)
                print()
            self.error.append(0)

        if np.sum(next_state) == 0:
            self.win.append(1)
            return next_state, 1, True

        if opts[0] == 1:
            actions = self.get_optimal_action(next_state)
            action = actions[np.random.randint(len(actions))]
        elif opts[1] == 1:
            actions = self.get_possible_actions(next_state)
            action = actions[np.random.randint(len(actions))]
        else:
            actions = self.get_mal_random_action(next_state)
            action = actions[np.random.randint(len(actions))]

        next_state = self.update(action)

        if np.sum(next_state) == 0:
            self.win.append(0)
            return next_state, -1, True

        return next_state, 0, False
    """