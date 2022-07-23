import numpy as np

"""
Nim Enviroment for specified amount of piles.
"""


class NimEnv:
    def __init__(self, n, stones, max_remove):
        self.n = n
        self.max_remove = max_remove
        self.stones = stones
        self.state = stones.copy()
        self.error = []
        self.win = []
        self.game_over = False

    def reset(self):
        self.state = self.stones.copy()
        self.game_over = False
        return self.state

    def update(self, action):
        self.state[action[0]] -= action[1]

        if np.sum(self.state) == 0:
            self.game_over = True

        return self.state

    def get_state(self):
        return self.state.copy()

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

        first_limit = opts[0]
        second_limit = opts[0] + opts[1]

        rand = np.random.rand()

        if rand < first_limit:
            # print(next_state)
            actions = self.get_optimal_action(next_state)
            action = actions[np.random.randint(len(actions))]
        elif rand < second_limit:
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

    def get_possible_actions(self, state):
        return [[i, j] for i in range(self.n) for j in range(1, min(state[i] + 1, self.max_remove + 1))]

    def get_optimal_action(self, state):
        # print(state)
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
                poss_actions.append([i, a - ns])

        return poss_actions

        # xor = min(xor, max(state))
        #
        # return [[i, xor] for i in range(self.n) if state[i] >= xor]

    def get_mal_random_action(self, state):
        opt_actions = self.get_optimal_action(state)

        poss_actions = self.get_possible_actions(state)

        ret = []
        for poss_action in poss_actions:
            if poss_action not in opt_actions:
                ret.append(poss_action)

        return ret if len(ret) != 0 else poss_actions

    def reset_error(self):
        self.error = []

    def get_error(self):
        return self.error.copy()

    def reset_win(self):
        self.win = []

    def get_win(self):
        return self.win.copy()
