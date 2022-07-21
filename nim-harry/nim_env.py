import numpy as np

"""
Nim Enviroment for specified amount of piles.
"""


class NimEnv:
    def __init__(self, n, stones_per_pile):
        self.n = n
        self.stones = stones_per_pile
        self.state = [stones_per_pile] * n
        self.game_over = False

    def reset(self):
        self.state = [self.stones] * self.n
        self.game_over = False
        return self.state

    def update(self, action):
        self.state[action[0]] -= action[1]

        if np.sum(self.state) == 0:
            self.game_over = True

        return self.state

    def get_state(self):
        return self.state

    def step(self, action):
        next_state = self.update(action)
        if self.game_over:
            return next_state, 1, True

        optimal_action = self.get_optimal_action()
        # print(self.state)
        # print('Optimal ' + str(len(optimal_action)))
        next_state = self.update(optimal_action[np.random.randint(0, len(optimal_action))])

        if self.game_over:
            return next_state, -1, True

        return next_state, 0, False

    def get_possible_actions(self, state):
        return [(i, j) for i in range(self.n) for j in range(1, state[i] + 1)]

    def get_optimal_action(self):
        xor = 0
        for i in range(self.n):
            xor ^= self.state[i]

        if xor == 0:
            return self.get_possible_actions(self.state)

        xor %= max(self.state)

        return [(i, xor) for i in range(self.n) if self.state[i] >= xor]
