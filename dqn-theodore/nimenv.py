#####################################################
# nimenv.py
# ---------------------------------------------------
# @author: theodoremui
# @date: Nov, 2022
#####################################################

import numpy as np
import random

from gym import Env, spaces

MAX_REWARD = 2
MIN_REWARD = -1
INC_REWARD = 1

class NimEnv(Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, stones):
        '''
        example stones: [3, 4, 5]
        '''
        super(NimEnv, self).__init__()

        # get number of stone piles
        self.num_piles = len(stones)
        self.stones = stones

        self.reward_range = (MIN_REWARD, MAX_REWARD)
        self.action_space = spaces.Discrete(np.sum(self.stones))
        # including the max stones in each pile: needs to add ones
        self.observation_space = \
            spaces.MultiDiscrete(stones+np.ones(len(stones)))

        self.reset()

    def reset(self, return_info=False, **kwargs):
        # super(NimEnv, self).reset(**kwargs)
        self.state = self.stones.copy()
        self.done = False
        self.truncated = False
        if return_info:
            return np.array(self.state), {}
        return np.array(self.state)

    def render(self, prefix=""):
        print(f"{prefix}{self.state}")

    def close(self):
        self.reset()

    # is action valid for the current 'state'?
    def is_action_valid(self, action):
        valid = False
        for pile in range(self.num_piles):
            if action <= self.stones[pile]: # use the original 'self.stones'
                if action > 0 and action <= self.state[pile]: valid = True
                break
            else:
                action -= self.stones[pile] # use the original 'self.stones'
        return valid

    # assumes:
    # piles starts with 0
    # num_stones starts with 0
    def get_action(self, which_pile, num_stones):
        action = 0
        for p in range(which_pile):
            action += self.stones[p]
        action += num_stones
        return action

    # updating the "self.state" based on new "action"
    # note that "action" is a number 0<=action<=max move
    # ASSUME: action (# stones to remove from a pile) is valid
    def _update(self, action):
        assert self.is_action_valid(action), f"action {action} is invalid"
        for pile in range(self.num_piles):
            if action <= self.state[pile]:
                self.state[pile] -= action
                break
            else:
                action -= self.stones[pile] # use the original 'self.stones'

        if sum(self.state) == 0:
            self.done = True

    def step(self, action):
        reward = 0
        xor_sum_initial = self.get_xor_sum(self.state)
        try:
            self._update(action)
        except: pass # invalid action
        xor_sum_final = self.get_xor_sum(self.state)

        if xor_sum_initial == 0:
            if xor_sum_final == 0:
                reward = INC_REWARD
            else:
                reward = MIN_REWARD
        elif xor_sum_final == 0:
                reward = MAX_REWARD

        info = {}
        return np.array(self.state), reward, self.done, info

    def get_xor_sum(self, state):
        xor = 0
        for i in range(len(state)):
            xor ^= state[i]
        return xor

    def get_state(self):
        return np.array(self.state.copy())

    # def get_possible_actions(self, state):
    #     num_stones_left = np.sum(state)
    #     return [x for x in range(1,num_stones_left+1)]
    def get_possible_actions(self):
        num_orig_stones = np.sum(self.stones)
        return [x for x in range(1,num_orig_stones+1)]

    def get_random_action(self):
        possibles = self.get_possible_actions()
        action = possibles[np.random.randint(len(possibles))]
        while not self.is_action_valid(action):
            action = possibles[np.random.randint(len(possibles))]
        return action

    def get_optimal_action(self, state):
        xor_sum = self.get_xor_sum(state)
        if xor_sum == 0:
            #return self.get_possible_actions(np.array(state))
            return self.get_possible_actions()

        possible_actions = []

        for p in range(len(state)):
            num_stones = state[p]
            ns = xor_sum ^ num_stones
            if ns < num_stones:
                # The winning move is to reduce the size of pile p to ns
                # Reference: https://en.wikipedia.org/wiki/Nim
                possible_actions.append(self.get_action(p, num_stones-ns))

        return possible_actions

    def get_mal_random_action(self, state):
        opt_actions = self.get_optimal_action(state)

        #poss_actions = self.get_possible_actions(state)
        poss_actions = self.get_possible_actions()

        ret = []
        for poss_action in poss_actions:
            if poss_action not in opt_actions:
                ret.append(poss_action)

        return ret if len(ret) != 0 else poss_actions
