###########################################################
# nimtrain.py
# ---------------------------------------------------------
# @author: theodoremui
# @date: Nov, 2022
###########################################################
# ignore library compile target warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# after gym-0.25.2, the "step" API requires separate
#         terminated vs truncated return boolean 
__requires__ = ['gym <= 0.25.2']


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

from rl.core import Processor
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy

###########################################################
# SETUP
###########################################################

REPLAY_BUFFER_SIZE=10000
TRAIN_NB_STEPS=100000
PILES = np.array([2,3,4])
NUM_PILES = len(PILES)
AGENT_WEIGHTS_FILE = f"wts/dqnwts-{'-'.join(PILES.astype(str))}"

memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=1)

# policy = BoltzmannQPolicy()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.0,
                              value_min=0.1,
                              value_test=0.05,
                              nb_steps=TRAIN_NB_STEPS)

from nimenv import NimEnv, MIN_REWARD, MAX_REWARD

env = NimEnv(PILES)
obs = env.reset(seed=42)
print(f"checker: obs = {obs}, obs_space = {env.observation_space}")
# first check to make sure that the env is in compatible shape
from gym.utils.env_checker import check_env
check_env(env)

model = Sequential()

model.add(Flatten(input_shape=(1, NUM_PILES)))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
#model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(env.action_space.n, activation='softmax'))
print(model.summary())

class NimProcessor(Processor):
    def process_observation(self, observation):
        # assert len(observation) == len(PILES)
        return observation

    def process_state_batch(self, batch):
        return batch

    def process_reward(self, reward):
        return np.clip(reward, MIN_REWARD, MAX_REWARD)

class NimDQNAgent(DQNAgent):
    def forward(self, observation):
        action = super().forward(observation)
        count = 0
        while not env.done and not env.is_action_valid(action):
            action = super().forward(observation)
            count = count + 1
            if count> 4: # something is not right to choose so many times
                action = env.get_random_action()
                print(f"\t***> new action: {action}")
                break
        return action

agent = NimDQNAgent(model=model, 
                    nb_actions=env.action_space.n, 
                    memory=memory, 
                    processor=NimProcessor(),
                    nb_steps_warmup=500,
                    target_model_update=500, 
                    batch_size=128, 
                    gamma=0.99,
                    policy=policy)

###########################################################
# TRAINING
###########################################################
agent.compile(optimizer=Adam(learning_rate=1e-3),
              metrics=["mae"]) 

if os.path.exists(AGENT_WEIGHTS_FILE):
    print(f"Loading weights at: {AGENT_WEIGHTS_FILE} ...")
    agent.load_weights(AGENT_WEIGHTS_FILE)
else: 
    # RL training
    agent.fit(env, nb_steps=TRAIN_NB_STEPS, visualize=False, verbose=1)
    agent.save_weights(AGENT_WEIGHTS_FILE, overwrite=True)


###########################################################
# INTERACTIVE PLAY
###########################################################

def play():
    more_game = True
    while more_game:
        print(f"=======NEW GAME=========")
        env.reset()
        while not env.done:
            print(f"{env.state}")
            action = agent.forward(env.state)            
            obs, reward, done, info = env.step(action)
            print(f"\tAgent action: {action} => {obs} {reward} {done}")

            if env.done:
                print('\tGame over, agent wins')
            else:
                print(env.state)
                input_mode = True
                while input_mode:
                    try:
                        usr_pile = int(input('\tEnter pile (0-based): '))
                        stones_remove = int(input('\tStones to remove: '))
                        user_action = env.get_action(usr_pile, stones_remove)
                        if stones_remove != 0 and \
                           env.is_action_valid(user_action):
                            obs, reward, done, info = env.step(user_action)
                            print(f"\t{obs} {reward} {done}")
                            if env.done: print('\tGame over, you win')
                            input_mode = False
                        else: 
                            print('\tInvalid move: please try again')
                    except:
                        more_game = False # assume user wants to quit
                        input_mode = False
                        env.done = True

play()