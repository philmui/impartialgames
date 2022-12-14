import numpy as np
import gym

from stable_baselines3 import DQN

from nimenv import NimEnv, MIN_REWARD, MAX_REWARD

PILES = np.array([3, 4, 5])
NUM_PILES = len(PILES)
AGENT_WEIGHTS_FILE = f"wts/dqnwts-{'-'.join(PILES.astype(str))}"

env = NimEnv(PILES)

class NimDQN(DQN):
    def predict(self, obs, **kwargs):
        action, _states = model.predict(obs, **kwargs)
        while not env.done and not env.is_action_valid(action):
            action, _states = model.predict(obs, **kwargs)
        return action, _states

model = NimDQN(policy="MlpPolicy", 
               env=env, 
               buffer_size=1000,
               batch_size=32,
               learning_starts=1000,
               target_update_interval=1000,
               verbose=1)
model.learn(total_timesteps=100, log_interval=1)
model.save("wts/stablenim")

# model = DQN.load("deepq_cartpole")

obs = env.reset()
for _ in range(10):
    env.render()

    action, _states = model.predict(obs, deterministic=True)
    print(f"\taction: {action}")
    obs, reward, done, info = env.step(action)
    print(f"\t{obs}, {reward}, {done}")
    if done:
      obs = env.reset()