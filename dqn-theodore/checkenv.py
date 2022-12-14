import numpy as np
from nimenv import NimEnv

env = NimEnv([3,4,5])

print(f"===> {env.state}")
actions = env.get_optimal_action(np.array(env.state))
action = actions[np.random.randint(len(actions))]
env.step(action)
print('Agent action: ', action)

