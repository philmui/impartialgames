from env import Nim
from DQN import DQN
import numpy as np

if __name__ == "__main__":
    env = Nim(3, [1, 3, 5])
    agent = DQN(env=env)

    eps_hist = []
    for i in range(1000):
        state = env.reset()

        score = 0

        while True:
            action = agent.act(state.reshape(1, len(state)))
            new_state, reward, done = env.step(action)

            score += reward

            agent.remember(state.reshape(1, len(state)), action, reward, new_state.reshape(1, len(state)), done)
            agent.replay()

            state = new_state.copy()
            if done:
                break

        eps_hist.append(score)
        print("Episode: ", i, "Score: ", score, "Average score: ", np.mean(eps_hist[-100:]), "Epsilon: ", agent.get_epsilon())

        if i % 10 == 0:
            agent.target_train()

    print("Done")

    state = env.reset()
    while True:
        while True:
            print(state)
            action = agent.act(state.reshape(1, len(state)), play=True)
            print('Agent action: ', action)

            env.update(action)

            state = env.get_state()

            if env.is_game_over():
                print('Game over, agent wins')

            print(state)

            usr_pile = int(input('Enter pile: '))
            usr_remove = int(input('Enter remove: '))

            env.update(usr_pile * env.max_remove + usr_remove - 1)

            state = env.get_state()

            if env.is_game_over():
                print('Game over, you win')
