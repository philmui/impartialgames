from env import Nim
from DQN import DQN
import numpy as np

if __name__ == "__main__":
    env = Nim(3, [1, 3, 5])
    agent = DQN(env=env)

    eps_hist = []
    for i in range(500):
        state = env.reset()

        score = 0

        while True:
            best_actions = env.get_optimal_action(state)

            action = agent.act(state.reshape(1, len(state)))
            new_state, reward, done = env.step(action)

            if action in best_actions:
                agent.update_accuracy(1)
            else:
                agent.update_accuracy(0)

            if done:
                if reward > 0:
                    agent.update_win_rate(1)
                else:
                    agent.update_win_rate(0)

            score += reward

            agent.remember(state.reshape(1, len(state)), action, reward, new_state.reshape(1, len(state)), done)
            agent.replay()
            agent.target_train()

            state = new_state.copy()
            if done:
                break

        if i != 0 and i % 50 == 0:
            agent.save("nim_model_ " + str(i) + ".h5")

        eps_hist.append(score)
        print("Episode: ", i, "Score: ", score, "Average score: ", np.mean(eps_hist[-100:]), "Epsilon: ", agent.get_epsilon())

    print("Done")

    state = env.reset()
    while True:
        state = env.reset()
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
