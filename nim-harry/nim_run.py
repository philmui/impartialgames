import numpy as np

from nim_env import NimEnv
from nim_rl import QAgent


def update():
    for episode in range(10000):
        if episode % 1000 == 0:
            print('Episode ' + str(episode))

        env.reset()

        while True:
            # print(state)
            state = env.get_state()
            action = RL.get_action(state)
            # print(action)
            next_state, reward, game_over = env.step(action)
            # print(next_state)
            # print()
            RL.update_q_table(state, action, reward, next_state)

            if game_over:
                break


def play():
    while True:
        print('Current Pile: ' + str(env.get_state()))

        pile = input('Enter Pile: ')
        amt = input('Enter Amount: ')

        env.update([int(pile), int(amt)])

        if np.sum(env.get_state()) == 0:
            print('Game Over You Win!')
            break

        print('Current Pile: ' + str(env.get_state()))

        action = RL.get_action(env.get_state(), play=True)

        print('Computer chooses from pile ' + str(action[0]) + ' with amount ' + str(action[1]))

        env.update(action)

        if np.sum(env.get_state()) == 0:
            print('Game Over Computer Wins')
            break


if __name__ == '__main__':
    env = NimEnv(n=3, stones_per_pile=9, max_remove=5)
    RL = QAgent(exp_rate=0.1, discount_rate=0.9, learning_rate=0.1, epsilon=0.1, nim_env=env)

    update()

    print(RL.get_q_table())

    env.reset()

    play()
