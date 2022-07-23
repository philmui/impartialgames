import numpy as np
import matplotlib.pyplot as plt
from nim_env import NimEnv
from nim_rl import QAgent


def trainQvsAny(option): # first argument is for percentage optimal, second for percentage random, third for percentage mal optimal
    x_error = []
    y_error = []
    x_win = []
    y_win = []
    for episode in range(100000):
        if episode % 1000 == 0:
            print('Episode: ' + str(episode))
            RL.set_epsilon(RL.epsilon * 0.95)


        if episode % 100 == 0:
            x_error.append(episode)
            y_error.append(np.mean(env.get_error()))

            x_win.append(episode)
            y_win.append(np.mean(env.get_win()))
            env.reset_win()
            env.reset_error()
            # print('Episode ' + str(episode))

        env.reset()

        opts = [0, 0, 0]

        rand = np.random.rand()
        if rand < option[0]:
            opts = [1, 0, 0]
        elif rand < option[0] + option[1]:
            opts = [0, 1, 0]
        else:
            opts = [0, 0, 1]

        while True:
            state = env.get_state()
            action = RL.get_action(state)

            if episode > 99900:
                next_state, reward, game_over = env.step(action, opts=opts, print_=True)
            else:
                next_state, reward, game_over = env.step(action, opts=opts, print_=False)

            RL.update_q_table(state, action, reward, next_state)

            if game_over:
                break


    print(RL.epsilon)
    plt.plot(x_error, y_error, label='Accuracy Rate', color='red')
    plt.plot(x_win, y_win, label='Win Rate', color='blue')
    plt.legend()
    plt.title('Mix 10/90 agent Accuracy Rate and Win Rate')
    plt.show()


def play():
    env.reset()
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
    env = NimEnv(n=3, stones=[5, 10, 7], max_remove=3)
    RL = QAgent(discount_rate=1, learning_rate=0.1, epsilon=0.1, nim_env=env)

    # print(env.get_optimal_action([10, 6]))

    trainQvsAny([1, 0, 0])

    print(RL.get_q_table())

    while True:
        play()
