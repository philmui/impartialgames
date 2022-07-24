import numpy as np
import matplotlib.pyplot as plt
from nim_env import NimEnv
from nim_rl import QAgent


def population_train(people, num_pile, rounds=50000):
    for round in range(rounds):
        if round % 1000 == 0:
            print('Round: ', round)
        trained = 0
        while trained < len(people) // 2:
            p1 = np.random.randint(0, len(people))
            p2 = np.random.randint(0, len(people))
            while p1 == p2:
                p2 = np.random.randint(0, len(people))

            q1 = people[p1]
            q2 = people[p2]

            """
            initialize piles array to have a length of num_pile and fill it with random numbers between 1 and 10
            """

            piles = np.random.randint(1, 10, num_pile)

            E = NimEnv(num_pile, piles)
            q1.set_env(E)
            q2.set_env(E)

            QvQ(E, q1, q2)

            trained += 2

    for q in people:
        q.plot(q.get_name() + ' performance')


def QvQ(E, q1, q2):
    E.reset()
    reward_q1 = 0
    reward_q2 = 0
    while True:
        state = E.get_state()
        q1_action = q1.get_action(state)
        next_state = E.update(q1_action)

        if np.sum(next_state) == 0:
            reward_q1 = 1
            reward_q2 = -1
            break

        q1.update_q_table(state, q1_action, reward_q1, next_state)

        state = next_state.copy()

        q2_action = q2.get_action(state)
        next_state = E.update(q2_action)

        if np.sum(next_state) == 0:
            reward_q1 = -1
            reward_q2 = 1
            break

        q2.update_q_table(state, q2_action, reward_q2, next_state)

    q1.update_q_table(state, q1_action, reward_q1, next_state)
    q2.update_q_table(state, q2_action, reward_q2, next_state)

    q1.add_points()
    q2.add_points()


# def trainQvsAny(option): # first argument is for percentage optimal, second for percentage random, third for percentage mal optimal
#     x_error = []
#     y_error = []
#     x_win = []
#     y_win = []
#     for episode in range(50000):
#         if episode % 1000 == 0:
#             print('Episode: ' + str(episode))
#             RL.set_epsilon(RL.epsilon * 0.95)
#
#
#         if episode % 100 == 0:
#             x_error.append(episode)
#             y_error.append(np.mean(env.get_error()))
#
#             x_win.append(episode)
#             y_win.append(np.mean(env.get_win()))
#             env.reset_win()
#             env.reset_error()
#             # print('Episode ' + str(episode))
#
#         env.reset()
#
#         rand = np.random.rand()
#         flagged = False
#         if rand < option[0]:
#             opts = [1, 0, 0]
#         elif rand < option[0] + option[1]:
#             opts = [0, 1, 0]
#         else:
#             opts = [0, 0, 1]
#             rand2 = np.random.rand()
#             if rand2 < env.flag:
#                 flagged = True
#
#         while True:
#             state = env.get_state()
#             action = RL.get_action(state)
#
#             next_state, reward, game_over = env.step(action, opts=opts, print_=False)
#
#             RL.update_q_table(state, action, reward, next_state, flagged)
#
#             if game_over:
#                 break
#
#
#     print(RL.epsilon)
#     plt.plot(x_error, y_error, label='Accuracy Rate', color='red')
#     plt.plot(x_win, y_win, label='Win Rate', color='blue')
#     plt.legend()
#     plt.title('Mix 10/90 agent Accuracy Rate and Win Rate')
#     plt.show()

"""
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
"""

if __name__ == '__main__':
    RL1 = QAgent('Q1', discount_rate=1, learning_rate=0.45, epsilon=0.1)
    RL2 = QAgent('Q2', discount_rate=1, learning_rate=0.45, epsilon=0.1)
    RL3 = QAgent('Q3', discount_rate=1, learning_rate=0.45, epsilon=0.1)

    population_train([RL1, RL2, RL3], 2, rounds=50000)
    # env = NimEnv(n=3, stones=[15, 10, 7], flag=0.5)
    # RL = QAgent(discount_rate=1, learning_rate=0.1, epsilon=0.1, nim_env=env)
    #
    # trainQvsAny([0.4, 0, 0.6])
    #
    # # print(RL.get_q_table())
    #
    # while True:
    #     play()
