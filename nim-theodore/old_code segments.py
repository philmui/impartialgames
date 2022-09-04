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

"""
def trainQvsAny(option): # first argument is for percentage optimal, second for percentage random, third for percentage mal optimal
    x_error = []
    y_error = []
    x_win = []
    y_win = []
    for episode in range(50000):
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

        rand = np.random.rand()
        flagged = False
        if rand < option[0]:
            opts = [1, 0, 0]
        elif rand < option[0] + option[1]:
            opts = [0, 1, 0]
        else:
            opts = [0, 0, 1]
            rand2 = np.random.rand()
            if rand2 < env.flag:
                flagged = True

        while True:
            state = env.get_state()
            action = RL.get_action(state)

            next_state, reward, game_over = env.step(action, opts=opts, print_=False)

            RL.update_q_table(state, action, reward, next_state, flagged)

            if game_over:
                break


    print(RL.epsilon)
    plt.plot(x_error, y_error, label='Accuracy Rate', color='red')
    plt.plot(x_win, y_win, label='Win Rate', color='blue')
    plt.legend()
    plt.title('Mix 10/90 agent Accuracy Rate and Win Rate')
    plt.show()
"""

"""
for episode in range(50000):
    if episode % 1000 == 0:
        print('Episode: ' + str(episode))
        if episode > 40000:
            RL1.set_epsilon(0)
            
    E = NimEnv(3, [15, 10, 7])
    RL1.set_env(E)
    QvQ(E, RL1, 'optimal')

RL1.plot(RL1.get_name() + ' performance')
"""