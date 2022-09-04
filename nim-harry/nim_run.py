import numpy as np
from nim_env import NimEnv
from nim_rl import QAgent
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

# 1 - Optimal, 2 - Mal, 3 - Population
# 4 - Random


"""
YouTube Approach

{
a - Percentage of times where youtube recommends the same "side" engine or player to q1
b - Percentage of times where youtube recommends something random to q1

[a, b]

[a, a+b]

random = np.random.rand()
if random < youtube_probability[0]:
    QvQ(env, q1, q1.get_side())
else:
    QvQ(env, q1, (not q1.get_side()))
}
"""

def youtube_train(people, piles, rounds=50000, recommendation=[0, 1.0], discern_rate=0):
    recommendation[1] += recommendation[0]

    for r in range(rounds):
        if r % 1000 == 0:
            print('Round: ', r)
            if r + 50000 == rounds:
                for q in people:
                    q.set_epsilon(0)

        rand = np.random.rand()

        p1 = np.random.randint(0, len(people))
        q1 = people[p1]
        side = q1.get_side()
        env = NimEnv(len(piles), piles)
        q1.set_env(env)

        possible = None

        if r < rounds // 2:
            possible = 'optimal' if side == 0 else 'mal'
        else:
            if rand < recommendation[0]:
                possible = 'optimal' if side == 0 else 'mal'
            elif rand < recommendation[1]:
                possible = 'optimal' if side == 1 else 'mal'

        QvQ(env, q1, possible)

    for q in people:
        q.plot(q.get_name() + ' performance with ' + str(int(recommendation[0] * 100.0)) + '% suggesting biased content', 'YouTube Plots',
                     q.get_name() + '_' + str(int(recommendation[0] * 100.0)) + '.png', 13)



"""
First parameter is Optimal
Second parameter is Mal
Third parameter is Population
"""
def population_train(people, piles, rounds=50000, probability=[0, 0, 1], discern_rate=0):
    for i in range(1, len(probability)):
        probability[i] = probability[i - 1] + probability[i]

    print(probability)

    for r in range(rounds):
        if r % 1000 == 0:
            print('Round: ', r)
            if r + 20000 >= rounds:
                RL1.set_epsilon(0)
                RL2.set_epsilon(0)

        trained = 0

        rand = np.random.rand()

        p1 = np.random.randint(0, len(people))
        q1 = people[p1]
        env = NimEnv(len(piles), piles)
        q1.set_env(env)

        if rand < probability[0]:
            QvQ(env, q1, 'optimal', discern_rate)
            continue

        if rand < probability[1]:
            QvQ(env, q1, 'mal', discern_rate)
            continue

        while trained < len(people) // 2:
            p2 = np.random.randint(0, len(people))
            while p1 == p2:
                p2 = np.random.randint(0, len(people))

            q2 = people[p2]
            q2.set_env(env)

            QvQ(env, q1, q2, discern_rate)
            trained += 2

    for q in people:
        q.plot(
            q.get_name() + ' performance with ' + str(probability) + ' population and discern rate '
            + str(int(discern_rate*100)) + '%', 'Discerning Plots',
            q.get_name() + '_' + str(int(discern_rate * 100.0)) + '.png', 8)


def QvQ(env, q1, q2, discern=0):
    env.reset()
    reward_q1 = 0
    agent = True
    flagged = False
    if q2 == 'optimal' or q2 == 'mal':
        q2_lean = 1 if q2 == 'optimal' else -1
        agent = False

    # random decimal between 0 and 1
    rand = np.random.rand()

    if rand < discern and q2_lean == -1:
        # print(str(q2_lean) + ' ' + str(rand))
        flagged = True

    while True:
        state = env.get_state()
        q1_action = q1.get_action(state)
        next_state = env.update(q1_action)

        if np.sum(next_state) == 0:
            reward_q1 = 1
            break

        state2 = next_state.copy()

        if agent:
            action = q2.get_action(state2)
        elif q2 == 'optimal':
            actions = env.get_optimal_action(state2)
            action = actions[np.random.randint(len(actions))]
        else:
            actions = env.get_mal_random_action(state2)
            action = actions[np.random.randint(len(actions))]

        next_state = env.update(action)

        if np.sum(next_state) == 0:
            reward_q1 = -1
            break

        q1.update_q_table(state, q1_action, reward_q1, next_state, flagged)

    ### HERE
    # Reward < 0, q1 loses, reward > 0, q1 wins
    # if q2 was mal
    # random float between 0 and 1
    # if random < dicerning rate, flagged

    q1.update_q_table(state, q1_action, reward_q1, next_state, flagged)

    q1.add_points()


def play(q, pile):
    env = NimEnv(len(pile), pile)
    env.reset()
    while True:
        print('Current Pile: ' + str(env.get_state()))

        pile = input('Enter Pile: ')
        amt = input('Enter Amount: ')

        env.update([int(pile) - 1, int(amt)])

        if np.sum(env.get_state()) == 0:
            print('Game Over You Win!')
            break

        print('Current Pile: ' + str(env.get_state()))

        action = q.get_action(env.get_state(), play=True)

        print('Computer chooses from pile ' + str(action[0] + 1) + ' with amount ' + str(action[1]))

        env.update(action)

        if np.sum(env.get_state()) == 0:
            print('Game Over Computer Wins')
            break


if __name__ == '__main__':
    """
    tests = [[1, 0], [0.99, 0.01], [0.95, 0.05], [0.90, 0.10], [0.50, 0.50], [0.10, 0.90], [0.05, 0.95], [0.01, 0.99], [0, 1]]

    for i in range(len(tests)):
        RL1 = QAgent('Q1', discount_rate=1, learning_rate=0.45, epsilon=0.1, side=0)
        RL2 = QAgent('Q2', discount_rate=1, learning_rate=0.45, epsilon=0.1, side=1)

        people = [RL1, RL2]
        youtube_train(people, [1, 3, 5], rounds=200000, recommendation=tests[i])
    """

    RL1 = QAgent('Q1', discount_rate=1, learning_rate=0.45, epsilon=0.1, side=0)
    RL2 = QAgent('Q2', discount_rate=1, learning_rate=0.45, epsilon=0.1, side=1)
    population = [RL1, RL2]
    population_train(population, [1, 3, 5], rounds=50000, probability=[0.01, 0.99, 0], discern_rate=1)