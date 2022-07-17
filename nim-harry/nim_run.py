from nim_env import Nim
from nim_rl import QLearningTable


def update():
    for episode in range(100000):
        if episode % 10000 == 0:
            print('Episode: ' + str(episode))
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(str(observation))

            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                break

    print('game over')


if __name__ == '__main__':
    env = Nim()
    RL = QLearningTable()

    update()

    print(RL.q_table)
    compression_opts = dict(method='zip',
                            archive_name='out.csv')
    RL.q_table.to_csv('out.zip', index=False,
                      compression=compression_opts)
