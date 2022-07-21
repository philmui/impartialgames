from nim_env import NimEnv
from nim_rl import QAgent


def update():
    for episode in range(10000):
        if episode % 1000 == 0:
            print('Episode ' + str(episode))

        env.reset()

        state = env.get_state()

        while True:
            # print(state)
            action = RL.get_action(state)
            # print(action)
            next_state, reward, game_over = env.step(action)
            # print(next_state)
            # print()
            RL.update_q_table(state, action, reward, next_state)
            state = next_state.copy()

            if game_over:
                break


if __name__ == '__main__':
    env = NimEnv(n=3, stones_per_pile=8, max_remove=3)
    RL = QAgent(exp_rate=0.1, discount_rate=0.9, learning_rate=0.1, epsilon=0.1, nim_env=env)

    update()

    print(RL.get_q_table())
