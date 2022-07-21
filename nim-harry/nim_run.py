from nim_env import NimEnv
from nim_rl import QAgent


def update():
    for episode in range(100):
        if episode % 1000 == 0:
            print('Episode ' + str(episode))

        state = env.reset()

        while not env.game_over:
            action = RL.get_action(state)
            next_state, reward, game_over = env.step(action)
            RL.update_q_table(state, action, reward, next_state)
            state = next_state


if __name__ == '__main__':
    env = NimEnv(n=1, stones_per_pile=13)
    RL = QAgent(exp_rate=0.1, discount_rate=0.9, learning_rate=0.1, epsilon=0.1, nim_env=env)

    update()

    print(RL.get_q_table())
