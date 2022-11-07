from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import random
import numpy as np


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.45
        self.tau = 0.05

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()

        state_shape = np.array(self.env.get_state()).shape
        model.add(Dense(24, input_dim=state_shape[0], activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.env.action_space))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                pred = self.target_model.predict(new_state)[0]
                poss_actions = self.env.get_possible_actions(new_state)

                Q_future = max(pred[poss_actions])

                target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = load_model(name)
        self.target_model = load_model(name)

    def act(self, state, play=False):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon and not play:
            poss_actions = np.array(self.env.get_possible_actions(state))
            return np.random.choice(poss_actions)

        pred = self.model.predict(state)[0]
        poss_actions = self.env.get_possible_actions(state)

        mx = max(pred[poss_actions])
        return np.random.choice(np.where(pred == mx)[0])

    def get_epsilon(self):
        return self.epsilon
