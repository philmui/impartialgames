Basic Information about this version's neural network and game set up.

```py
self.gamma = 1
self.epsilon = 1.0
self.epsilon_min = 0.01
self.epsilon_decay = 0.995
self.learning_rate = 0.01

def create_model(self):
    model = Sequential()

    state_shape = np.array(self.env.get_state()).shape
    model.add(Dense(100, input_dim=state_shape[0], activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(self.env.action_space))
    model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

    print(model.summary())

    return model

env = Nim(3, [1, 3, 5])
agent = DQN(env=env)

# 500 Games
```