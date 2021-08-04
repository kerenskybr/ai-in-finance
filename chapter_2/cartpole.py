import gym
import numpy as np
import pandas as pd
np.random.seed(100)
from pylab import plt
plt.style.use('seaborn')
import tensorflow as tf
tf.random.set_seed(100)
env = gym.make('CartPole-v0')
env.seed(100)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

action_size = env.action_space.n

[env.action_space.sample() for _ in range(10)]

state_size = env.observation_space.shape[0]

state = env.reset()

state, reward, done, _ = env.step(env.action_space.sample())

data = pd.DataFrame()
state = env.reset()
length = []
for run in range(25000):
    done = False
    prev_state = env.reset()
    treward = 1
    results = []
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        results.append({'s1': prev_state[0], 's2': prev_state[1],
            's3': prev_state[2], 's4': prev_state[3],
            'a': action, 'r': reward})
        treward += reward if not done else 0
        prev_state = state

    if treward >= 110:
        data = data.append(pd.DataFrame(results))
        length.append(treward)

print(np.array(length).mean())

print(data.info())

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=env.observation_space.shape[0]))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(data[['s1', 's2', 's3', 's4']], data['a'], epochs=25, verbose=False, validation_split=0.2)

res = pd.DataFrame(model.history.history)

def epoch():
    done = False
    state = env.reset()
    treward = 1
    while not done:
        action = np.where(model.predict(np.atleast_2d(state))[0][0] > 0.5, 1, 0)
        state, reward, done, _ = env.step(action)
        treward += reward if not done else 0
    return treward

res = np.array([epoch() for _ in range(100)])