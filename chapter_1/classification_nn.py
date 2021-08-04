import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import pandas as pd

f = 5
n = 10

np.random.seed(100)

x = np.random.randint(0, 2, (n, f))
y = np.random.randint(0, 2, n)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=f))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

h = model.fit(x, y, epochs=50, verbose=False)

res = pd.DataFrame(h.history)
res.plot(figsize=(10, 6));

plt.show()