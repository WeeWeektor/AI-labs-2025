from itertools import product

import numpy as np
import tensorflow as tf

x = np.array(list(product([0, 1], repeat=3)))
y = np.array([0, 1, 1, 1, 0, 0, 0, 1])

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x, y, epochs=100, verbose=0)

loss, accuracy = model.evaluate(x, y, verbose=0)
print(f'loss: {loss}', f'accuracy: {accuracy}', sep='\n')

predictions = model.predict(x)
for inp, pred in zip(x, predictions):
    print(f'{inp} -> {round(pred[0])} (p={pred[0]:.2f})')
