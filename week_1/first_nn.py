import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define and compile Neural Network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Providing data
xs = np.array([-1.0, 0.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 3.0, 5.0, 7.0], dtype=float)

# Training the Neural Network
model.fit(xs, ys, epochs=500)

# Predict
print(model.predict([10.0]))