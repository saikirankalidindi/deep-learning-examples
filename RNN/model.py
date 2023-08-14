#import required libararies
import numpy as np
import tensorflow as tf
from tensoflow import keras
from keras import layers

# Generate some sample data for a time series prediction task
def generate_sample_data(seq_length=100, num_samples=1000):
    X = np.zeros((num_samples, seq_length, 1))
    y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        X[i, :, 0] = np.sin(np.linspace(0, 20, seq_length) + np.random.rand() * 0.2)
        y[i, 0] = np.sin(21)
    return X, y

# Generate sample data
X_train, y_train = generate_sample_data()

# Define the RNN model
model = tf.keras.Sequential([
    layers.SimpleRNN(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict using the trained model
sample_input = np.sin(np.linspace(20.1, 40.1, X_train.shape[1])).reshape(1, -1, 1)
predicted_output = model.predict(sample_input)
print("Predicted output:", predicted_output[0, 0])
