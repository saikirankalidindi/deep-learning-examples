import numpy as np
import tensorflow as tf
from keras import layers

# Generate some sample data for a classification task
def generate_sample_data(num_samples=1000):
    X = np.random.rand(num_samples, 2)  # Two features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary classification based on the sum of features
    return X, y

# Generate sample data
X_train, y_train = generate_sample_data()

# Define the ANN model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Generate a new data point for prediction
new_data_point = np.array([[0.8, 0.5]])
predicted_class = model.predict_classes(new_data_point)
print("Predicted class:", predicted_class[0][0])
