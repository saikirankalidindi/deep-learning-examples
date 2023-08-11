import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, datasets

# Load and preprocess the dataset (MNIST in this case)
(train_images, _), (_, _) = datasets.mnist.load_data()
train_images = (train_images.astype('float32') - 127.5) / 127.5
train_images = np.expand_dims(train_images, axis=-1)

# Define the generator model
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Create the generator
generator = make_generator_model()

# Define the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Create the discriminator
discriminator = make_discriminator_model()

# Define the loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Define the training loop
def train_gan(generator, discriminator, epochs, batch_size, noise_dim):
    for epoch in range(epochs):
        for step in range(train_images.shape[0] // batch_size):
            # Train the discriminator
            noise = np.random.randn(batch_size, noise_dim)
            generated_images = generator.predict(noise)
            real_images = train_images[step * batch_size : (step + 1) * batch_size]
            combined_images = np.concatenate([real_images, generated_images])
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            labels += 0.05 * np.random.random(labels.shape)  # Add noise to labels

            with tf.GradientTape() as disc_tape:
                predictions = discriminator(combined_images)
                d_loss = cross_entropy(labels, predictions)

            gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Train the generator
            noise = np.random.randn(batch_size, noise_dim)

            with tf.GradientTape() as gen_tape:
                generated_images = generator(noise)
                predictions = discriminator(generated_images)
                g_loss = cross_entropy(np.ones((batch_size, 1)), predictions)

            gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")

# Train the GAN
epochs = 50
batch_size = 128
noise_dim = 100
train_gan(generator, discriminator, epochs, batch_size, noise_dim)

# Generate and display some sample images
num_samples_to_generate = 10
noise = np.random.randn(num_samples_to_generate, noise_dim)
generated_images = generator.predict(noise)

plt.figure(figsize=(10, 1))
for i in range(num_samples_to_generate):
    plt.subplot(1, num_samples_to_generate, i + 1)
    plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
plt.show()
