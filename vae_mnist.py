import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio
import optuna
from tensorflow.keras import layers, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.stats import norm
from optuna.visualization import plot_optimization_history, plot_contour, plot_slice

# Constants
LATENT_SPACE_PLOTS_DIR = '/content/gdrive/MyDrive/NJIT/CS370/Final Exam/latent_space_plots'
GENERATED_IMAGES_PLOTS_DIR = '/content/gdrive/MyDrive/NJIT/CS370/Final Exam/generated_images_plots'
EPOCHS = 50  # Change as needed
OPTUNA_TRIALS = 15  # Change as needed

# Ensure the directories exist
os.makedirs(LATENT_SPACE_PLOTS_DIR, exist_ok=True)
os.makedirs(GENERATED_IMAGES_PLOTS_DIR, exist_ok=True)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
original_dim = x_train.shape[1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Sampling function for VAE
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Build VAE model
def build_vae(latent_dim, intermediate_dim, learning_rate):
    # Encoder
    inputs = layers.Input(shape=(original_dim,), name='encoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # VAE loss
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs) * original_dim
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1) * -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Compile VAE
    optimizer = Adam(learning_rate=learning_rate)
    vae.compile(optimizer=optimizer)
    return vae, encoder, decoder

# Optuna objective function
def objective(trial):
    latent_dim = trial.suggest_int('latent_dim', 2, 20)
    intermediate_dim = trial.suggest_int('intermediate_dim', 128, 1024)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    vae, _, _ = build_vae(latent_dim, intermediate_dim, learning_rate)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = vae.fit(
        x_train, x_train,
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[early_stopping],
        verbose=0
    )

    return min(history.history['val_loss'])

# Start Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=OPTUNA_TRIALS)

# Visualization of optimization history
plot_optimization_history(study)
plot_contour(study, params=['latent_dim', 'intermediate_dim', 'learning_rate'])
plot_slice(study)

# Retrieve the best hyperparameters
best_params = study.best_trial.params
vae, encoder, decoder = build_vae(
    best_params['latent_dim'],
    best_params['intermediate_dim'],
    best_params['learning_rate']
)

# Train the VAE with the best hyperparameters
vae.fit(
    x_train, x_train,
    epochs=EPOCHS,
    batch_size=best_params['batch_size'],
    validation_data=(x_test, x_test)
)

# Functions to save plots
def save_latent_space_plot(encoder, data, labels, filename):
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.savefig(filename)
    plt.close()

def save_generated_images_plot(decoder, latent_dim, filename, grid_size=15, figure_size=28):
    figure = np.zeros((figure_size * grid_size, figure_size * grid_size))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, grid_size))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, grid_size))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi] + [0.0] * (latent_dim - 2)])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(figure_size, figure_size)
            figure[i * figure_size: (i + 1) * figure_size, j * figure_size: (j + 1) * figure_size] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

# Save plots for latent space and generated images
latent_space_filename = os.path.join(LATENT_SPACE_PLOTS_DIR, 'latent_space.png')
generated_images_filename = os.path.join(GENERATED_IMAGES_PLOTS_DIR, 'generated_images.png')
save_latent_space_plot(encoder, x_test, y_test, latent_space_filename)
save_generated_images_plot(decoder, best_params['latent_dim'], generated_images_filename)

