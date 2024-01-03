!pip install tensorflow numpy matplotlib imageio optuna plotly
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

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

original_dim = x_train.shape[1]

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(latent_dim, intermediate_dim, learning_rate):
    # Encoder
    inputs = layers.Input(shape=(original_dim,), name='encoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)

    # VAE model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(latent_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # VAE loss
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Compile VAE
    optimizer = Adam(learning_rate=learning_rate)
    vae.compile(optimizer=optimizer)
    return vae, encoder, decoder
def objective(trial):
    # Hyperparameters to be optimized
    latent_dim = trial.suggest_int('latent_dim', 2, 20)
    intermediate_dim = trial.suggest_int('intermediate_dim', 128, 1024)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Build VAE model
    vae, _, _ = build_vae(latent_dim, intermediate_dim, learning_rate)

    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Training the model
    history = vae.fit(
        x_train, x_train,
        epochs=50,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[early_stopping],
        verbose=0
    )

    return min(history.history['val_loss'])
# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)

# Best hyperparameters
best_params = study.best_trial.params
vae, encoder, decoder = build_vae(best_params['latent_dim'], best_params['intermediate_dim'], best_params['learning_rate'])

# Train the model with the best parameters
vae.fit(
    x_train, x_train,
    epochs=100,
    batch_size=best_params['batch_size'],
    validation_data=(x_test, x_test)
)

# Directory to save the plots
os.makedirs('/content/gdrive/MyDrive/NJIT/CS370/Final Exam/latent_space_plots', exist_ok=True)
os.makedirs('/content/gdrive/MyDrive/NJIT/CS370/Final Exam/generated_images_plots', exist_ok=True)

from optuna.visualization import plot_optimization_history
plot_optimization_history(study)


from optuna.visualization import plot_contour
plot_contour(study, params=['latent_dim', 'intermediate_dim', 'learning_rate'])


from optuna.visualization import plot_slice
plot_slice(study)

best_params = study.best_trial.params

print(best_params)  # Should output the best latent_dim, intermediate_dim, and learning_rate
# Rebuild the VAE model with the best hyperparameters (excluding batch_size)
vae, encoder, decoder = build_vae(
    best_params['latent_dim'],
    best_params['intermediate_dim'],
    best_params['learning_rate']
)

# Train the model with the best parameters, including the best batch_size
vae.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=best_params['batch_size'],  # Use the best batch_size from Optuna study
    validation_data=(x_test, x_test)
)
# Function to plot and save the latent space at a given epoch
def save_latent_space_plot(encoder, epoch, data, labels, figure_size=(12, 10), dpi=100):
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=figure_size, dpi=dpi)
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title(f'Latent Space Visualization at Epoch {epoch}')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    filename = f'/content/gdrive/MyDrive/NJIT/CS370/Final Exam/latent_space_plots/latent_space_epoch_{epoch}.png'
    plt.savefig(filename, bbox_inches='tight')  # Save the figure with tight bounding box
    plt.close()
    return filename

# Function to plot and save the generated images from the latent space at a given epoch
def save_generated_images_plot(decoder, epoch, latent_dim, grid_size=15, figure_size=28):
    figure = np.zeros((figure_size * grid_size, figure_size * grid_size))
    # Use the norm.ppf function to get more interesting points in the latent space
    grid_x = norm.ppf(np.linspace(0.05, 0.95, grid_size))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, grid_size))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # z_sample must have a shape of (1, latent_dim)
            z_sample = np.array([[xi, yi] + [0.0] * (latent_dim - 2)])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(figure_size, figure_size)
            figure[i * figure_size: (i + 1) * figure_size,
                   j * figure_size: (j + 1) * figure_size] = digit
    filename = f'/content/gdrive/MyDrive/NJIT/CS370/Final Exam/generated_images_plots/generated_images_epoch_{epoch}.png'
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    return filename

# Initialize the lists to store filenames
filenames = []
generated_images_filenames = []

for epoch in range(10000):
    if epoch % 10 == 0:
        latent_space_filename = save_latent_space_plot(encoder, epoch, x_test, y_test)
        filenames.append(latent_space_filename)
        # Pass the best_params['latent_dim'] to the function
        generated_image_filename = save_generated_images_plot(decoder, epoch, best_params['latent_dim'])
        generated_images_filenames.append(generated_image_filename)

# Creating GIFs
with imageio.get_writer('latent_space_evolution.gif', mode='I', loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

with imageio.get_writer('generated_images_evolution.gif', mode='I', loop=0) as writer:
    for filename in generated_images_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

