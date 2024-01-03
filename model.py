import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(original_dim, latent_dim, intermediate_dim, learning_rate):
    inputs = layers.Input(shape=(original_dim,), name='encoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs) * original_dim
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1) * -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    vae.compile(optimizer=optimizer)
    return vae, encoder, decoder
