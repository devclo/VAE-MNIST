import matplotlib.pyplot as plt
from scipy.stats import norm
from .config import LATENT_SPACE_PLOTS_DIR, GENERATED_IMAGES_PLOTS_DIR

def save_latent_space_plot(encoder, x_test, y_test, filename):
    z_mean, _, _ = encoder.predict(x_test)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='viridis', alpha=0.7)
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
