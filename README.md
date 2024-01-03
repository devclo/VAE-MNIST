# Variational Autoencoder (VAE) for MNIST #

## Overview ##

This project contains a TensorFlow implementation of a Variational Autoencoder (VAE), designed to work with the MNIST dataset. The VAE model is capable of generating new images that resemble the handwritten digits from the MNIST dataset by learning a latent representation of the input data.

## Features ##

* Load and preprocess the MNIST dataset.
* Build and train a VAE model with TensorFlow.
* Visualize the latent space.
* Generate new images from the latent space.
* Save visualizations and generated images as plots.
* Create GIFs demonstrating the evolution of the latent space and generated images over training epochs.


## Dependencies ##

* Python 3.x
* TensorFlow 2.x
* NumPy
* Matplotlib
* SciPy
* ImageIO

## Installation ##

To set up the environment for this project, follow these steps:

1. Ensure that you have Python 3 installed on your system.
2. Install the required Python packages using pip:

```bash
pip install numpy matplotlib scipy tensorflow imageio
```
3. Clone this repository or download the script to your local machine.


## Usage ##

To run the script, simply execute it with Python:

```bash
python vae_mnist.py
```
By default, the script will train the VAE model for 100 epochs, save the latent space and generated images every 10 epochs, and create GIFs to visualize the training progression.

## Output ##

* The script will create two directories: latent_space_plots and generated_images_plots to store the epoch-wise visualizations.
Two GIFs will be created: latent_space_evolution.gif and generated_images_evolution.gif, showing the evolution of the VAE outputs over the training period.
Customization

* You can adjust the number of epochs, batch size, or learning rate by modifying the corresponding variables in the script.
To change the frequency of saving plots, modify the condition if epoch % 10 == 0: to a different interval.
Hyperparameters such as intermediate_dim and latent_dim can be tuned for optimization.

## Notes ##

The model's performance and quality of generated images can vary based on the chosen hyperparameters.
The script includes a custom training loop that can be further customized to include callbacks or additional metrics.



