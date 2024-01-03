import os
from .config import LATENT_SPACE_PLOTS_DIR, GENERATED_IMAGES_PLOTS_DIR, EPOCHS
from .data import load_data
from .training import train_and_optimize
from .plotting import save_latent_space_plot, save_generated_images_plot

def main():
    x_train, x_test, y_test = load_data()
    original_dim = x_train.shape[1]

    study = train_and_optimize(x_train, x_test, original_dim)

    best_params = study.best_trial.params
    vae, encoder, decoder = build_vae(
        original_dim,
        best_params['latent_dim'],
        best_params['intermediate_dim'],
        best_params['learning_rate']
    )

    vae.fit(
        x_train, x_train,
        epochs=EPOCHS,
        batch_size=best_params['batch_size'],
        validation_data=(x_test, x_test)
    )

    latent_space_filename = os.path.join(LATENT_SPACE_PLOTS_DIR, 'latent_space.png')
    generated_images_filename = os.path.join(GENERATED_IMAGES_PLOTS_DIR, 'generated_images.png')
    save_latent_space_plot(encoder, x_test, y_test, latent_space_filename)
    save_generated_images_plot(decoder, best_params['latent_dim'], generated_images_filename)

if __name__ == "__main__":
    main()
