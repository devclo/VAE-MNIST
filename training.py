import optuna
from .model import build_vae
from .config import EPOCHS

def objective(trial, x_train, x_test, original_dim):
    latent_dim = trial.suggest_int('latent_dim', 2, 20)
    intermediate_dim = trial.suggest_int('intermediate_dim', 128, 1024)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    vae, _, _ = build_vae(original_dim, latent_dim, intermediate_dim, learning_rate)
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

def train_and_optimize(x_train, x_test, original_dim):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, x_train, x_test, original_dim), n_trials=OPTUNA_TRIALS)
    return study
