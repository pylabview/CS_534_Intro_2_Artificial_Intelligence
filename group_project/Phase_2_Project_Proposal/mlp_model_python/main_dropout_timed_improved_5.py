#!/usr/bin/env python3
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # TF imported at top to avoid repeated overhead
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import functools
import itertools
from pathlib import Path

# Reproducibility: set both NumPy and TensorFlow random seeds for consistent results
np.random.seed(42)
tf.random.set_seed(42)


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m {secs:.2f}s" if mins else f"{secs:.2f}s"
        print(f"\n🚀 Script completed in {time_str}!")
        return result

    return wrapper


class TimeStopping(Callback):
    def __init__(self, max_seconds=300):
        super().__init__()
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.max_seconds:
            print(f"\n⏱️ Stopping training after {self.max_seconds}s")
            self.model.stop_training = True


# Prediction caching: repeated model.predict calls can be slow on large datasets
def eval_classification(model, X, y, name, labels=None, save_path=None):
    preds = np.rint(model.predict(X))
    print(f"\n=== {name} ===")
    print(classification_report(y, preds, target_names=labels))
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    df = pd.DataFrame(
        {
            "Accuracy": accuracy_score(y, preds),
            "Precision": precision_score(y, preds),
            "Recall": recall_score(y, preds),
            "F1 Score": f1_score(y, preds),
        },
        index=[name],
    )
    if save_path:
        disp = ConfusionMatrixDisplay.from_predictions(y, preds, display_labels=labels)
        out = Path(save_path) / f"confusion_matrix_{name}.png"
        disp.figure_.savefig(out)
        plt.close(disp.figure_)
    return df


# Helper to save training history plots, including both training and validation metrics
def save_training_history(history, out_dir):
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(history.history.keys())
    epochs = range(1, len(history.history[metrics[0]]) + 1)
    for metric in metrics:
        plt.figure()
        plt.plot(epochs, history.history[metric], label=metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"Training History: {metric}")
        plt.xticks(list(epochs))
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"history_{metric}.png")
        plt.close()
    return metrics


@timeit
def main():
    # CSV loading wrapped in try/except for robust I/O
    try:
        df_train = pd.read_csv(Path("Split_Data/Model_Ready/train.csv"))
        df_val = pd.read_csv(Path("Split_Data/Model_Ready/val.csv"))
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    X_train, y_train = df_train.drop("label", axis=1), df_train["label"]
    X_val, y_val = df_val.drop("label", axis=1), df_val["label"]

    # Hyperparameters
    layer_configs = [[64], [64, 64], [64, 128, 64], [256, 128, 128, 256]]
    epochs_list = [50, 100, 200, 500]
    patiences = [3, 5, 7, 9]
    dropout_rates = [0.1, 0.2, 0.3]
    # l1_rates = [0.0, 1e-5, 1e-4, 1e-3]
    l1_rates = [0.0]
    l2_rates = [0.0, 1e-5, 1e-4, 1e-3]

    best_f1, best_model, best_history, best_config = -1, None, None, {}
    labels = ["No Phishing", "Phishing"]

    # Build and evaluate models over hyperparameter grid using itertools.product
    for layers, epochs, patience, dr, l1, l2 in itertools.product(
        layer_configs, epochs_list, patiences, dropout_rates, l1_rates, l2_rates
    ):
        tf.keras.backend.clear_session()
        regs = regularizers.l1_l2(l1=l1, l2=l2)
        model = Sequential(
            [
                Dense(
                    layers[0],
                    activation="relu",
                    kernel_regularizer=regs,
                    input_shape=(X_train.shape[1],),
                ),
                Dropout(dr),
                *[
                    layer
                    for units in layers[1:]
                    for layer in (
                        Dense(units, activation="relu", kernel_regularizer=regs),
                        Dropout(dr),
                    )
                ],
                Dense(1, activation="sigmoid", kernel_regularizer=regs),
            ]
        )
        model.compile(
            optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
                TimeStopping(300),
            ],
            verbose=0,
        )

        val_df = eval_classification(model, X_val, y_val, "val", labels)
        f1 = val_df.loc["val", "F1 Score"]
        if f1 > best_f1:
            best_f1, best_model, best_history = f1, model, history
            best_config = {
                "layers": layers,
                "epochs": epochs,
                "patience": patience,
                "dropout_rate": dr,
                "l1": l1,
                "l2": l2,
            }

    print("Best config:", best_config)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    best_model.save(results_dir / "best.h5")
    eval_classification(
        best_model, X_val, y_val, "best_val", labels, save_path=results_dir
    )

    # Save history plots
    save_training_history(best_history, results_dir)


if __name__ == "__main__":
    main()
