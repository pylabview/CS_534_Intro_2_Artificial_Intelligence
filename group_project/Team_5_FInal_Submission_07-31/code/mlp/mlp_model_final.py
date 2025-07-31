#!/usr/bin/env python3
"""
mlp_pipeline.py

A script to train and evaluate multiple MLP configurations for binary classification.
Implements:
 - GPU setup and TensorFlow logging suppression
 - Timing decorator
 - Custom TimeStopping callback
 - Modular data loading, model building, training, evaluation
 - Configurable paths and verbosity via argparse
"""

import argparse
import functools
import itertools
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate MLP models with grid search"
    )
    parser.add_argument(
        "--train-csv", type=Path, required=True, help="Path to training CSV file"
    )
    parser.add_argument(
        "--val-csv", type=Path, required=True, help="Path to validation CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save models and results",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def timeit(func):
    """Decorator to measure execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info("⏱️ Total execution time: %.2fs", elapsed)
        return result

    return wrapper


class TimeStopping(Callback):
    """Stop training when exceeding a time limit (in seconds)."""

    def __init__(self, max_seconds: float = 300.0, verbose: bool = True):
        super().__init__()
        self.max_seconds = max_seconds
        self.verbose = verbose
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch: int, logs=None):
        if time.time() - self.start_time > self.max_seconds:
            if self.verbose:
                logging.getLogger(__name__).info(
                    "⏱️ Stopping training - time limit of %ss reached.",
                    self.max_seconds,
                )
            self.model.stop_training = True


def prep_pc() -> None:
    """Configure TensorFlow logging and GPU memory growth."""
    # Quiet TensorFlow logging
    tf.get_logger().setLevel("ERROR")
    # Allow GPU memory growth if GPUs are available
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_data(csv_path: Path) -> (np.ndarray, np.ndarray):
    df = pd.read_csv(csv_path)
    X = df.drop(columns="label").values
    y = df["label"].values
    return X, y


def eval_classification(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    name: str = "model",
    labels=None,
    plot: bool = False,
) -> pd.DataFrame:
    """Evaluate model and return metrics DataFrame; optionally plot confusion matrix."""
    logger = logging.getLogger(__name__)
    preds = np.rint(model.predict(X))
    logger.info("=== %s ===", name)
    logger.info("%s", classification_report(y, preds, target_names=labels))

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    data = {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1 Score": f1_score(y, preds),
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "FNR": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
    }
    df = pd.DataFrame(index=[name], data=data)

    if plot:
        ConfusionMatrixDisplay.from_predictions(y, preds, display_labels=labels)
        plt.title(name)
        plt.show()

    return df


def plot_history(history: tf.keras.callbacks.History, plot: bool = False) -> None:
    """Plot training & validation metrics from a Keras History object."""
    if not plot:
        return
    metrics = [m for m in history.history.keys() if not m.startswith("val_")]
    epochs = range(1, len(history.history[metrics[0]]) + 1)

    for metric in metrics:
        plt.plot(epochs, history.history[metric], label=metric)
        val_metric = f"val_{metric}"
        if val_metric in history.history:
            plt.plot(epochs, history.history[val_metric], label=val_metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(metric)
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)
        plt.show()


def build_model(
    layers: list[int],
    input_dim: int,
    dropout_rate: float,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Construct and compile a Sequential MLP model."""
    model = Sequential()
    # Input layer
    model.add(Dense(layers[0], activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for units in layers[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


@timeit
def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prep_pc()

    # Load datasets
    X_train, y_train = load_data(args.train_csv)
    X_val, y_val = load_data(args.val_csv)
    input_dim = X_train.shape[1]

    # Hyperparameter grid
    param_grid = {
        "layers": [[32], [64, 64], [64, 128, 64], [64, 256, 64]],
        "epochs": [50, 100, 150, 250],
        "min_delta": [0.0, 0.001],
        "patience": [3, 5, 7, 8],
        "dropout_rate": [0.3, 0.5],
    }

    best_f1 = -1.0
    best_config = {}
    all_results = []
    labels = ["No Phishing Email", "Phishing Email"]

    for layers, epochs, min_delta, patience, dropout_rate in itertools.product(
        *param_grid.values()
    ):
        config = {
            "layers": layers,
            "epochs": epochs,
            "min_delta": min_delta,
            "patience": patience,
            "dropout_rate": dropout_rate,
        }
        logger.info("Config: %s", config)

        model = build_model(layers, input_dim, dropout_rate)
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                min_delta=min_delta,
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            ),
            TimeStopping(max_seconds=300.0, verbose=False),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        train_df = eval_classification(
            model, X_train, y_train, name=f"train {layers}", labels=labels
        )
        val_df = eval_classification(
            model, X_val, y_val, name=f"val {layers}", labels=labels
        )

        all_results.append(pd.concat([train_df, val_df]))
        val_f1 = val_df.loc[f"val {layers}", "F1 Score"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_config = {**config, "val_f1": best_f1}
            best_model = model
            best_history = history

    # Save best model
    model_path = args.output_dir / "best_mlp_model.h5"
    best_model.save(model_path)
    logger.info("Best model saved to %s", model_path)

    # Save all results
    results_df = pd.concat(all_results)
    results_path = args.output_dir / "mlp_model_results.csv"
    results_df.to_csv(results_path)
    logger.info("Results saved to %s", results_path)

    # Optional plots for best model
    logger.info("Plotting best model training history...")
    plot_history(best_history, plot=True)

    logger.info("Displaying confusion matrix for best model...")
    eval_classification(
        best_model,
        X_val,
        y_val,
        name="Best Model on Validation",
        labels=labels,
        plot=True,
    )


if __name__ == "__main__":
    main()
