#!/usr/bin/env python3
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam


def prep_pc():
    # Quiet TensorFlow logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


class TimeStopping(Callback):
    """Stop training when exceeding a time limit (in seconds)."""

    def __init__(self, max_seconds=300):
        super().__init__()
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.max_seconds:
            print(
                f"\n⏱️ Stopping training - time limit of {self.max_seconds} seconds reached."
            )
            self.model.stop_training = True


def eval_classification(model, X, y, name="model", labels=None, plot=False):
    """Evaluate and optionally plot a confusion matrix."""
    preds = np.rint(model.predict(X))
    print(f"\n=== {name} ===")
    print(classification_report(y, preds, target_names=labels))

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    score_df = pd.DataFrame(
        index=[name],
        data={
            "Accuracy": [accuracy_score(y, preds)],
            "Precision": [precision_score(y, preds)],
            "Recall": [recall_score(y, preds)],
            "F1 Score": [f1_score(y, preds)],
            "FPR": [fp / (fp + tn) if (fp + tn) > 0 else np.nan],
            "FNR": [fn / (fn + tp) if (fn + tp) > 0 else np.nan],
        },
    )

    if plot:
        ConfusionMatrixDisplay.from_predictions(y, preds, display_labels=labels)
        plt.title(name)
        plt.show()

    return score_df


def plot_history(history, plot=False):
    """Plot training & validation metrics from a Keras History."""
    if not plot:
        return
    metrics = [m for m in history.history.keys() if not m.startswith("val_")]
    for metric in metrics:
        plt.plot(history.history[metric], label=metric)
        val_metric = f"val_{metric}"
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=val_metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    prep_pc()

    # --- Load Data ---
    df_train = pd.read_csv("Split_Data/Model_Ready/train.csv")
    X_train = df_train.drop(columns="label")
    y_train = df_train["label"]
    df_val = pd.read_csv("Split_Data/Model_Ready/val.csv")
    X_val = df_val.drop(columns="label")
    y_val = df_val["label"]

    # --- Hyperparameter Space ---
    layer_configs = [[64], [128, 64], [256, 128, 64]]
    epochs_list = [50, 100]
    min_deltas = [0.0, 0.001]
    patiences = [1, 3, 5]

    best_f1 = -1
    best_model = None
    best_history = None
    best_config = None
    all_scores = []
    labels = ["No Phishing Email", "Phishing Email"]

    # --- Grid Search ---
    for layers in layer_configs:
        for epochs in epochs_list:
            for min_delta in min_deltas:
                for patience in patiences:
                    print(
                        f"\nConfig: layers={layers}, epochs={epochs}, min_delta={min_delta}, patience={patience}"
                    )

                    model = Sequential()
                    model.add(
                        Dense(
                            layers[0],
                            activation="relu",
                            input_shape=(X_train.shape[1],),
                        )
                    )
                    for units in layers[1:]:
                        model.add(Dense(units, activation="relu"))
                    model.add(Dense(1, activation="sigmoid"))

                    model.compile(
                        optimizer=Adam(learning_rate=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )

                    early_stopping = EarlyStopping(
                        monitor="val_loss",
                        min_delta=min_delta,
                        patience=patience,
                        restore_best_weights=True,
                        verbose=1,
                        mode="min",
                    )
                    time_stopping = TimeStopping(max_seconds=300)

                    history = model.fit(
                        X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=32,
                        callbacks=[early_stopping, time_stopping],
                        verbose=2,
                    )

                    # Collect metrics without plotting
                    plot_history(history, plot=False)
                    train_scores = eval_classification(
                        model,
                        X_train,
                        y_train,
                        name=f"train {layers}-{epochs}-p{patience}-d{min_delta}",
                        labels=labels,
                        plot=False,
                    )
                    val_scores = eval_classification(
                        model,
                        X_val,
                        y_val,
                        name=f"val {layers}-{epochs}-p{patience}-d{min_delta}",
                        labels=labels,
                        plot=False,
                    )

                    all_scores.append(pd.concat([train_scores, val_scores]))

                    val_f1 = val_scores.iloc[0]["F1 Score"]
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_model = model
                        best_history = history
                        best_config = {
                            "layers": layers,
                            "epochs": epochs,
                            "min_delta": min_delta,
                            "patience": patience,
                            "val_f1_score": val_f1,
                        }

    # --- Report & Save ---
    print("\n🏆 Best Model Configuration:")
    for k, v in best_config.items():
        print(f"{k}: {v}")

    best_model.save("best_mlp_model.h5")
    print("\n📁 Best model saved as 'best_mlp_model.h5'")

    all_results_df = pd.concat(all_scores)
    all_results_df.to_csv("mlp_model_results.csv", index=True)
    print("Results saved to mlp_model_results.csv")

    # --- Final Plots for Best Model ---
    print("\n📊 Training History of Best Model:")
    plot_history(best_history, plot=True)

    print("\n🧪 Confusion Matrix & Metrics on Best Validation Set:")
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
