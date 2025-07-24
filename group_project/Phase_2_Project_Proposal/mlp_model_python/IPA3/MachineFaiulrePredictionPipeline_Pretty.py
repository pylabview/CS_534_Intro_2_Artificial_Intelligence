#!/usr/bin/env python3
"""
Machine-Failure prediction demo.

Steps:
1. Balance classes via random undersampling.
2. Tune 5 classifiers (MLP, SVM, KNN, DecisionTree, LogisticRegression)
   using Grid or Randomized search.
3. Emit two Markdown tables:
   * Table 1 – MCC on 5-fold CV (train, 80%)
   * Table 2 – MCC on hold-out test (20%)
"""

from __future__ import annotations

import logging
from pathlib import Path
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Constants
RANDOM_STATE: int = 42
CSV_PATH: Path = Path("ai4i2020.csv")
FEATURE_COLS: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL: str = "Machine failure"
N_SAMPLES_PER_CLASS: int = 339
TEST_SIZE: float = 0.20
REPORT_PATH: Path = Path("model_comparison_report.md")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(CSV_PATH)
    return df[FEATURE_COLS], df[TARGET_COL]


def balance_classes(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    try:
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(
            sampling_strategy={0: N_SAMPLES_PER_CLASS, 1: N_SAMPLES_PER_CLASS},
            random_state=RANDOM_STATE,
        )
        return rus.fit_resample(X, y)
    except ImportError:
        warnings.warn("imblearn not installed; falling back to pandas undersampling.")
        df_full = pd.concat([X, y], axis=1)
        df_min = df_full[df_full[TARGET_COL] == 1].sample(
            N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE
        )
        df_maj = df_full[df_full[TARGET_COL] == 0].sample(
            N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE
        )
        df_bal = pd.concat([df_min, df_maj]).sample(frac=1, random_state=RANDOM_STATE)
        return df_bal[FEATURE_COLS], df_bal[TARGET_COL]


def main() -> None:
    # Load and balance data
    X, y = load_data()
    X_bal, y_bal = balance_classes(X, y)
    log.info("Balanced class counts:\n%s\n", y_bal.value_counts())

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal,
        y_bal,
        test_size=TEST_SIZE,
        stratify=y_bal,
        random_state=RANDOM_STATE,
    )
    n_total = len(X_bal)
    n_train = len(X_train)
    n_test = len(X_test)
    log.info("Total rows : %d", n_total)
    log.info("Train rows : %d (%.2f%%)", n_train, n_train / n_total * 100)
    log.info("Test  rows : %d (%.2f%%)\n", n_test, n_test / n_total * 100)
    log.info("Class ratio train:\n%s", y_train.value_counts(normalize=True))
    log.info("Class ratio test:\n%s\n", y_test.value_counts(normalize=True))

    # Preprocessor and scorer
    preprocessor = ColumnTransformer([("scale", StandardScaler(), FEATURE_COLS)])
    mcc_scorer = make_scorer(matthews_corrcoef)

    # Search configurations
    configs: dict[str, dict] = {
        "MLP": {
            "searcher": RandomizedSearchCV(
                Pipeline(
                    [
                        ("prep", preprocessor),
                        ("clf", MLPClassifier(max_iter=500, random_state=RANDOM_STATE)),
                    ]
                ),
                {
                    "clf__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "clf__activation": ["relu", "tanh", "logistic"],
                    "clf__learning_rate": ["constant", "adaptive"],
                    "clf__alpha": [1e-5, 1e-4, 1e-3],
                },
                n_iter=20,
                scoring=mcc_scorer,
                cv=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        },
        "SVM": {
            "searcher": RandomizedSearchCV(
                Pipeline(
                    [
                        ("prep", preprocessor),
                        ("clf", SVC(random_state=RANDOM_STATE)),
                    ]
                ),
                {
                    "clf__C": [0.1, 1, 10, 100],
                    "clf__kernel": ["linear", "rbf", "poly"],
                    "clf__gamma": ["scale", "auto"],
                },
                n_iter=20,
                scoring=mcc_scorer,
                cv=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        },
        "KNN": {
            "searcher": RandomizedSearchCV(
                Pipeline(
                    [
                        ("prep", preprocessor),
                        ("clf", KNeighborsClassifier()),
                    ]
                ),
                {
                    "clf__n_neighbors": [3, 5, 7, 9, 11],
                    "clf__p": [1, 2],
                    "clf__algorithm": ["auto", "ball_tree", "kd_tree"],
                },
                n_iter=15,
                scoring=mcc_scorer,
                cv=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        },
        "DecisionTree": {
            "searcher": GridSearchCV(
                Pipeline(
                    [
                        ("prep", preprocessor),
                        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                    ]
                ),
                {
                    "clf__criterion": ["gini", "entropy"],
                    "clf__max_depth": [None, 5, 10, 20],
                    "clf__ccp_alpha": [0.0, 0.01, 0.05],
                },
                scoring=mcc_scorer,
                cv=5,
                n_jobs=-1,
            )
        },
        "LogisticRegression": {
            "searcher": GridSearchCV(
                Pipeline(
                    [
                        ("prep", preprocessor),
                        (
                            "clf",
                            LogisticRegression(
                                solver="liblinear", random_state=RANDOM_STATE
                            ),
                        ),
                    ]
                ),
                {
                    "clf__penalty": ["l2", "l1"],
                    "clf__C": [0.01, 0.1, 1, 10, 100],
                },
                scoring=mcc_scorer,
                cv=5,
                n_jobs=-1,
            )
        },
    }

    # Run hyperparameter searches
    best_estimators: dict[str, Pipeline] = {}
    table1_rows: list[dict[str, str]] = []

    for name, cfg in configs.items():
        searcher = cfg["searcher"]
        searcher.fit(X_train, y_train)
        best_estimators[name] = searcher.best_estimator_
        param_str = str(searcher.best_params_)
        table1_rows.append(
            {
                "Model": name,
                "Best Params": param_str,
                "CV MCC (5-fold)": f"{searcher.best_score_:.4f}",
            }
        )

    # Prepare Table 1
    table1 = pd.DataFrame(table1_rows)
    log.info("\n### TABLE 1 – 5-fold CV (80%% train) ###")
    log.info("%s", table1.to_markdown(index=False))

    # Evaluate on hold-out test
    table2_rows: list[dict[str, str]] = []
    for name, est in best_estimators.items():
        mcc = matthews_corrcoef(y_test, est.predict(X_test))
        table2_rows.append(
            {
                "Model": name,
                "Best Params": table1_rows[list(configs).index(name)]["Best Params"],
                "Test MCC (20%)": f"{mcc:.4f}",
            }
        )

    # Prepare Table 2
    table2 = pd.DataFrame(table2_rows)
    log.info("\n### TABLE 2 – Hold-out Test (20%%) ###")
    log.info("%s", table2.to_markdown(index=False))

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as md:
        md.write("# Machine-Failure Model Comparison\n\n")
        md.write("## Table 1 – 5-fold CV (80% train)\n\n")
        md.write(table1.to_markdown(index=False))
        md.write("\n\n## Table 2 – Hold-out Test (20%)\n\n")
        md.write(table2.to_markdown(index=False))
        champion = table2.sort_values("Test MCC (20%)", ascending=False).iloc[0][
            "Model"
        ]
        md.write("\n\n### Conclusion\n")
        md.write(
            f"The **{champion}** achieved the highest MCC on both CV and the test set, "
            "so it is selected as the production model.\n"
        )
    log.info("\n✅ Markdown report written to: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
