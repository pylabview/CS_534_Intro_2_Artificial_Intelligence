"""Machine‑Failure Predictor Pipeline

Usage::
    python machine_failure_pipeline.py \
        --csv ai4i2020.csv \
        --report out.md \
        --seed 42

Features
--------
* Dynamic undersampling (majority → minority count).
* Hyper‑parameter search (Grid/Random) for 5 models.
* Early‑stopping MLP, suppression of convergence warnings.
* Console tables and Markdown report (timestamped if not provided).
* CLI flags: `--csv`, `--report`, `--seed`.
* Clean exit codes and logging.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import parallel_backend

# ────────────────────────────── CLI ──────────────────────────────── #
parser = argparse.ArgumentParser(
    description="Train & compare ML models for machine‑failure prediction; emit a Markdown report."
)
parser.add_argument(
    "--csv", default="ai4i2020.csv", help="Path to input CSV (default: ai4i2020.csv)"
)
parser.add_argument(
    "--report", help="Output Markdown report filename (default: timestamped)"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

# ───────────────────── global config / constants ─────────────────── #
FEATURE_COLS: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"

set_config(transform_output="pandas")
logging.basicConfig(level="INFO", format="%(message)s")

# ─────────────────────────── helpers ─────────────────────────────── #


def suppress_convergence_warnings() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def undersample(df: pd.DataFrame, seed: int):
    """Undersample majority class to minority size; fallback to pandas if imblearn missing."""
    try:
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=seed)
        return rus.fit_resample(df[FEATURE_COLS], df[TARGET])
    except ImportError:
        min_n = df[TARGET].value_counts().min()
        df_bal = (
            df.groupby(TARGET, group_keys=False)
            .apply(lambda d: d.sample(min_n, random_state=seed))
            .sample(frac=1, random_state=seed)
        )
        return df_bal[FEATURE_COLS], df_bal[TARGET]


def split_stats(y_tr: pd.Series, y_te: pd.Series, total: int) -> None:
    logging.info("Total rows : %d", total)
    logging.info("Train rows : %d  (%.2f%%)", len(y_tr), len(y_tr) / total * 100)
    logging.info("Test  rows : %d   (%.2f%%)\n", len(y_te), len(y_te) / total * 100)
    logging.info("Class ratio train:\n%s", y_tr.value_counts(normalize=True))
    logging.info("\nClass ratio test:\n%s\n", y_te.value_counts(normalize=True))


@contextmanager
def wide_markdown():
    with pd.option_context("display.width", 0, "display.max_colwidth", None):
        yield


def make_search(cfg: dict, scorer, seed: int):
    """Factory returning a configured GridSearchCV or RandomizedSearchCV.
    RandomizedSearchCV gets a `random_state`; GridSearchCV does not accept it.
    """
    common = dict(scoring=scorer, cv=5, n_jobs=-1, verbose=0)
    if cfg["type"] == "random":
        return RandomizedSearchCV(
            cfg["pipeline"],
            cfg["params"],
            n_iter=cfg["n_iter"],
            random_state=seed,
            **common,
        )
    # GridSearchCV has no random_state kwarg
    return GridSearchCV(cfg["pipeline"], cfg["params"], **common)


# ───────────────────────────── main ──────────────────────────────── #


def main() -> None:
    suppress_convergence_warnings()

    # Load data
    try:
        df = pd.read_csv(Path(args.csv))
    except Exception as exc:
        logging.error("❌  Cannot read %s: %s", args.csv, exc)
        sys.exit(1)

    # Balance classes
    X_bal, y_bal = undersample(df, args.seed)
    logging.info("Balanced class counts:\n%s\n", y_bal.value_counts())

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=args.seed
    )
    split_stats(y_tr, y_te, len(X_bal))

    # Preprocessing & scorer
    preproc = ColumnTransformer([("scale", StandardScaler(), FEATURE_COLS)])
    mcc = make_scorer(matthews_corrcoef)

    # Search configs
    search_cfgs = {
        "MLP": {
            "type": "random",
            "n_iter": 20,
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    (
                        "clf",
                        MLPClassifier(
                            max_iter=400,
                            early_stopping=True,
                            n_iter_no_change=10,
                            random_state=args.seed,
                        ),
                    ),
                ]
            ),
            "params": {
                "clf__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "clf__activation": ["relu", "tanh"],
                "clf__learning_rate": ["constant", "adaptive"],
                "clf__alpha": [1e-5, 1e-4, 1e-3],
            },
        },
        "SVM": {
            "type": "grid",
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    ("clf", SVC(probability=True, random_state=args.seed)),
                ]
            ),
            "params": {
                "clf__C": [0.1, 1, 10, 100],
                "clf__kernel": ["linear", "rbf"],
                "clf__gamma": ["scale", "auto"],
            },
        },
        "KNN": {
            "type": "grid",
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            "params": {
                "clf__n_neighbors": [3, 5, 7, 9],
                "clf__p": [1, 2],
                "clf__algorithm": ["auto", "ball_tree"],
            },
        },
        "DecisionTree": {
            "type": "grid",
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    ("clf", DecisionTreeClassifier(random_state=args.seed)),
                ]
            ),
            "params": {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__ccp_alpha": [0.0, 0.01],
            },
        },
        "LogisticRegression": {
            "type": "grid",
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    (
                        "clf",
                        LogisticRegression(solver="liblinear", random_state=args.seed),
                    ),
                ]
            ),
            "params": {
                "clf__penalty": ["l2", "l1"],
                "clf__C": [0.01, 0.1, 1, 10],
            },
        },
    }

    best_models, cv_rows = {}, []

    with parallel_backend("threading"):
        for name, cfg in search_cfgs.items():
            logging.info("🔍  Tuning %s …", name)
            searcher = make_search(cfg, mcc, args.seed)
            searcher.fit(X_tr, y_tr)
            best_models[name] = searcher.best_estimator_
            cv_rows.append(
                {
                    "Model": name,
                    "Best Params": str(searcher.best_params_),
                    "CV MCC": searcher.best_score_,
                }
            )

    tbl_cv = pd.DataFrame(cv_rows).sort_values("CV MCC", ascending=False)

    # Test evaluation
    test_rows = []
    for name, est in best_models.items():
        test_rows.append(
            {
                "Model": name,
                "Best Params": str(est.get_params(deep=False)),
                "Test MCC": matthews_corrcoef(y_te, est.predict(X_te)),
            }
        )

    tbl_test = pd.DataFrame(test_rows).sort_values("Test MCC", ascending=False)
    champion = tbl_test.iloc[0]["Model"]

    # Markdown report
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report or f"model_comparison_{ts}.md"

    with open(report_path, "w", encoding="utf-8") as md, wide_markdown():
        md.write("# Machine‑Failure Model Comparison\n\n")
        md.write("## Table 1 – 5‑fold CV Results\n\n")
        md.write(tbl_cv.to_markdown(index=False))
        md.write("\n\n## Table 2 – 20 % Hold‑out Test Results\n\n")
        md.write(tbl_test.to_markdown(index=False))
        md.write("\n\n### Conclusion\n\n")
        md.write(
            f"The **{champion}** model achieved the highest MCC on the test set and is selected for deployment.\n"
        )

    with wide_markdown():
        print("\n### CV Results\n", tbl_cv.to_markdown(index=False))
        print("\n### Test Results\n", tbl_test.to_markdown(index=False))
        print(f"\n✅  Report saved to '{report_path}'")

    sys.exit(0)


# ──────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()
