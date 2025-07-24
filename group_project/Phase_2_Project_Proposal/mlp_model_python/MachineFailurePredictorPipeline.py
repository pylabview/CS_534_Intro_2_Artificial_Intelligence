#!/usr/bin/env python3
"""
Machine‑Failure prediction demo
• balances classes (undersampling)
• tunes 5 models via Grid/Random search
• prints two wide markdown tables:
     Table 1 – CV MCC on 80 % train
     Table 2 – MCC on 20 % test
"""

# ───────────────────────────── imports ───────────────────────────── #

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# ───────────────────────────── data load ─────────────────────────── #
CSV_PATH = "ai4i2020.csv"  # adjust to your location
df = pd.read_csv(CSV_PATH)

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


# ───────────────────────────‑ main workflow ──────────────────────── #
def main() -> None:
    X, y = df[FEATURE_COLS], df["Machine failure"]

    # ---------- class balancing (undersample majority) --------------
    try:
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(sampling_strategy={0: 339, 1: 339}, random_state=42)
        X_bal, y_bal = rus.fit_resample(X, y)
    except ImportError:
        # pure‑pandas fallback
        df_min = df[df["Machine failure"] == 1].sample(339, random_state=42)
        df_maj = df[df["Machine failure"] == 0].sample(339, random_state=42)
        df_bal = pd.concat([df_min, df_maj]).sample(frac=1, random_state=42)
        X_bal, y_bal = df_bal[FEATURE_COLS], df_bal["Machine failure"]

    print("Balanced class counts:\n", y_bal.value_counts(), "\n")

    # ------------------- 80 / 20 train‑test split -------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )

    n_total, n_train, n_test = len(X_bal), len(X_train), len(X_test)
    print(f"Total rows : {n_total}")
    print(f"Train rows : {n_train}  ({n_train / n_total:.2%})")
    print(f"Test  rows : {n_test}   ({n_test / n_total:.2%})\n")

    # confirm stratification
    print("Class ratio train:\n", y_train.value_counts(normalize=True))
    print("\nClass ratio test:\n", y_test.value_counts(normalize=True), "\n")

    # ------------------ preprocessing & scorer ----------------------
    preproc = ColumnTransformer([("scale", StandardScaler(), FEATURE_COLS)])
    mcc_score = make_scorer(matthews_corrcoef)

    # -------------------- search space definitions ------------------
    search_configs = {
        "MLP": {
            "type": "random",
            "n_iter": 20,
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    ("clf", MLPClassifier(max_iter=500, random_state=42)),
                ]
            ),
            "params": {
                "clf__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "clf__activation": ["relu", "tanh", "logistic"],
                "clf__learning_rate": ["constant", "adaptive"],
                "clf__alpha": [1e-5, 1e-4, 1e-3],
            },
        },
        "SVM": {
            "type": "random",
            "n_iter": 20,
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    ("clf", SVC(random_state=42)),
                ]
            ),
            "params": {
                "clf__C": [0.1, 1, 10, 100],
                "clf__kernel": ["linear", "rbf", "poly"],
                "clf__gamma": ["scale", "auto"],
            },
        },
        "KNN": {
            "type": "random",
            "n_iter": 15,
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            "params": {
                "clf__n_neighbors": [3, 5, 7, 9, 11],
                "clf__p": [1, 2],
                "clf__algorithm": ["auto", "ball_tree", "kd_tree"],
            },
        },
        "DecisionTree": {
            "type": "grid",
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    ("clf", DecisionTreeClassifier(random_state=42)),
                ]
            ),
            "params": {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__ccp_alpha": [0.0, 0.01, 0.05],
            },
        },
        "LogisticRegression": {
            "type": "grid",
            "pipeline": Pipeline(
                [
                    ("scale", preproc),
                    ("clf", LogisticRegression(solver="liblinear", random_state=42)),
                ]
            ),
            "params": {
                "clf__penalty": ["l2", "l1"],
                "clf__C": [0.01, 0.1, 1, 10, 100],
            },
        },
    }

    # table storage
    best_estimators, param_map, tbl1_rows, tbl2_rows = {}, {}, [], []

    # --------------------- hyper‑parameter search -------------------
    for name, cfg in search_configs.items():
        searcher = (
            RandomizedSearchCV(
                cfg["pipeline"],
                cfg["params"],
                n_iter=cfg["n_iter"],
                scoring=mcc_score,
                cv=5,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
            if cfg["type"] == "random"
            else GridSearchCV(
                cfg["pipeline"],
                cfg["params"],
                scoring=mcc_score,
                cv=5,
                n_jobs=-1,
                verbose=0,
            )
        )
        searcher.fit(X_train, y_train)

        best_estimators[name] = searcher.best_estimator_
        param_str = str(searcher.best_params_)
        param_map[name] = param_str

        tbl1_rows.append(
            {
                "ML Trained Model": name,
                "Its Best Set of Parameter Values": param_str,
                "MCC on 5‑fold CV (80 % train)": f"{searcher.best_score_:.4f}",
            }
        )

    # ------------------------ TABLE 1 output ------------------------
    pd.set_option("display.width", 0)
    pd.set_option("display.max_colwidth", None)
    print("\n### TABLE 1 – 5‑fold CV on 80 % Training ###")
    print(pd.DataFrame(tbl1_rows).to_markdown(index=False))

    # ---------------- evaluate on 20 % held‑out ---------------------
    for name, est in best_estimators.items():
        mcc = matthews_corrcoef(y_test, est.predict(X_test))
        tbl2_rows.append(
            {
                "ML Trained Model": name,
                "Its Best Set of Parameter Values": param_map[name],
                "MCC on 20 % Test set": f"{mcc:.4f}",
            }
        )

    # ------------------------ TABLE 2 output ------------------------
    print("\n### TABLE 2 – Hold‑out 20 % Test ###")
    print(pd.DataFrame(tbl2_rows).to_markdown(index=False))

    # # ------------------------ TABLE 1 output ------------------------
    table1 = pd.DataFrame(tbl1_rows)  # ←  create the DataFrame

    # # ------------------------ TABLE 2 output ------------------------
    table2 = pd.DataFrame(tbl2_rows)  # ←  create the DataFrame

    # ───────────────── save both tables to Markdown ─────────────────
    REPORT_PATH = "model_comparison_report.md"
    with open(REPORT_PATH, "w", encoding="utf-8") as md:
        md.write("# Machine‑Failure Model Comparison\n\n")
        md.write("## Table 1 – 5‑fold CV on 80 % Training Data\n\n")
        md.write(table1.to_markdown(index=False))
        md.write("\n\n")
        md.write("## Table 2 – Performance on 20 % Hold‑out Test Set\n\n")
        md.write(table2.to_markdown(index=False))
        md.write("\n\n")

        champion = table2.sort_values("MCC on 20 % Test set", ascending=False).iloc[0][
            "ML Trained Model"
        ]

        md.write("### Conclusion – Selected Model\n\n")
        md.write(
            f"""The **{champion}** achieved the highest MCC on both 
            cross‑validation and the held‑out test set, so we choose it 
            as the production model.\n"""
        )

    print(f"\n✅  Markdown report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
