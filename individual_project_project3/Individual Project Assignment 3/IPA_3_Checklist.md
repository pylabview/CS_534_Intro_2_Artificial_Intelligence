# CS 534 – Individual Assignment 3  
## **End-to-End To-Do List (with latest data-quality notes)**

> **Deliverable recap**  
> *One ZIP* containing:  
> 1. `MachineFailurePredictorPipeline.py` (entry-point `main()`)  
> 2. **Raw CSV only** (`ai4i2020.csv`)  
> 3. A PDF holding **Table 1 (CV results)** and **Table 2 (test-set results)**  

---

### 📂 0  Convert & Inspect Raw Data
| ✔︎    | Task                                  | Details                                                      |
| ---- | ------------------------------------- | ------------------------------------------------------------ |
| ☐    | **0.1  Convert workbook → CSV**       | Save `ai4i2020.xlsx` as `ai4i2020.csv`; commit the CSV to the repo (not the XLSX). |
| ☐    | **0.2  Sanity check dataset**         | Print to console:<br>• shape → `10 000 × 14`<br>• zero missing values<br>• `Machine failure` positives = **339** (3.39 %) |
| ☐    | **0.3  Log correlations & red flags** | High collinearity: `Air` ↔ `Process` T (+0.88) and `Torque` ↔ `Speed` (–0.88). Note in code comments. |

---

### 🏗️ 1  Project Skeleton
| ✔︎    | Task                        | Details                                                      |
| ---- | --------------------------- | ------------------------------------------------------------ |
| ☐    | **1.1  Create repo & venv** | Use PyCharm (or VS Code) with a clean virtual-env.           |
| ☐    | **1.2  Stub pipeline file** | Functions: `load_data()`, `preprocess()`, `balance()`, `split_data()`, `tune_models()`, `evaluate()`, `main()`. |

---

### 🧹 2  Load & Initial Printout
| ✔︎    | Task                   | Console requirements (per rubric)             |
| ---- | ---------------------- | --------------------------------------------- |
| ☐    | **2.1  Read CSV**      | Print first 5 rows of the seven kept columns. |
| ☐    | **2.2  Describe data** | Print `.info()` and class distribution.       |

---

### 🛠️ 3  Feature Engineering & Pre-processing
| ✔︎    | Task                                  | Details                                                      |
| ---- | ------------------------------------- | ------------------------------------------------------------ |
| ☐    | **3.1  Drop IDs**                     | Remove `UDI` and `Product ID`.                               |
| ☐    | **3.2  One-hot encode `Type`**        | 3 dummies → `Type_L`, `Type_M`, `Type_H`.                    |
| ☐    | **3.3  Optional engineered features** | • `ΔT = ProcessT − AirT` (HDF rule)<br>• `Power = Torque × RotSpeed` (PWF rule)<br>Then **decide** whether to drop one of the raw temps to curb multicollinearity. |
| ☐    | **3.4  Scale numeric cols**           | Use `StandardScaler` inside a `ColumnTransformer`.           |
| ☐    | **3.5  Print transformed df shape**   | Required console output.                                     |

---

### ⚖️ 4  Class Re-balancing
| ✔︎    | Task                                              | Details                                        |
| ---- | ------------------------------------------------- | ---------------------------------------------- |
| ☐    | **4.1  Random under-sample majority**             | Goal: **339 failure + 339 normal = 678 rows**. |
| ☐    | **4.2  Stratify by `Machine failure` (+ `Type`)** | Ensures all `Type` tiers survive in folds.     |
| ☐    | **4.3  Print new class counts**                   | Confirm balance.                               |

---

### 🔀 5  Train/Test Split
| ✔︎    | Task                          | Details                                             |
| ---- | ----------------------------- | --------------------------------------------------- |
| ☐    | **5.1  80 / 20 split**        | After balancing; use `test_size=0.2`, `stratify=y`. |
| ☐    | **5.2  Hold test set unseen** | No peeking until Step 7.                            |

---

### 🧪 6  Model Tuning (5-fold CV, metric = MCC) — **25 pts**
| ✔︎    | Model                      | Hyper-parameter grid / notes                                 |
| ---- | -------------------------- | ------------------------------------------------------------ |
| ☐    | **Logistic Regression**    | `C`, `penalty` (`l1`, `l2`)                                  |
| ☐    | **SVM (RBF)**              | `C`, `gamma`                                                 |
| ☐    | **KNN**                    | `n_neighbors`, `weights`, `metric`                           |
| ☐    | **Decision Tree**          | `max_depth`, `min_samples_split`, `class_weight`             |
| ☐    | **MLPClassifier**          | `hidden_layer_sizes`, `alpha`, `learning_rate_init`          |
| ☐    | **6.x  Record CV results** | Build **Table 1** (model, best params, mean MCC ± std). Print to console. |

---

### 🏆 7  Final Evaluation on Test Set
| ✔︎    | Task                                    | Details                                          |
| ---- | --------------------------------------- | ------------------------------------------------ |
| ☐    | **7.1  Refit each model on full train** | Using its CV-best hyper-parameters.              |
| ☐    | **7.2  Compute test metrics**           | MCC (primary), Accuracy, Precision, Recall, F1.  |
| ☐    | **7.3  Populate Table 2**               | Print and save to PDF.                           |
| ☐    | **7.4  Select winner**                  | Highest test MCC; note tie-break rule in report. |

---

### 🖨️ 8  Console & PDF Outputs
| ✔︎    | Required printouts             | Where         |
| ---- | ------------------------------ | ------------- |
| ☐    | Raw data head & info (Step 2)  | Console       |
| ☐    | Balanced class counts (Step 4) | Console       |
| ☐    | Table 1 – CV scores            | Console & PDF |
| ☐    | Table 2 – Test scores          | Console & PDF |

---

### 📦 9  Package & Submit
| ✔︎    | Task                            | Details                                                      |
| ---- | ------------------------------- | ------------------------------------------------------------ |
| ☐    | **9.1  Generate PDF**           | `matplotlib.table`, `pandas.to_latex` → `pdfkit`, or similar. |
| ☐    | **9.2  Zip**                    | `ai4i2020.csv` + `MachineFailurePredictorPipeline.py` + PDF. |
| ☐    | **9.3  Verify reproducibility** | `python MachineFailurePredictorPipeline.py` must reproduce all console prints and regenerate the PDF. |

---

### 🧹 10  Polish & Document
| ✔︎    | Task                          | Details                                                      |
| ---- | ----------------------------- | ------------------------------------------------------------ |
| ☐    | **10.1  PEP-8 & doc-strings** | Each function with type hints, concise description.          |
| ☐    | **10.2  Logging**             | Use `logging` module (`INFO` level) for dataset stats and model progress. |
| ☐    | **10.3  README (optional)**   | Short run instructions (helps grader).                       |

---

### ⭐ 11  (Extra Credit) Chase Bonus
| ✔︎    | Task                                           | Details                                 |
| ---- | ---------------------------------------------- | --------------------------------------- |
| ☐    | **11.1  Alternative resampling or ensembling** | E.g., SMOTE + bagging; submit best MCC. |
| ☐    | **11.2  Aim top-2 MCC**                        | Grants +10 pts per rubric.              |

---

## **Data-Quality Notes Recap**
1. **No missing values** — skip imputation.  
2. **Strong collinearity** — consider engineered features (`ΔT`, `Power`) *or* drop redundant originals.  
3. **Severe class imbalance** — under-sample majority class to 1:1 ratio before split.  
4. **Minor `Type_H` sparsity** — maintain its presence via stratification.

> *Include these checks in pipeline logging so graders see you validated the dataset.*