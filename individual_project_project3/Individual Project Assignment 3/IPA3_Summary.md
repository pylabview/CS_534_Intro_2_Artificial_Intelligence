# CS 534 Individual Project Assignment 3: Summary

> **Explainable Artificial Intelligence for Predictive Maintenance Applications**  
> Stephan Matzka et al. :contentReference[oaicite:0]{index=0}

---

## 1. Purpose and Motivation  
Predictive‐maintenance (PdM) systems detect incipient machine failures but often rely on “black-box” models that operators distrust. This paper:  
1. Provides a **public, realistic synthetic PdM dataset** (10 000 records).  
2. Compares two **explainability strategies**—shallow decision-tree surrogates vs. a model-agnostic feature-deviation interface—to turn predictions into human-understandable explanations. :contentReference[oaicite:1]{index=1}

---

## 2. Data Set Construction  
- **Size & Format:** 10 000 rows × (6 features + 1 label).  
- **Features:**  
  - `productID` (quality tier L/M/H)  
  - Air temperature (K)  
  - Process temperature (K)  
  - Rotational speed (rpm)  
  - Torque (Nm)  
  - Tool-wear (min)  
- **Failure Modes (independent rules):**  
  1. **TWF** – tool-wear ≥ 200–240 min  
  2. **HDF** – ΔT < 8.6 K **and** speed < 1380 rpm  
  3. **PWF** – power (torque × speed) outside [3500, 9000] W  
  4. **OSF** – torque × wear > tier-specific limit (11 000–13 000 min·Nm)  
  5. **RNF** – 0.1 % random failures  
- **Imbalance:** 339 failures (3.39 %) :contentReference[oaicite:2]{index=2}

---

## 3. Baseline (“Black-Box”) Classifier  
- **Model:** Bagged-trees ensemble with a 30× higher cost on false negatives.  
- **Performance (5-fold CV):**  
  - **Recall (failures):** 86.7 %  
  - **Specificity (normal):** 98.7 %  
- **Feature Importance:** Torque & rotational speed dominate; temperatures and `productID` contribute little. :contentReference[oaicite:3]{index=3}

---

## 4. Explanation Approaches  

| Approach                        | Mechanism                                                    | Pros                                               | Cons                                                   |
| ------------------------------- | ------------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------ |
| **Shallow Decision Trees**      | 15 trees (≤ 4 nodes) trained on feature-subsets; for each prediction, use the tree with highest global importance | • Clear, rule-based thresholds<br>• Intuitive      | • No explanation if no tree matches (4/20 cases)       |
| **Feature-Deviation Interface** | Z-score each feature for the queried sample; report the two largest absolute deviations | • Always yields an explanation<br>• Model-agnostic | • Often only “partially useful” (flags one irrelevant) |

*Evaluation on 20 stratified points:*  
- **Decision Trees:** 9 very useful; 2 partially; 4 none; 5 not-applicable (misclassified)  
- **Feature Deviations:** 6 very useful; 14 partially useful; 0 none :contentReference[oaicite:4]{index=4}

---

## 5. Key Findings  
1. **Quality vs. Coverage Trade-off:**  
   - Decision trees provide richer, rule-based explanations but may fail to mimic the ensemble (no explanation).  
   - Feature deviations always explain something, offering safe fallback at lower precision.  
2. **Data-Design Impact:** Failure modes involving rarely used features (e.g. `productID`) are harder to classify and explain.  
3. **Practical Recommendation:** Show decision-tree explanation when available; otherwise default to feature deviations. :contentReference[oaicite:5]{index=5}

---

## 6. Limitations & Future Work  
- Only two XAI methods studied; plan to evaluate **LIME**, **SHAP**, etc.  
- Investigate dynamic thresholding in decision trees to reduce “no explanation” cases.  
- Validate findings on **real-world PdM logs** beyond synthetic data. :contentReference[oaicite:6]{index=6}

---

## 7. Take-Away for CS 534  
- Understand **surrogate models** vs. **feature-attribution** explainers.  
- Use **human-centric metrics** (very/partial/limited/none) to evaluate explanations.  
- **Combine methods** in deployment to balance interpretability quality with robustness. :contentReference[oaicite:7]{index=7}