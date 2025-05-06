# 🧪 Scratch Detection on Wafer Maps – End-to-End ML Pipeline

**👤 Author:** Eyal Shubeli  
**📧 Email:** eyalshubeli1@gmail.com  
**📅 Date:** 2025-05-06

---

## 🧠 Problem Statement

Given spatial and statistical features of dies on semiconductor wafers, the goal is to identify which dies are part of a **scratch defect**.  
This is a highly imbalanced classification task where the positive class (`IsScratchDie=True`) is extremely rare.

---

## 🚦 Workflow Summary

### ✅ Step 1: Exploratory Data Analysis (EDA)
- Initial inspection of wafer layouts, distribution of labels, and zone-based segmentation.
- Identification of strong class imbalance (less than 1% positives).
- Detection of feature blocks by wafer zones: 0–6, 6–10, 10–16, 16, and 17.

---

### 🛠️ Step 2: Feature Extraction from Multiple Sources
- Loaded multiple `.parquet` files with different types of features:
  - Statistical densities
  - Defect cluster features
  - Geometric distances
  - Neighborhood patterns
- Each file represented features extracted from different spatial scopes.

---

### 🔗 Step 3: Feature Table Merging
- Merged all feature tables on keys:  
  `['WaferName', 'DieX', 'DieY', 'IsGoodDie', 'IsScratchDie']`
- Deduplicated overlapping columns (`_x`, `_y` suffixes).
- Verified full alignment across wafers and dies.

---

### 🧹 Step 4: Feature Selection & Reduction
- Removed duplicate, constant, and low-variance columns.
- Removed features not present in real test set.
- Retained only high-signal features based on:
  - Variance threshold
  - Tree-based feature importance (LightGBM/XGBoost)

---

### 🧪 Step 5: Balanced Train/Val/Test Split
- Created stratified subsets with **controlled positive sampling**:
  - 5,000 positive samples in both `val` and `test`
  - ~100,000 negative samples for each to match 1:20 ratio
  - Remaining positives used for training
- Final training set: 5,000 pos + 100,000 neg (shuffled)

---

### 🤖 Step 6: Model Training and Evaluation
Trained and compared the following classifiers:
- Random Forest
- Logistic Regression
- Ridge Classifier
- LightGBM
- Extra Trees
- **XGBoost** (Best)

**Evaluation Metrics:**
| Model     | Precision | Recall | F1 Score | Accuracy |
|-----------|-----------|--------|----------|----------|
| XGBoost   | 0.87      | 0.92   | 0.89     | 0.99     |
| Others    | Lower precision or recall on rare class |

✅ XGBoost was selected for its strong balance of sensitivity and robustness under class imbalance.

---

### 🚀 Step 7: Final Inference and Submission
- Applied final model to real test set (`test_features.parquet`)
- Filled NAs, selected required features
- Saved predictions to:

---

## 💡 Future Directions

- CNN-based representation learning from wafer images
- Semi-supervised learning for label-scarce cases
- Feature importance analysis via SHAP
- Full pipeline deployment using FastAPI + Docker

---

## 📬 Contact

📧 eyalshubeli1@gmail.com  
🚀 Built with ❤️ and XGBoost.

