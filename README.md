# Credit Card Fraud Detection System

A production-style fraud detection pipeline built for the BFSI (Banking, Financial Services, and Insurance) domain. The system combines statistical methods, unsupervised anomaly detection, and supervised machine learning into a hybrid ensemble scoring model with business impact simulation.

---

## Problem Statement

Credit card fraud is a critical challenge in the financial industry. Fraudulent transactions are rare (heavily imbalanced dataset) and often indistinguishable from legitimate ones based on a single feature alone. The goal is to build a system that:

- Detects fraud with high recall (missing fraud is costly)
- Minimizes false positives (flagging legitimate transactions wastes analyst time)
- Generates actionable alerts with explainable risk reasons
- Quantifies the business cost of different detection thresholds

---

## Approach

The pipeline follows a deliberate progression from simple to complex:

```
Statistical Baselines (Z-score, IQR)
        ↓
Unsupervised Anomaly Detection
        ↓
Supervised ML Models
        ↓
Hybrid Ensemble Risk Scoring
        ↓
Alert Generation + Business Impact Simulation
```

### Why this progression?
Statistical methods were tested first to establish a baseline and understand their limitations. When Z-score and IQR failed to capture fraud (fraud in this dataset is contextual, not univariate), the pipeline moved to multivariate ML approaches.

---

## Key Components

### 1. Feature Engineering
- `is_night` — transactions between 00:00–05:00 flagged as high-risk hours
- `high_velocity_flag` — more than 4 transactions in last 24 hours
- `low_trust_flag` — device trust score below 40
- One-hot encoding of merchant categories

### 2. Statistical Baselines
| Method | Result |
|--------|--------|
| Z-score Detection | Failed — fraud is not univariate extreme |
| IQR Detection | Minimal detection, high false positives |

**Finding:** Fraud behavior is contextual and requires multivariate approaches.

### 3. Unsupervised Anomaly Detection
| Model | Fraud Precision | Fraud Recall | Fraud F1 |
|-------|----------------|--------------|----------|
| Isolation Forest(contamination=0.015) | 0.41 | 0.43 | 0.42 |
| Isolation Forest(contamination=0.030) | 0.37 | 0.73 | 0.49 |
| One-Class SVM | 0.29 | Best recall (0.67) | 0.40 |
| Local Outlier Factor | 0.12 | 0.10 | 0.11 |

Isolation Forest(contamination=0.030) achieved the highest fraud recall among unsupervised models.

### 4. Supervised Models
| Model | ROC-AUC |
|-------|---------|
| Random Forest (class_weight=balanced) | High |
| XGBoost (scale_pos_weight) | Highest |

XGBoost with `scale_pos_weight` tuned to the fraud ratio outperformed Random Forest on ROC-AUC.

### 5. Hybrid Ensemble Risk Score
Final risk score combines three signals:

```python
final_score = (
    0.30 * stat_score +     # Rule-based: foreign txn, location mismatch, velocity, device trust
    0.40 * iso_score +      # Isolation Forest anomaly score (MinMax normalized)
    0.30 * xgb_score       # XGBoost predicted probability
)
```

Transactions bucketed into Low / Medium / High risk tiers.

### 6. Threshold Sensitivity Analysis
Scanned thresholds from 0.10 to 0.90 to find the optimal precision-recall tradeoff.

**Key finding:** Threshold of 0.75 achieved 96.7% fraud detection rate with zero false positives — significantly better than the default 0.50 threshold.

### 7. Business Impact Simulation
```
fraud_loss   = ₹500 per missed fraud
review_cost  = ₹50  per manual review

Results at High risk tier:
- Fraud caught:       30
- Fraud missed:        0
- False positives:    12
- Total business cost: ₹600
```

### 8. Alert Generation System
- Each high-risk transaction generates an alert with a unique ID
- Explainable reasons attached: Foreign Transaction, Location Mismatch, High Velocity, Low Device Trust
- Alerts exported to CSV for analyst review workflow
- False positive root cause analysis included

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | ML models, preprocessing pipelines |
| XGBoost | Gradient boosted classifier |
| Imbalanced-learn | SMOTE for class imbalance |
| Matplotlib, Seaborn | Visualizations |
| UUID, Datetime | Alert generation |

---

## Project Structure

```
Fraud_Detection/
├── fraud_pipeline.ipynb      # Main notebook — full pipeline
├── credit_card_fraud_10k.csv # Dataset (10,000 transactions)
├── outputs/
│   ├── alerts_generated.csv       # High-risk alerts
│   ├── alerts_investigated.csv    # With analyst assignments
│   ├── prioritized_alerts.csv     # Sorted by risk score
│   └── false_positive_cases.csv   # FP analysis
└── requirement.txt
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

# Open the notebook
jupyter notebook fraud_pipeline.ipynb
```

Run cells sequentially from top to bottom.

---

## Key Results

- **Zero fraud missed** at the optimized High risk threshold
- **96.7% detection rate** at threshold 0.75 with no false positives
- **One-Class SVM** best among unsupervised models (recall: 0.67)
- **XGBoost** best overall with highest ROC-AUC
- **Total business cost** at optimal threshold: ₹600 (12 manual reviews only)

---

## Known Limitations

- Dataset is synthetic/anonymized — real-world fraud patterns may differ
- Rule-based stat score weights (0.25 each) are heuristic, not learned
- Alert investigation workflow is simulated, not connected to a live system
- Model retraining pipeline not yet implemented for concept drift

---

## Certificate

This project was completed as part of a virtual externship with **Zetheta Algorithms Private Limited** under the role of **Fraud Detection Engineer** (Anomaly Detection Systems track).

Certificate issued: 16 March 2026

---

## Author

**Parag Pawar**
- GitHub: https://github.com/paragpawar0077
- LinkedIn: https://www.linkedin.com/in/parag-pawar-5383693a5/
- Email: paragpawar0077@gmail.com
