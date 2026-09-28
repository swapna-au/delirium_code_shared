# Final Model Specification

## Development configuration

- **Numeric imputation**: MICE (IterativeImputer), max_iter=5
- **Numeric scaling**: StandardScaler (zero mean, unit variance)
- **Categorical imputation**: Most frequent value
- **Categorical encoding**: One-hot (drop first, ignore unknown)
- **Feature selection stage 1**: L1 logistic regression (LogisticRegressionCV, cv=5)
- **Feature selection stage 2**: Random forest importance (SelectFromModel, n_estimators=50)
- **RF hyperparameters**: {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 2, 'random_state': 42}
- **LR hyperparameters**: {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 500, 'random_state': 42}
- **Probability calibration**: Platt scaling (sigmoid), fitted on held-out 20%
- **Train/calibration split**: 80% model training / 20% calibration (stratified)
- **Random seed**: 42
- **scikit-learn version**: 1.6.1
- **numpy version**: 2.2.4
- **N selected features**: 7

## Selected features

- age_patient
- x4at_score
- total_acb_score
- unique_drug_count
- nurse_verified_risk_of_cogn_impairment_Yes
- boc_risk_nominal_Yes
- dementia_1

## Logistic Regression — coefficients (standardised, per SD)

| Feature | Coefficient | Odds Ratio |
|---|---|---|
| boc_risk_nominal_Yes | 1.6534 | 5.225 |
| nurse_verified_risk_of_cogn_impairment_Yes | 0.8475 | 2.334 |
| dementia_1 | 0.7956 | 2.216 |
| age_patient | 0.3457 | 1.413 |
| x4at_score | 0.3043 | 1.356 |
| total_acb_score | 0.1964 | 1.217 |
| unique_drug_count | -0.1656 | 0.847 |
| (Intercept) | -4.0691 |  |

Platt scaling (LR): a=-1.0159, b=-0.0020

## Random Forest — feature importances

| Feature | Importance |
|---|---|
| x4at_score | 0.2878 |
| boc_risk_nominal_Yes | 0.2596 |
| age_patient | 0.1858 |
| dementia_1 | 0.1004 |
| nurse_verified_risk_of_cogn_impairment_Yes | 0.0866 |
| total_acb_score | 0.0456 |
| unique_drug_count | 0.0342 |

Platt scaling (RF): a=-7.5743, b=3.6232

## Applying the model

- `final_model_rf.pkl` / `final_model_lr.pkl` contain the complete pipeline (preprocessing + feature selection + model + Platt calibration). Load with `joblib.load(...)` and call `.predict_proba(X_raw)[:, 1]` on raw feature data with the same columns.
- Coefficients are on STANDARDISED predictors (per 1 SD); to obtain a raw-scale equation, back-transform using the StandardScaler means/SDs stored in the pipeline.