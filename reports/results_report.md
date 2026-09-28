# Results

## Study cohort

The cohort comprised 28,565 admission episodes, of whom 2,284 (8.0%) developed delirium. 27 candidate predictors (4 numeric, 23 categorical) were evaluated; two-stage feature selection (L1-penalised logistic regression followed by random-forest importance) retained a stable core of 6 features across all folds: age, behaviours of concern, 4AT score, anticholinergic burden score.

## Model development and discrimination

A random forest (RF) and L1-penalised logistic regression (LR) were developed using 10-fold stratified nested cross-validation with 3-fold inner GridSearchCV for hyperparameter tuning and Platt scaling for probability calibration. In cross-validation, the calibrated RF and LR showed comparable discrimination (RF c-statistic 0.8778 [0.8711–0.8846]; LR 0.8776 [0.8719–0.8832]). The RF achieved higher sensitivity (0.1835 [0.1603–0.2067] vs 0.1568 [0.1369–0.1766]) and F1-score (0.2672 [0.2381–0.2963] vs 0.2413 [0.2145–0.2681]) and was selected as the primary model given the clinical priority of case identification.

Internal validity was assessed by bootstrap resampling (500 iterations) with Harrell's optimism correction. Optimism in the c-statistic was small (0.0224), indicating limited overfitting. The optimism-corrected c-statistic was 0.8889 for the RF and 0.8783 for the LR. Bootstrap out-of-sample discrimination (calibrated) was 0.9038 [0.8976–0.9120] (RF) and 0.8788 [0.8781–0.8792] (LR).

## Calibration

Following Platt scaling, both models were well calibrated. The RF calibration slope was 0.9322 [0.8702–0.9947] with a Brier score of 0.0533 [0.0518–0.0553] (Brier skill score 0.3006); the LR calibration slope was 0.9679 [0.9008–1.0387] with a Brier score of 0.0570 [0.0569–0.0572] (Brier skill score 0.2244). Both Brier scores were below the null-model benchmark of 0.074 (calibration-in-the-large intercept RF 0.5062; LR 0.3765). Overall, the calibration slopes and calibration curves (Figure [X]) indicated good agreement between predicted and observed risk across the range of predicted probabilities.

## Clinical utility

Decision curve analysis (Figure [X]) showed that both calibrated models provided greater net benefit than the 'treat all' and 'treat none' strategies across the clinically relevant range. At a 15% risk threshold, the net benefit of the RF model was 0.0392, exceeding 'treat all' (-0.0829).

## Operating characteristics

At a 15% decision threshold, the calibrated RF achieved a sensitivity of 0.6173 and specificity of 0.9375 (PPV 0.462, NPV 0.9657), flagging 10.68% of admissions for review (Table [X]). Sensitivity and specificity across thresholds are shown in Figure [X].

## Subgroup analysis by sex

Model performance (random forest) by sex subgroup:

  - M: n=14494, prevalence=7.55%, AUC=0.9106, Sensitivity@15%=0.6018

  - F: n=14065, prevalence=8.45%, AUC=0.9114, Sensitivity@15%=0.6313

  - All: n=28565, prevalence=8.0%, AUC=0.9112, Sensitivity@15%=0.6173


Performance was consistent across sex, supporting equitable model behaviour.

## Sensitivity analyses

Model discrimination (AUC) was robust across sensitivity analyses:

  - 1. Without 4AT (RF): with=0.878, without=0.869, difference=0.009

  - 1. Without 4AT (LR): with=0.895, without=0.89, difference=0.005

  - 2. Symptom Variables (RF): with=0.878, without=0.861, difference=0.017

  - 2. Symptom Variables (LR): with=0.895, without=0.875, difference=0.02

  - 3. Unit of Analysis (RF): with=0.878, without=0.878, difference=0.0

  - 3. Unit of Analysis (LR): with=0.895, without=0.894, difference=0.0

## Appendix: Model specification

Two-stage feature selection retained a stable core of 4 features selected in all cross-validation folds (age, behaviours of concern, 4AT score, anticholinergic burden score). When refitted on the full development dataset, the final model additionally retained cognitive impairment risk, giving 6 predictors in total. Both the 4-feature cross-validation core and the 6-feature final model are reported for transparency.

### Development configuration

  - Numeric imputation: MICE (IterativeImputer), max_iter=5

  - Numeric scaling: StandardScaler (zero mean, unit variance)

  - Categorical imputation: Most frequent value

  - Categorical encoding: One-hot (drop first, ignore unknown)

  - Feature selection stage 1: L1 logistic regression (LogisticRegressionCV, cv=5)

  - Feature selection stage 2: Random forest importance (SelectFromModel, n_estimators=50)

  - RF hyperparameters: n_estimators=200, max_depth=10, min_samples_leaf=2

  - LR hyperparameters: C=1.0, penalty=l1, solver=liblinear, max_iter=500

  - Probability calibration: Platt scaling (sigmoid), fitted on held-out 20%

  - Train/calibration split: 80% model training / 20% calibration (stratified)

  - Random seed: 42

  - scikit-learn version: 1.6.1

  - numpy version: 2.2.4

  - N selected features: 6

### Logistic regression — coefficients (standardised, per 1 SD)

  - behaviours of concern: coefficient = 1.9277, odds ratio = 6.874

  - dementia_1.0: coefficient = 0.928, odds ratio = 2.5294

  - age: coefficient = 0.4219, odds ratio = 1.5249

  - 4AT score: coefficient = 0.3509, odds ratio = 1.4203

  - anticholinergic burden score: coefficient = 0.2747, odds ratio = 1.3162

  - number of anticholinergic medications: coefficient = -0.2468, odds ratio = 0.7813

  - (Intercept): coefficient = -3.7897

### Random forest — feature importances

  - 4AT score: 0.316

  - behaviours of concern: 0.2787

  - age: 0.2053

  - dementia_1.0: 0.1191

  - anticholinergic burden score: 0.0478

  - number of anticholinergic medications: 0.033
