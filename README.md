# Post-Induction Hypotension
## Overview
Post-induction hypotension (PIH) is a common and potentially complication during anesthesia induction. This project provides a comprehensive machine learning pipeline to predict PIH occurrence using physiological data from the [VitalDB](https://vitaldb.net/) dataset.

This project was built during a research internship supervised by [Bob AUBOUIN](https://github.com/BobAubouin) and Kaouther MOUSSA.

## Installation

Use a new virtual env and Python 3.11 for maximal compatibility.

```bash
git clone https://github.com/KevinGUILBAUT/PIH_analysis PIH
cd PIH
pip install -e .[dev]
```

## Project Structure

This project includes scripts for data preparation, feature extraction, visualization, and model training. Below is a summary of each component.


### Data Preparation

- `select_cases.py`  
  Selects the cases having specific features from the VitalDB dataset, and generates a txt having these cases.

- `create_database.py`  
  Creates the database from the txt cases with specific features that we will use in a feature_extraction script.

### Feature Extraction

- `script_feature_extract_base.py`  
  Extracts features from physiological signals using linear regressions.

- `script_feature_extract_smote_rfe.py` 
  Extracts linear trend features from patient time series data, applies SMOTE for class balancing, and uses RFE for feature selection.

I also try to incorporate some Propofol and Remifentanil injection to see the impact on the performances. For physiological signals, I didn't change anything  :

- `script_feature_extraction_two_reg.py`  
  Extracts features from physiological signals using linear regressions. For each injection, two regressions are performed on the post-induction signal: one over the entire post-induction period, and one over its second half only.

- `script_feature_extraction_half.py`  
  Extracts features from physiological signals using linear regressions. Two separate regressions are performed for injections: one on the first half and one on the second half of the post-induction period.

- `script_feature_extraction_mean.py`  
  Extracts features from physiological signals using linear regressions. For injection signals, we compute the mean signal for each injection, then performing a single regression on these averaged signals.
  
### Model Training & Optimization

- `train_XXX.py`  
  Optimizes a single XXX classifier using Optuna on the training set produced by `script_feature_extract_base.py`.

- `train_10_xgboost_skfold.py`
  Optimizes 10 XGBoost classifiers using Optuna with StratifiedKFold cross-validation from sklearn to produce different training and testing sets. Generates comprehensive results including mean ROC and PRC performance figures.
