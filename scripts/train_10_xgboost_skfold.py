from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import optuna
import json
import matplotlib.pyplot as plt
import shap

from hp_pred.experiments import objective_xgboost_roc

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
N_MODELS = 10  # Number of folds to use
BASE_SEED = 42
N_TRIALS = 100
NB_CV = 3  # For hyperparameter optimization within each fold

# Load the original dataset
final_dataset = pd.read_csv("./data/features_extracted/data.csv")

# Convert bool to int
bool_columns = final_dataset.select_dtypes(include=['bool']).columns
if len(bool_columns) > 0:
    print(f"Converting boolean columns to integers: {list(bool_columns)}")
    final_dataset[bool_columns] = final_dataset[bool_columns].astype(int)

# Prepare features and target
X = final_dataset.drop(columns='label')
y = final_dataset['label']
FEATURE_NAME = list(X.columns)

# Create results storage
results = []
models = {}
model_folder = Path("./data/models/10_models")
if not model_folder.exists():
    model_folder.mkdir(parents=True)

# Create figures folder for SHAP plots
figures_folder = model_folder / "figures"
if not figures_folder.exists():
    figures_folder.mkdir(parents=True)

# Storage for ROC and PRC curves
roc_curves = []
prc_curves = []
all_test_labels = []
all_test_probas = []

# Storage for macro-averaging - we'll store test data for each model
test_datasets = []
train_datasets = []  # Store train datasets for SHAP

# Create StratifiedKFold with N_MODELS folds
print(f"Creating StratifiedKFold with {N_MODELS} folds...")
skf_main = StratifiedKFold(n_splits=N_MODELS, shuffle=True, random_state=BASE_SEED)

# Create fold indices
fold_indices = list(skf_main.split(X, y))

split_configs = []
for i in range(10):
    test_folds = [(i + j) % 10 for j in range(3)]  # 3 folds en test
    train_folds = [f for f in range(10) if f not in test_folds]
    split_configs.append((train_folds, test_folds))

for fold_idx, (train_folds, test_folds) in enumerate(split_configs):
    print(f"\nFold {fold_idx+1}/10: Train on folds {train_folds}, Test on folds {test_folds}")
    print("-" * 40)

    train_idx = np.concatenate([fold_indices[i][1] for i in train_folds])
    test_idx = np.concatenate([fold_indices[i][1] for i in test_folds])

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Create train and test datasets
    train = X_train.copy()
    train['label'] = y_train.values
    test = X_test.copy()
    test['label'] = y_test.values
    
    # Create cross-validation splits for hyperparameter optimization
    skf_cv = StratifiedKFold(n_splits=NB_CV, shuffle=True, random_state=BASE_SEED + fold_idx)
    cv_split_col = pd.Series(index=train.index, dtype="object")
    
    for cv_i, (_, val_idx) in enumerate(skf_cv.split(train.drop(columns='label'), train['label'])):
        cv_split_col.iloc[val_idx] = f'cv_{cv_i}'
    
    train['cv_split'] = cv_split_col
    test['cv_split'] = 'test'
    
    train = train.dropna(subset=FEATURE_NAME)
    test = test.dropna(subset=FEATURE_NAME)
    
    print(f"Train: {len(train)} samples, Test: {len(test)} samples")
    print(f"Train positive rate: {train['label'].mean():.3f}, Test positive rate: {test['label'].mean():.3f}")
    
    # Prepare CV data for hyperparameter optimization
    number_fold = len(train.cv_split.unique())
    data_train_cv = [train[train.cv_split != f'cv_{cv_i}'] for cv_i in range(number_fold)]
    data_test_cv = [train[train.cv_split == f'cv_{cv_i}'] for cv_i in range(number_fold)]
    
    # Hyperparameter optimization
    sampler = optuna.samplers.TPESampler(seed=BASE_SEED + fold_idx)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    print("Optimizing hyperparameters...")
    study.optimize(
        lambda trial: objective_xgboost_roc(trial, data_train_cv, data_test_cv, FEATURE_NAME),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    # Get best parameters and train final model
    best_params = study.best_params
    model = xgb.XGBClassifier(**best_params)
    model.fit(train[FEATURE_NAME], train['label'], verbose=0)
    
    # Evaluate model
    y_pred = model.predict(test[FEATURE_NAME])
    y_pred_proba = model.predict_proba(test[FEATURE_NAME])[:, 1]
    
    # Store test data with predictions for macro-averaging
    test_with_pred = test.copy()
    test_with_pred['y_pred_proba'] = y_pred_proba
    test_datasets.append(test_with_pred)
    
    # Store train data for SHAP
    train_datasets.append(train)
    
    # Store predictions and labels for aggregated curves
    all_test_labels.extend(test['label'].values)
    all_test_probas.extend(y_pred_proba)
    
    # Calculate curves for this model
    fpr, tpr, _ = roc_curve(test['label'], y_pred_proba)
    precision, recall, _ = precision_recall_curve(test['label'], y_pred_proba)
    
    # Store curves
    roc_curves.append({'fpr': fpr, 'tpr': tpr, 'fold': fold_idx+1})
    prc_curves.append({'precision': precision, 'recall': recall, 'fold': fold_idx+1})
    
    # Calculate metrics
    accuracy = accuracy_score(test['label'], y_pred)
    auc_score = roc_auc_score(test['label'], y_pred_proba)
    auprc = average_precision_score(test['label'], y_pred_proba)
    
    # Store results
    fold_result = {
        'fold': fold_idx + 1,
        'train_size': len(train),
        'test_size': len(test),
        'train_positive_rate': train['label'].mean(),
        'test_positive_rate': test['label'].mean(),
        'best_cv_score': study.best_value,
        'test_accuracy': accuracy,
        'test_auc': auc_score,
        'test_auprc': auprc,
        'best_params': best_params,
    }
    
    results.append(fold_result)
    models[f'fold_{fold_idx+1}'] = model
    
    # Save individual model
    model_filename = f"xgb_model_fold_{fold_idx+1}.json"
    model_file = model_folder / model_filename
    model.save_model(model_file)
    
    print(f"CV Score: {study.best_value:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc_score:.4f}")
    print(f"Test AUPRC: {auprc:.4f}")

print("\n" + "=" * 70)
print("SUMMARY OF ALL FOLDS")
print("=" * 70)

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Print summary statistics
print(f"\nPerformance Summary:")
print(f"Mean Test Accuracy: {results_df['test_accuracy'].mean():.4f} ± {results_df['test_accuracy'].std():.4f}")
print(f"Mean Test AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
print(f"Mean Test AUPRC: {results_df['test_auprc'].mean():.4f} ± {results_df['test_auprc'].std():.4f}")
print(f"Mean CV Score: {results_df['best_cv_score'].mean():.4f} ± {results_df['best_cv_score'].std():.4f}")

# Detailed results table
print(f"\nDetailed Results:")
print(results_df[['fold', 'best_cv_score', 'test_accuracy', 'test_auc', 'test_auprc', 'train_size', 'test_size']].to_string(index=False))

# ============================================================================
# PLOT AVERAGED ROC AND PRC CURVES
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING AVERAGED ROC AND PRC CURVES")
print("=" * 70)

# Macro-averaging (interpolate each curve then average)
def plot_macro_averaged_curves():
    fpr_grid = np.linspace(0, 1, 1000)
    recall_grid = np.linspace(0, 1, 1000)
    
    tprs = []  # Pour ROC
    precisions = []  # Pour PRC
    roc_aucs = []
    prc_aucs = []
    
    for i in range(N_MODELS):
        test = test_datasets[i]
        y_true = test['label'].values
        y_score = test['y_pred_proba'].values
        
        # ROC
        fpr_i, tpr_i, _ = roc_curve(y_true, y_score)
        tpr_interp = np.interp(fpr_grid, fpr_i, tpr_i)
        tprs.append(tpr_interp)
        roc_auc_i = auc(fpr_i, tpr_i)
        roc_aucs.append(roc_auc_i)
        
        # PRC
        precision_i, recall_i, _ = precision_recall_curve(y_true, y_score)
        # Need to reverse for recall
        precision_interp = np.interp(recall_grid, recall_i[::-1], precision_i[::-1])
        precisions.append(precision_interp)
        prc_auc_i = average_precision_score(y_true, y_score)
        prc_aucs.append(prc_auc_i)
    
    # Mean
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    mean_roc_auc = np.mean(roc_aucs)
    mean_prc_auc = np.mean(prc_aucs)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Macro-Average
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random classifier')
    ax1.plot(fpr_grid, mean_tpr, label=f'Macro-averaged ROC (AUC = {mean_roc_auc:.3f})', 
             color='blue', linewidth=2)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve (Macro Average)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # PRC Macro-Average
    # Calculate baseline precision from all test data
    all_test_labels_array = np.array(all_test_labels)
    baseline_precision = np.mean(all_test_labels_array)
    
    ax2.axhline(y=baseline_precision, color='k', linestyle='--', alpha=0.5, 
                label=f'Random classifier (AP = {baseline_precision:.3f})')
    ax2.plot(recall_grid, mean_precision, label=f'Macro-averaged PRC (AP = {mean_prc_auc:.3f})', 
             color='green', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('PRC Curve (Macro Average)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_folder / 'macro_averaged_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Macro-averaged ROC AUC: {mean_roc_auc:.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Macro-averaged PRC AUC: {mean_prc_auc:.4f} ± {np.std(prc_aucs):.4f}")

# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def plot_shap_values(fold_indices=None, max_display=15, save_fig=True):
    """Plot SHAP values for selected folds"""
    if fold_indices is None:
        fold_indices = range(min(3, N_MODELS))  # Plot first 3 folds by default
    
    print("\n" + "=" * 70)
    print("GENERATING SHAP ANALYSIS")
    print("=" * 70)
    
    for idx in fold_indices:
        fold_name = f"fold_{idx+1}"
        model = models[fold_name]
        test_data = test_datasets[idx]
        train_data = train_datasets[idx]
        
        print(f"\nComputing SHAP values for {fold_name}...")
        
        # Prepare data
        X_test = test_data[FEATURE_NAME]
        X_train = train_data[FEATURE_NAME]
        
        # Sample data if too large (for computational efficiency)
        if len(X_train) > 1000:
            print(f"Sampling 1000 training samples for SHAP background (from {len(X_train)} total)")
            X_train_sample = X_train.sample(n=1000, random_state=BASE_SEED + idx)
        else:
            X_train_sample = X_train
            
        if len(X_test) > 500:
            print(f"Sampling 500 test samples for SHAP explanation (from {len(X_test)} total)")
            X_test_sample = X_test.sample(n=500, random_state=BASE_SEED + idx)
        else:
            X_test_sample = X_test
        
        try:
            # Create explainer
            explainer = shap.Explainer(model, X_train_sample)
            shap_values = explainer(X_test_sample)
            
            # Bar plot (feature importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test_sample, 
                            plot_type="bar", max_display=max_display, show=False)
            plt.title(f'SHAP Feature Importance - {fold_name}')
            plt.tight_layout()
            if save_fig:
                plt.savefig(figures_folder / f"SHAP_importance_{fold_name}.png", 
                            dpi=300, bbox_inches='tight')
            plt.show()
            
            # Summary plot (feature impact)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test_sample, max_display=max_display, show=False)
            plt.title(f'SHAP Summary Plot - {fold_name}')
            plt.tight_layout()
            if save_fig:
                plt.savefig(figures_folder / f"SHAP_summary_{fold_name}.png", 
                            dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"SHAP plots generated for {fold_name}")
            
        except Exception as e:
            print(f"Error computing SHAP values for {fold_name}: {e}")
            continue

# Generate plots

print("\nGenerating macro-averaged curves...")
plot_macro_averaged_curves()

# Generate SHAP plots for selected folds
print("\nGenerating SHAP analysis...")
plot_shap_values(fold_indices=range(min(3, N_MODELS)), max_display=15, save_fig=True)

# Save results
results_file = model_folder / "stratified_kfold_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

results_csv = model_folder / "stratified_kfold_results.csv"
results_df.to_csv(results_csv, index=False)

print(f"\nResults saved to:")
print(f"- {results_file}")
print(f"- {results_csv}")
print(f"- Individual models saved in {model_folder}")
print(f"- Averaged curves plots saved in {model_folder}")
print(f"- SHAP plots saved in {figures_folder}")