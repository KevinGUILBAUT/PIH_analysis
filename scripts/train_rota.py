from pathlib import Path
import pickle
import optuna
import pandas as pd
import joblib
from aeon.classification.sklearn import RotationForestClassifier
from sklearn.model_selection import train_test_split, KFold
from hp_pred.experiments import objective_rotationforest
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

rng_seed = 42

model_filename = "rotationforest_model_opt.pkl"

train = pd.read_csv('./data/features_extracted/train.csv')
test = pd.read_csv('./data/features_extracted/test.csv')

FEATURE_NAME = list(test.columns.difference(['label','cv_split']))

train = train.dropna(subset=FEATURE_NAME)
test = test.dropna(subset=FEATURE_NAME)

print(
    f"{len(train):,d} train samples, "
    f"{len(test):,d} test samples, "
    f"label mean: {test['label'].mean():.4f}"
)

# Set model file
model_folder = Path("./data/models")
model_folder.mkdir(exist_ok=True)
model_file = model_folder / model_filename

if model_file.exists():
    model = joblib.load(model_file)
else:
    number_fold = len(train.cv_split.unique())
    data_train_cv = [train[train.cv_split != f'cv_{i}'] for i in range(number_fold)]
    data_test_cv = [train[train.cv_split == f'cv_{i}'] for i in range(number_fold)]
    
    sampler = optuna.samplers.TPESampler(seed=rng_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_rotationforest(trial, data_train_cv, data_test_cv, FEATURE_NAME),
        n_trials=100,
        show_progress_bar=True,
    )
    
    best_params = study.best_params
    model = RotationForestClassifier(**best_params, random_state=rng_seed)
    model.fit(train[FEATURE_NAME], train.label)
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)