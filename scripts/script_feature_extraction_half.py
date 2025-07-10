"""  
Extracts features from physiological signals using linear regressions. Two separate regressions are performed for injections: one on the first half and one on the second half of the post-induction period.
"""

# Import
import sys
import pandas as pd
import numpy as np
import python_anesthesia_simulator as pas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
import os
from sklearn.model_selection import train_test_split, StratifiedKFold


warnings.filterwarnings('ignore')

def compute_linear_trends_for_patient(patient_data, 
                                    window_sizes,
                                    signal_mapping=None):
    """
    Compute linear trend features for a single patient's time series data.
    """
    if signal_mapping is None:
        signal_mapping = {
            'BIS': 'bis',
            'MAP': 'map', 
            'NI_MAP': 'ni_map',
            'SBP': 'sbp',
            'NI_SBP': 'ni_sbp', 
            'DBP': 'dbp',
            'NI_DBP': 'ni_dbp',
            'HR': 'hr',
            'SpO2': 'spo2',
            'EtCO2': 'etco2',
            'Body_Temp': 'body_temp',
            'RR': 'rr',
        }
    
    patient_features = {}
    
    # Get patient static info (from first row)
    first_row = patient_data.iloc[0]
    patient_features['age'] = first_row['age']
    patient_features['asa'] = first_row['asa']
    patient_features['gender'] = first_row['gender']
    patient_features['weight'] = first_row['weight']
    patient_features['height'] = first_row['height']
    patient_features['bmi'] = first_row['bmi']
    
    patient_features['preop_hyper'] = first_row['preop_hyper']
    patient_features['preop_diab'] = first_row['preop_diab']
    patient_features['preop_ecg'] = first_row['preop_ecg']
    patient_features['preop_pft'] = first_row['preop_pft']
    patient_features['preop_crea'] = first_row['preop_crea']
    patient_features['label'] = first_row['hypotension']

    patient_features['MAP_base_case'] = first_row['MAP_base_case']
    patient_features['Sbp_base_case'] = first_row['Sbp_base_case']
    patient_features['Dbp_base_case'] = first_row['Dbp_base_case']

    patient_features['Target_Propo_Base'] = first_row['Target_Propo_Base']
    patient_features['Target_Remi_Base'] = first_row['Target_Remi_Base']

    patient_features['duration_first_injec_propo'] = first_row['duration_first_injec_propo']
    patient_features['mean_first_injec_propo'] = first_row['mean_first_injec_propo']
    patient_features['duration_first_injec_remi'] = first_row['duration_first_injec_remi']
    patient_features['mean_first_injec_remi'] = first_row['mean_first_injec_remi']

    patient_features['optype'] = first_row['optype']
    patient_features['department'] = first_row['department']

    induction_start = first_row['start_induc_time']
    intubation_start = first_row['start_intubation_time']

    # Only between induction and intubation for the drugs
    drug_df = patient_data.iloc[induction_start:intubation_start].copy()

    window_sizes = [len(drug_df)] 

    for drug in ['Propofol', 'Remifentanil','Vol_Propo','Vol_Remi']:
        drug_values = drug_df[drug].fillna(0).values 


        in_injection = False
        nonzero_values = []

        # Statistics
        if drug in ['Propofol', 'Remifentanil']:
            for val in drug_values:
                if val > 0:
                    nonzero_values.append(val)
                    if not in_injection:
                        in_injection = True
                    else:
                        in_injection = False

            patient_features[f'mean_{drug.lower()}'] = np.mean(nonzero_values) if nonzero_values else np.nan
            patient_features[f'max_{drug.lower()}'] = np.max(nonzero_values) if nonzero_values else np.nan
            patient_features[f'std_{drug.lower()}'] = np.std(nonzero_values) if nonzero_values else np.nan
        else:
            patient_features[f'last_vol_{drug.lower()}'] = drug_values[-1]

        # Linear Regression on the 2 parts
        for window in window_sizes:
            window_size = min(window, len(drug_values))
            if window_size < 4: 
                # First part
                patient_features[f"{drug}_slope_{window}_first"] = np.nan
                patient_features[f"{drug}_intercept_{window}_first"] = np.nan
                patient_features[f"{drug}_residual_std_{window}_first"] = np.nan
                patient_features[f"{drug}_mse_{window}_first"] = np.nan
                patient_features[f"{drug}_pred_mid_{window}_first"] = np.nan
                
                # Second part
                patient_features[f"{drug}_slope_{window}_second"] = np.nan
                patient_features[f"{drug}_intercept_{window}_second"] = np.nan
                patient_features[f"{drug}_residual_std_{window}_second"] = np.nan
                patient_features[f"{drug}_mse_{window}_second"] = np.nan
                patient_features[f"{drug}_pred_mid_{window}_second"] = np.nan
                continue

            y = drug_values[-window_size:]
            X = np.arange(-window_size, 0).reshape(-1, 1)
            
            half_size = window_size // 2
            
            # First part 
            y_first = y[:half_size]
            X_first = X[:half_size]
            
            # Second part
            y_second = y[half_size:]
            X_second = X[half_size:]
            
            # Fit linear regression on the first part
            valid_mask_first = ~np.isnan(y_first)
            n_valid_first = np.sum(valid_mask_first)
            
            if n_valid_first >= 2: 
                X_fit_first = X_first[valid_mask_first]
                y_fit_first = y_first[valid_mask_first]
                
                try:
                    model_first = LinearRegression()
                    model_first.fit(X_fit_first, y_fit_first)
                    y_pred_first = model_first.predict(X_fit_first)
                    residuals_first = y_fit_first - y_pred_first
                    
                    patient_features[f"{drug}_slope_{window}_first"] = model_first.coef_[0]
                    patient_features[f"{drug}_intercept_{window}_first"] = model_first.intercept_
                    patient_features[f"{drug}_residual_std_{window}_first"] = residuals_first.std()
                    patient_features[f"{drug}_mse_{window}_first"] = np.mean(residuals_first ** 2)
                    
                    mid_first_idx = half_size // 2
                    X_mid_first = X_first[mid_first_idx:mid_first_idx+1]
                    patient_features[f"{drug}_pred_mid_{window}_first"] = model_first.predict(X_mid_first)[0]
                    
                except Exception:
                    patient_features[f"{drug}_slope_{window}_first"] = np.nan
                    patient_features[f"{drug}_intercept_{window}_first"] = np.nan
                    patient_features[f"{drug}_residual_std_{window}_first"] = np.nan
                    patient_features[f"{drug}_mse_{window}_first"] = np.nan
                    patient_features[f"{drug}_pred_mid_{window}_first"] = np.nan
            else:
                patient_features[f"{drug}_slope_{window}_first"] = np.nan
                patient_features[f"{drug}_intercept_{window}_first"] = np.nan
                patient_features[f"{drug}_residual_std_{window}_first"] = np.nan
                patient_features[f"{drug}_mse_{window}_first"] = np.nan
                patient_features[f"{drug}_pred_mid_{window}_first"] = np.nan
            
            # Fit linear regression on the second part
            valid_mask_second = ~np.isnan(y_second)
            n_valid_second = np.sum(valid_mask_second)
            
            if n_valid_second >= 2: 
                X_fit_second = X_second[valid_mask_second]
                y_fit_second = y_second[valid_mask_second]
                
                try:
                    model_second = LinearRegression()
                    model_second.fit(X_fit_second, y_fit_second)
                    y_pred_second = model_second.predict(X_fit_second)
                    residuals_second = y_fit_second - y_pred_second
                    
                    patient_features[f"{drug}_slope_{window}_second"] = model_second.coef_[0]
                    patient_features[f"{drug}_intercept_{window}_second"] = model_second.intercept_
                    patient_features[f"{drug}_residual_std_{window}_second"] = residuals_second.std()
                    patient_features[f"{drug}_mse_{window}_second"] = np.mean(residuals_second ** 2)
                    
                    mid_second_idx = len(y_second) // 2
                    X_mid_second = X_second[mid_second_idx:mid_second_idx+1]
                    patient_features[f"{drug}_pred_mid_{window}_second"] = model_second.predict(X_mid_second)[0]
                    
                except Exception:
                    patient_features[f"{drug}_slope_{window}_second"] = np.nan
                    patient_features[f"{drug}_intercept_{window}_second"] = np.nan
                    patient_features[f"{drug}_residual_std_{window}_second"] = np.nan
                    patient_features[f"{drug}_mse_{window}_second"] = np.nan
                    patient_features[f"{drug}_pred_mid_{window}_second"] = np.nan
            else:
                patient_features[f"{drug}_slope_{window}_second"] = np.nan
                patient_features[f"{drug}_intercept_{window}_second"] = np.nan
                patient_features[f"{drug}_residual_std_{window}_second"] = np.nan
                patient_features[f"{drug}_mse_{window}_second"] = np.nan
                patient_features[f"{drug}_pred_mid_{window}_second"] = np.nan

    
    # Compute trends for each signal and window size
    for original_col, signal_name in signal_mapping.items():
        if original_col not in patient_data.columns:
            continue
            
        signal_values = patient_data[:induction_start][original_col].values #we only want before induction for those signals
        window_sizes = [60]
        # Skip if all values are NaN
        if np.all(np.isnan(signal_values)):
            for window in window_sizes:
                patient_features[f"{signal_name}_slope_{window}"] = np.nan
                patient_features[f"{signal_name}_intercept_{window}"] = np.nan
                patient_features[f"{signal_name}_residual_std_{window}"] = np.nan
            patient_features[f"{signal_name}_slope_full"] = np.nan
            patient_features[f"{signal_name}_intercept_full"] = np.nan
            patient_features[f"{signal_name}_residual_std_full"] = np.nan
            continue

        # 1. Regression on full signal
        X_full = np.arange(-len(signal_values),0).reshape(-1, 1)
        y_full = signal_values
        
        # Handle NaN values
        if original_col :
            valid_mask_full = ~np.isnan(y_full)
            if np.sum(valid_mask_full) < 2:
                patient_features[f"{signal_name}_slope_full"] = np.nan
                patient_features[f"{signal_name}_intercept_full"] = np.nan
                patient_features[f"{signal_name}_residual_std_full"] = np.nan
            else:
                X_fit_full = X_full[valid_mask_full]
                y_fit_full = y_full[valid_mask_full]
                
                try:
                    model_full = LinearRegression()
                    model_full.fit(X_fit_full, y_fit_full)
                    y_pred_full = model_full.predict(X_fit_full)
                    residuals_full = y_fit_full - y_pred_full
                    
                    patient_features[f"{signal_name}_slope_full"] = model_full.coef_[0]
                    patient_features[f"{signal_name}_intercept_full"] = model_full.intercept_
                    patient_features[f"{signal_name}_residual_std_full"] = residuals_full.std()
                    
                except Exception as e:
                    patient_features[f"{signal_name}_slope_full"] = np.nan
                    patient_features[f"{signal_name}_intercept_full"] = np.nan
                    patient_features[f"{signal_name}_residual_std_full"] = np.nan

        # 2. Regression on windows signals
        for window in window_sizes:
            window_size = min(window, len(signal_values))
            if window_size < 2:
                patient_features[f"{signal_name}_slope_{window}"] = np.nan
                patient_features[f"{signal_name}_intercept_{window}"] = np.nan
                patient_features[f"{signal_name}_residual_std_{window}"] = np.nan
                continue
                
            y = signal_values[-window_size:]
            X = np.arange(-len(y),0).reshape(-1, 1)
            
            # Handle NaN values 
            if original_col :
                valid_mask = ~np.isnan(y)
                if np.sum(valid_mask) < 2:
                    patient_features[f"{signal_name}_slope_{window}"] = np.nan
                    patient_features[f"{signal_name}_intercept_{window}"] = np.nan
                    patient_features[f"{signal_name}_residual_std_{window}"] = np.nan
                    continue
                X_fit = X[valid_mask]
                y_fit = y[valid_mask]
            else:
                X_fit = X
                y_fit = y
            
            try:
                # Fit linear regression
                model = LinearRegression()
                model.fit(X_fit, y_fit)
                y_pred = model.predict(X_fit)
                residuals = y_fit - y_pred
                
                patient_features[f"{signal_name}_slope_{window}"] = model.coef_[0]
                patient_features[f"{signal_name}_intercept_{window}"] = model.intercept_
                patient_features[f"{signal_name}_residual_std_{window}"] = residuals.std()
                
            except Exception as e:
                patient_features[f"{signal_name}_slope_{window}"] = np.nan
                patient_features[f"{signal_name}_intercept_{window}"] = np.nan
                patient_features[f"{signal_name}_residual_std_{window}"] = np.nan
    
    for original_col, signal_name in signal_mapping.items():
        if original_col in patient_data.columns:
            signal_values = patient_data[:induction_start][original_col].values
            patient_features[f"{signal_name}_mean"] = np.nanmean(signal_values)
            patient_features[f"{signal_name}_std_overall"] = np.nanstd(signal_values)
            patient_features[f"{signal_name}_min"] = np.nanmin(signal_values)
            patient_features[f"{signal_name}_max"] = np.nanmax(signal_values)
            if len(signal_values) > 0:
                signal_series = pd.Series(signal_values)
                patient_features[f"{signal_name}_last_value"] = signal_series.fillna(method='ffill').iloc[-1]
            else:
                patient_features[f"{signal_name}_last_value"] = np.nan

    return patient_features


def process_patient_parallel(args):
    """
    Process a single patient for parallel processing.
    """
    caseid, patient_data, window_sizes, signal_mapping = args
    try:
        return compute_linear_trends_for_patient(patient_data, window_sizes, signal_mapping)
    except Exception as e:
        print(f"Error processing patient {caseid}: {e}")
        return None
    
def drop_high_nan_columns(df, threshold=0.7):
    """
    Drop columns that have more than threshold (default 70%) of NaN values.
    """
    initial_cols = df.shape[1]
    
    # Calculate NaN ratio for each column
    nan_ratios = df.isnull().sum() / len(df)
    
    # Identify columns with more than 40% NaN values
    high_nan_40_percent = nan_ratios[nan_ratios > 0.4].index.tolist()
    if high_nan_40_percent:
        print(f"\nWarning: The following columns have more than 40% NaN values: ")
        for col in high_nan_40_percent:
            print(f"  - {col}: {nan_ratios[col]:.2%}")
        
    # Identify columns to drop
    cols_to_drop = nan_ratios[nan_ratios > threshold].index.tolist()
    
    # Drop columns
    df_cleaned = df.drop(columns=cols_to_drop)
    
    print(f"\nNaN column filtering:")
    print(f"- Initial number of columns: {initial_cols}")
    print(f"- Columns dropped (>{threshold*100}% NaN): {len(cols_to_drop)}")
    if cols_to_drop:
        print(f"- Dropped columns: {cols_to_drop}")
    print(f"- Final number of columns: {df_cleaned.shape[1]}")
    
    # Impute remaining NaNs with median
    num_imputed = df_cleaned.isnull().sum().sum()
    df_cleaned = df_cleaned.apply(lambda col: col.fillna(col.median()) if col.dtype != 'object' else col)

    print(f"- Remaining NaNs imputed with median: {num_imputed}")

    return df_cleaned


if __name__ == "__main__":
    Full_data = pd.read_csv('data/full_data.csv')
    print("\nComputing linear trend features for each patient...")

    # Prepare data for parallel processing 
    patient_groups = [(caseid, group) for caseid, group in Full_data.groupby('caseid')]

    # Configuration for trend extraction
    window_sizes =[] # Different time windows 
    signal_mapping = {
        'BIS': 'bis',
        #'MAP': 'map', 
        'NI_MAP': 'ni_map',
        #'SBP': 'sbp',
        'NI_SBP': 'ni_sbp', 
        #'DBP': 'dbp',
        'NI_DBP': 'ni_dbp',
        'HR': 'hr',
        'SpO2': 'spo2',
        #'EtCO2': 'etco2',
        #'Body_Temp': 'body_temp',
        #'RR': 'rr',
    }

    n_jobs = max(1, cpu_count() - 1)
    print(f"Using {n_jobs} parallel processors")

    args_list = [(caseid, patient_data, window_sizes, signal_mapping) 
                for caseid, patient_data in patient_groups]

    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(process_patient_parallel, args_list), 
            total=len(args_list),
            desc="Extracting trend features"
        ))

    # Filter out failed results
    valid_results = [result for result in results if result is not None]

    if valid_results:
        rng_seed=42
        final_dataset = pd.DataFrame(valid_results)
        
        print(f"\nFinal dataset shape: {final_dataset.shape}")
        print(f"Number of patients: {len(final_dataset)}")
        print(f"Number of features: {final_dataset.shape[1]}")
        
        # Drop columns with more than 70% NaN values
        final_dataset = drop_high_nan_columns(final_dataset, threshold=0.5)

        
        # Encodage des variables catégorielles
        final_dataset = pd.get_dummies(final_dataset, columns=['preop_ecg', 'preop_pft','optype', 'department'], prefix=['ecg', 'pft','op','dep'])

        # Save final dataset
        os.makedirs('./data/features_extracted', exist_ok=True)
        final_dataset.to_csv("./data/features_extracted/data.csv", index=False)
        
        print(f"\nDataset saved to:")
        print(f"- ./data/features_extracted/data.csv")
        
        # Display some statistics
        print(f"\nDataset statistics:")
        print(f"- Hypotension cases: {final_dataset['label'].sum()}")
        print(f"- Normal cases: {len(final_dataset) - final_dataset['label'].sum()}")
        print(f"- Hypotension rate: {final_dataset['label'].mean():.3f}")
        
        # Show feature columns
        trend_features = [col for col in final_dataset.columns if any(suffix in col for suffix in ['_slope_', '_intercept_', '_residual_std_'])]
        print(f"\nNumber of trend features: {len(trend_features)}")
        
        summary_features = [col for col in final_dataset.columns if any(suffix in col for suffix in ['_mean', '_std_overall', '_min', '_max', '_last_value'])]
        print(f"Number of summary features: {len(summary_features)}")

        # Convert bool into int
        bool_columns = final_dataset.select_dtypes(include=['bool']).columns
        if len(bool_columns) > 0:
            print(f"\nConverting boolean columns to integers: {list(bool_columns)}")
            final_dataset[bool_columns] = final_dataset[bool_columns].astype(int)
            print(f"Converted {len(bool_columns)} boolean columns to integers")
        else:
            print("\nNo boolean columns found to convert")

        X = final_dataset.drop(columns='label')
        y = final_dataset['label']

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng_seed, test_size=0.3,stratify=y)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        nb_cv = 3
        skf = StratifiedKFold(n_splits=nb_cv, shuffle=True, random_state=rng_seed)

        cv_split_col = pd.Series(index=train.index, dtype="object")
        for i, (_, val_idx) in enumerate(skf.split(train.drop(columns='label'), train['label'])):
            cv_split_col.iloc[val_idx] = f'cv_{i}'

        train['cv_split'] = cv_split_col
        test['cv_split'] = 'test'

        test.to_csv("./data/features_extracted/test.csv", index=False)
        train.to_csv("./data/features_extracted/train.csv", index=False)

    else:
        print("No valid results obtained. Please check your data and processing.")

        