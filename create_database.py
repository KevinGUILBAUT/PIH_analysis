"""Create a dataset from vitalDB"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

warnings.filterwarnings('ignore')


try:
    from vitaldb_local import load_cases
except:
    print('Could not import vitaldb_local, import online version')
    import vitaldb as vdb

    def load_cases(track_names: list, caseids: list):
        """Import a list of cases from vitaldb in a dataframe format."""
        dataframe_final = pd.DataFrame()
        for caseid in caseids:
            print(caseid)
            cases = vdb.VitalFile(caseid, track_names)
            dataframe_temp = cases.to_pandas(track_names, 1)
            dataframe_temp.insert(0, 'caseid', caseid)
            dataframe_final = pd.concat([dataframe_final, dataframe_temp], ignore_index=True)
        return dataframe_final


def process_patient(patient_data, perso_data):
    """
    Function to process an individual patient.
    This function will be executed in parallel.
    """
    caseid, Patient_df = patient_data
    print(f"Processing case {caseid}")
    
    try:
        # find MAP baseline
        Patient_df = Patient_df.copy()
        Patient_df.reset_index(inplace=True)
        Patient_df.drop(columns=['index'], inplace=True)
        Map_base_case = Patient_df['NI_MAP'].fillna(method='bfill')[0]
        Sbp_base_case = Patient_df['NI_SBP'].fillna(method='bfill')[0]
        DBP_base_case = Patient_df['NI_DBP'].fillna(method='bfill')[0]
        Target_Remi = Patient_df['Target_Remi'].replace(0, np.nan).fillna(method='bfill')[0]
        Target_Propo = Patient_df['Target_Propo'].replace(0, np.nan).fillna(method='bfill')[0]

        Patient_df.insert(len(Patient_df.columns), "Target_Remi_Base", Target_Remi)
        Patient_df.insert(len(Patient_df.columns), "Target_Propo_Base", Target_Propo)
        Patient_df.insert(len(Patient_df.columns), "MAP_base_case", Map_base_case)
        Patient_df.insert(len(Patient_df.columns), "Sbp_base_case", Sbp_base_case)
        Patient_df.insert(len(Patient_df.columns), "Dbp_base_case", DBP_base_case)

        # replace nan by previous value in drug rates
        Patient_df['Propofol'].fillna(method='bfill', inplace=True)
        Patient_df['Remifentanil'].fillna(method='bfill', inplace=True)

        # find first drug injection
        induction_start = 0
        for i in range(len(Patient_df)):
            if Patient_df.loc[i, 'Propofol'] != 0 or Patient_df.loc[i, 'Remifentanil'] != 0:
                induction_start = i
                break
        
        # removes after intubation
        # Detection of intubation period start
        # We look for when RR becomes constant for 2min30 (150 seconds)
        intubation_start = None
        window_size = 150

        if 'RR' in Patient_df.columns and not Patient_df['RR'].isna().all():
            Patient_df['RR'].replace(0, np.nan, inplace=True)
            Patient_df['RR'].fillna(method='ffill', inplace=True)

            """for i in range(induction_start, len(Patient_df) - window_size):
                # Extraire la fenêtre de RR
                rr_window = Patient_df.loc[i:i+window_size-1, 'RR']

                # Vérifier s'il y a des NaNs dans la fenêtre
                if rr_window.isna().any():
                    continue  # Passer à la fenêtre suivante

                rr_std = rr_window.std()
                #rr_mean = rr_window.mean()
                    
                # Considérer comme constant si l'écart-type absolu est très faible
                if (rr_std < 0.1):
                    intubation_start = i
                    intubation_detection_method = "variance"
                    break"""

        if intubation_start is None:
            # Generally intubation occurs within 10-15 minutes after induction
            intubation_start = min(induction_start + 900, len(Patient_df) - 1)  #  15 minutes after induction
            intubation_detection_method = "fixed_time"

        elif intubation_start == 0 :
            return None 
        
        Patient_df.insert(len(Patient_df.columns), "start_intubation_time", intubation_start)
        Patient_df.insert(len(Patient_df.columns), "start_induc_time", induction_start)

        drug_df = Patient_df[induction_start:intubation_start]

        for drug in ['Propofol', 'Remifentanil', 'Vol_Propo', 'Vol_Remi']:
            drug_values = drug_df[drug].fillna(0).values

            if drug in ['Propofol', 'Remifentanil']:
                num_injections = 0
                in_injection = False
                current_injection = []
                mean_at_start = np.zeros_like(drug_values)
                
                # For first injection only
                first_injection_mean = None
                first_injection_duration = None

                for i, val in enumerate(drug_values):
                    if val > 0:
                        current_injection.append(val)
                        if not in_injection:
                            start_idx = i
                            in_injection = True
                    else:
                        if in_injection:
                            mean_val = np.mean(current_injection)
                            mean_at_start[start_idx] = mean_val 
                            num_injections += 1

                            if num_injections == 1:
                                first_injection_mean = mean_val
                                first_injection_duration = len(current_injection)

                            current_injection = []
                            in_injection = False

                # Case where injection continues until the end
                if in_injection:
                    mean_val = np.mean(current_injection)
                    mean_at_start[start_idx] = mean_val
                    num_injections += 1
                    if num_injections == 1:
                        first_injection_mean = mean_val
                        first_injection_duration = len(current_injection)

                Patient_df[f'num_injections_{drug.lower()}'] = num_injections
                new_feature = np.zeros(len(Patient_df))
                new_feature[induction_start:intubation_start] = mean_at_start
                Patient_df[f'mean_injection_{drug.lower()}'] = new_feature

                # Store mean and duration of first injection
                if drug == 'Propofol':
                    Patient_df['mean_first_injec_propo'] = first_injection_mean
                    Patient_df['duration_first_injec_propo'] = first_injection_duration
                elif drug == 'Remifentanil':
                    Patient_df['mean_first_injec_remi'] = first_injection_mean
                    Patient_df['duration_first_injec_remi'] = first_injection_duration

            else:
                # For volumes we just store the last value
                Patient_df[f'last_vol_{drug.lower()}'] = drug_values[-1]

        ## Intubation figure
        plt.figure(figsize=(14, 6))
        plt.plot(Patient_df['RR'], label='RR (respiratory rate)', color='blue')
        plt.axvline(induction_start, color='green', linestyle='--', label='Début induction')
        plt.axvline(intubation_start, color='red', linestyle='--', label=f'Début intubation ({intubation_detection_method})')

        # If method is "variance", color the detection zone
        if intubation_detection_method == "variance":
            plt.axvspan(intubation_start, intubation_start + window_size, color='orange', alpha=0.3, label='Fenêtre RR stable')

        plt.xlabel('Temps (secondes)')
        plt.ylabel('RR')
        if intubation_detection_method == "variance":
            plt.title(f'Détection du début d intubation : {intubation_start/60} mns')
        else:
            plt.title(f'Détection du début d intubation via fixed time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./data/figures/case_{caseid}.png')

        Patient_df.insert(len(Patient_df.columns), "intubation_detection_method", intubation_detection_method)


        # Hypotension labeling after induction
        threshold = 0.35

        # We consider the patient has hypotension if there's a drop of 'threshold' compared to Map_Base_Case
        has_hypotension = (Patient_df[induction_start:intubation_start]['NI_MAP'] < (1 - threshold) * Map_base_case).any()
        Patient_df['hypotension'] = int(has_hypotension)

        # replace 0 by nan in BIS, MAP, SBP, DBP,SPo2, ETCO2, BT and HR
        Patient_df['NI_MAP'].replace(0, np.nan, inplace=True)
        Patient_df['NI_SBP'].replace(0, np.nan, inplace=True)
        Patient_df['NI_DBP'].replace(0, np.nan, inplace=True)
        
        Patient_df['BIS'].replace(0, np.nan, inplace=True)
        Patient_df['HR'].replace(0, np.nan, inplace=True)
        Patient_df['SpO2'].replace(0, np.nan, inplace=True)
        Patient_df['EtCO2'].replace(0, np.nan, inplace=True)
        Patient_df['Body_Temp'].replace(0, np.nan, inplace=True)

        Patient_df = Patient_df.fillna(method='ffill')

        # Add time column
        Patient_df.insert(1, "Time", np.arange(0, len(Patient_df['BIS'])))

        # Add personnal information to the dataframe
        age = perso_data.loc[perso_data['caseid'] == caseid, 'age'].astype(float).item()
        Patient_df.insert(len(Patient_df.columns), "age", age)
        asa = perso_data.loc[perso_data['caseid'] == caseid, 'asa'].astype(int).item()
        Patient_df.insert(len(Patient_df.columns), "asa", asa)
        gender = (perso_data[perso_data['caseid'] == caseid]['sex'] == 'M').astype(int).item()  # F = 0, M = 1
        Patient_df.insert(len(Patient_df.columns), "gender", gender)
        weight = perso_data.loc[perso_data['caseid'] == caseid, 'weight'].astype(float).item()
        Patient_df.insert(len(Patient_df.columns), "weight", weight)
        height = perso_data.loc[perso_data['caseid'] == caseid, 'height'].astype(float).item()
        Patient_df.insert(len(Patient_df.columns), "height", height)
        bmi = perso_data.loc[perso_data['caseid'] == caseid, 'bmi'].astype(float).item()
        Patient_df.insert(len(Patient_df.columns), "bmi", bmi)
        preop_htn = perso_data.loc[perso_data['caseid'] == caseid, 'preop_htn'].astype(int).item() # hypertension
        Patient_df.insert(len(Patient_df.columns), "preop_hyper", preop_htn)
        preop_dm = perso_data.loc[perso_data['caseid'] == caseid, 'preop_dm'].astype(int).item() # diabète
        Patient_df.insert(len(Patient_df.columns), "preop_diab", preop_dm) 
        preop_ecg = perso_data.loc[perso_data['caseid'] == caseid, 'preop_ecg'].item()
        Patient_df.insert(len(Patient_df.columns), "preop_ecg", preop_ecg)
        preop_pft = perso_data.loc[perso_data['caseid'] == caseid, 'preop_pft'].item() # Fonction pulmonaire
        Patient_df.insert(len(Patient_df.columns), "preop_pft", preop_pft)
        preop_cr = perso_data.loc[perso_data['caseid'] == caseid, 'preop_cr'].astype(float).item() #  Créatinine
        Patient_df.insert(len(Patient_df.columns), "preop_crea", preop_cr > 2.0)
        optype = perso_data.loc[perso_data['caseid'] == caseid, 'optype'].item() # Type d'opération
        Patient_df.insert(len(Patient_df.columns), "optype", optype)
        department = perso_data.loc[perso_data['caseid'] == caseid, 'department'].item() # Departement
        Patient_df.insert(len(Patient_df.columns), "department", department)

        # Clean up
        Patient_df.drop(columns=['index'], inplace=True, errors='ignore')
        
        print(f"Successfully processed case {caseid}")
        return Patient_df
    
    except Exception as e:
        print(f"Error processing case {caseid}: {str(e)}")
        return None


def main():
    perso_data = pd.read_csv("./data/info_clinic_vitalDB.csv", decimal='.')

    with open('./data/caseid_list.txt', 'r') as f:
        caselist = f.read().splitlines()

    caselist = [int(i) for i in caselist]

    # Import cases
    print("Loading cases from VitalDB...")
    cases = load_cases(['BIS/BIS', 'Orchestra/PPF20_RATE', 'Orchestra/RFTN20_RATE',
                        'Solar8000/NIBP_MBP', 'Solar8000/NIBP_DBP',
                        'Solar8000/NIBP_SBP', 'Solar8000/PLETH_HR',
                        'Solar8000/PLETH_SPO2', 'Solar8000/ETCO2',
                        'Orchestra/PPF20_CT', 'Orchestra/RFTN20_CT','Solar8000/BT','Solar8000/RR_CO2',
                        'Orchestra/PPF20_VOL','Orchestra/RFTN20_VOL'
                        ], caseids=caselist)

    # Rename columns
    cases.rename(columns={'BIS/BIS': 'BIS',
                        'Orchestra/PPF20_RATE': 'Propofol',
                        'Orchestra/RFTN20_RATE': "Remifentanil",
                        'Orchestra/PPF20_CT':'Target_Propo',
                        'Orchestra/RFTN20_CT':'Target_Remi',
                        'Orchestra/PPF20_VOL':'Vol_Propo',
                        'Orchestra/RFTN20_VOL':'Vol_Remi',
                        'Solar8000/NIBP_MBP': 'NI_MAP',
                        'Solar8000/NIBP_DBP': 'NI_DBP',
                        'Solar8000/NIBP_SBP': 'NI_SBP',
                        'Solar8000/PLETH_HR': "HR",
                        'Solar8000/PLETH_SPO2': "SpO2",
                        'Solar8000/ETCO2': "EtCO2",
                        'Solar8000/BT': "Body_Temp",
                        'Solar8000/RR_CO2': 'RR',
                        'Solar8000/ART_MBP' : 'MBP'
                    }, inplace=True)

    print("Preparing data for parallel processing...")
    patient_groups = list(cases.groupby('caseid'))
    
    n_processes = min(cpu_count(), len(patient_groups)) 
    print(f"Using {n_processes} processes for {len(patient_groups)} patients")


    process_func = partial(process_patient, 
                          perso_data=perso_data)


    print("Starting parallel processing...")
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_func, patient_groups)

    successful_results = [result for result in results if result is not None]
    print(f"Successfully processed {len(successful_results)} out of {len(patient_groups)} patients")

    if successful_results:
        print("Combining results...")
        Full_data = pd.concat(successful_results, ignore_index=True)
        
        print("Saving results...")
        Full_data.to_csv("./data/full_data.csv", index=False)
        print(f"Dataset saved with {len(Full_data)} rows")
    else:
        print("No patients were successfully processed!")

if __name__ == "__main__":
    main()
