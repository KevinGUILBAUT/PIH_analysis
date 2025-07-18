"""Select cases for the database"""
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vitaldb
from os.path import exists
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

warnings.filterwarnings('ignore')


# Create folder for figures if it doesn't exist
os.makedirs('./figures', exist_ok=True)



def process_case(caseid, main_signals):
    """
    Fonction pour traiter un cas individuel
    """
    print(f"Processing case {caseid}")
    
    gaz_signals = ['Primus/INSP_DES', 'Primus/INSP_SEVO', 'Primus/FIN2O']
    intra_drug_signals = ['AMD_RATE', 'DEX2_RATE', 'DEX4_RATE', 'DOBU_RATE', 'DOPA_RATE', 'DTZ_RATE', 'EPI_RATE',
                          'FUT_RATE', 'MRN_RATE', 'NEPI_RATE', 'NPS_RATE', 'NTG_RATE', 'OXY_RATE', 'PGE1_RATE',
                          'PHEN_RATE', 'PPF20_RATE', 'RFTN20_RATE', 'RFTN50_RATE', 'ROC_RATE', 'VASO_RATE', 'VEC_RATE']
    intra_drug_signals = ['Orchestra/' + el for el in intra_drug_signals]
    intra_drug_signals += [string.replace('RATE', 'VOL') for string in intra_drug_signals]
    signals = main_signals + gaz_signals + intra_drug_signals
    
    try:
        if exists(f'./data/cases/Case_{caseid}.csv'):
            dataframe_case = pd.read_csv(f'./data/cases/Case_{caseid}.csv')
            dataframe_case = dataframe_case[[signal for signal in signals if signal in dataframe_case.columns]]
        else:
            case = vitaldb.VitalFile(caseid, signals)
            dataframe_case = case.to_pandas(signals, 1)

        # Check that there is only Propofol and Remifentanil
        count_drugs = dataframe_case[[col_name for col_name in dataframe_case.columns if col_name.startswith(
            'Orchestra/')]].sum(axis=0).astype(bool).sum()

        if count_drugs > 4:
            return (caseid, 'too_many_drugs')

        # Check that there is no gaz usage
        total_gaz_usage = dataframe_case[[
            col_name for col_name in dataframe_case.columns if col_name.startswith('Primus/')]].sum(axis=0).sum()
        if total_gaz_usage > 10:
            return (caseid, 'has_gazes')
            
        # check that the volume of Propofol and Remifentanil is 0 at the first non nan value of the case
        if dataframe_case.loc[dataframe_case['Orchestra/PPF20_RATE'].first_valid_index(), 'Orchestra/PPF20_RATE'] > 0:
            return (caseid, 'missing_volume')
        if dataframe_case.loc[dataframe_case['Orchestra/RFTN20_RATE'].first_valid_index(), 'Orchestra/RFTN20_RATE'] > 0:
            return (caseid, 'missing_volume')
        
        
        # find first drug injection
        dataframe_case['Orchestra/RFTN20_RATE'].fillna(method='bfill', inplace=True)
        dataframe_case['Orchestra/PPF20_RATE'].fillna(method='bfill', inplace=True)
        induction_start = 0

        for i in range(len(dataframe_case)):
            if dataframe_case.loc[i, 'Orchestra/PPF20_RATE'] != 0 or dataframe_case.loc[i, 'Orchestra/RFTN20_RATE'] != 0:
                induction_start = i
                break

        #check there is at least one value before induction
        ni_map_first_index = dataframe_case['Solar8000/NIBP_MBP'].first_valid_index()
        
        if ni_map_first_index is None or ni_map_first_index > induction_start:
            return (caseid,'no_NI_MAP_before_induc')
        
        intubation_start = None
        window_size = 150

        if 'Solar8000/RR_CO2' in dataframe_case.columns and not dataframe_case['Solar8000/RR_CO2'].isna().all():
            dataframe_case['Solar8000/RR_CO2'].replace(0, np.nan, inplace=True)
            dataframe_case['Solar8000/RR_CO2'].fillna(method='ffill', inplace=True)

            # Detect intubation based on RR_CO2 signal flatness
            """for i in range(induction_start, len(dataframe_case) - window_size):
                # Extraire la fenêtre de RR
                rr_window = dataframe_case.loc[i:i+window_size-1, 'Solar8000/RR_CO2']

                # Vérifier s'il y a des NaNs dans la fenêtre
                if rr_window.isna().any():
                    continue  # Passer à la fenêtre suivante

                rr_std = rr_window.std()
                #rr_mean = rr_window.mean()
                    
                # Considérer comme constant si l'écart-type absolu est très faible
                if (rr_std < 0.1):
                    intubation_start = i
                    break
"""
        if intubation_start is None:
            # Generally intubation occurs within 10–15 minutes after induction
            intubation_start = min(induction_start + 900, len(dataframe_case) - 1)  # 15 minutes après induction

        elif intubation_start == 0 :
            return (caseid, 'intub_0')
        
        Ncase = len(dataframe_case[induction_start:intubation_start])
        # check there is at least 80% of NI_MAP data between induction and estimated intubation (2s step time so there is already 50% of missing data)
        if dataframe_case['Solar8000/NIBP_MBP'][induction_start:intubation_start].fillna(0).eq(0).sum()/Ncase > 0.6:
            return (caseid, 'too_many_nan')

        return (caseid, 'valid')
        
    except Exception as e:
        print(f"Error processing case {caseid}: {e}")
        return (caseid, 'error')

if __name__ == '__main__':
    
    perso_data = pd.read_csv("./data/info_clinic_vitalDB.csv")
    perso_data.dropna(subset=["caseid"], inplace=True)
    print(f"Initial number of cases: {perso_data.count().iloc[0]}")

    # Select cases with at least 30 monutes of data
    perso_data = perso_data[perso_data["caseend"] >= 30]

    print(f"Number of cases with at least 30 minutes of data: {perso_data.count().iloc[0]}")

    # Select cases without bolus drugs impacting BIS and MAP
    perso_data = perso_data[perso_data["intraop_mdz"] == 0]
    perso_data = perso_data[perso_data["intraop_ftn"] == 0]
    perso_data = perso_data[perso_data["intraop_eph"] == 0]
    perso_data = perso_data[perso_data["intraop_phe"] == 0]
    perso_data = perso_data[perso_data["intraop_epi"] == 0]

    print(f"Number of cases without bolus drugs impacting BIS and MAP: {perso_data.count().iloc[0]}")

    # select cases including the main signals
    Main_signals = ['Solar8000/NIBP_MBP']

    caseid_with_signals = vitaldb.find_cases(Main_signals)

    perso_data = perso_data[perso_data["caseid"].astype(int).isin(caseid_with_signals)]

    print(f"Number of cases including the main signals: {perso_data.count().iloc[0]}")

    # Select TIVA cases

    caseid_tiva = list(vitaldb.caseids_tiva)
    perso_data = perso_data[perso_data["caseid"].astype(int).isin(caseid_tiva)]

    print(f"Number of TIVA cases: {perso_data.count().iloc[0]}")

    # select cases without gazes
    caseid_n2o = list(vitaldb.caseids_n2o)
    casei_sevo = list(vitaldb.caseids_sevo)
    caseid_des = list(vitaldb.caseids_des)

    caseid_gazes = caseid_n2o + casei_sevo + caseid_des
    perso_data = perso_data[~perso_data["caseid"].astype(int).isin(caseid_gazes)]

    print(f"Number of cases without gazes: {perso_data.count().iloc[0]}")

    print("Starting parallel processing of cases...")
    caseid_list = perso_data["caseid"].astype(int).tolist()

    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes")

    process_case_partial = partial(process_case, main_signals=Main_signals)

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_case_partial, caseid_list)

    number_of_cases_with_more_drugs = 0
    number_of_cases_with_gazes = 0
    number_of_cases_with_missing_volume = 0
    nan_values = 0
    no_ni_map = 0
    final_caseid_list = []

    for caseid, status in results:
        if status == 'valid':
            final_caseid_list.append(caseid)
        elif status == 'too_many_drugs':
            number_of_cases_with_more_drugs += 1
        elif status == 'has_gazes':
            number_of_cases_with_gazes += 1
        elif status == 'missing_volume':
            number_of_cases_with_missing_volume += 1
        elif status == 'too_many_nan':
            nan_values += 1
        elif status == 'no_NI_MAP_before_induc':
            no_ni_map += 1

    print(f"Number of cases with more than 2 drugs: {number_of_cases_with_more_drugs}")
    print(f"Number of cases with gazes: {number_of_cases_with_gazes}")
    print(f"Number of cases with missing volume: {number_of_cases_with_missing_volume}")
    print(f"Number of cases with missing NI_MAP before induc: {(no_ni_map)}")
    print(f"Number of cases with missing NI_MAP between induc and intub: {(nan_values)}")

    print(f"Number of cases with good signal quality: {len(final_caseid_list)}")

    perso_data = perso_data[perso_data["caseid"].astype(int).isin(final_caseid_list)]

    # save the list of caseid
    with open('./data/caseid_list.txt', 'w') as f:
        for item in final_caseid_list:
            f.write("%s\n" % item)


    # Plot on selected data

    # Check the number of cases per operation name
    perso_data_by_surgeon = perso_data.groupby("opname")
    perso_data_by_surgeon.count().plot(kind="bar", y="caseid")

    # Plot Patient Characteristics

    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    # Age
    perso_data["age"].astype(float).plot(kind="hist", bins=20, ax=ax[0, 0])
    ax[0, 0].set_title('Age')
    # Weight
    perso_data["weight"].astype(float).plot(kind="hist", bins=20, ax=ax[0, 1])
    ax[0, 1].set_title('Weight')
    # Height
    perso_data["height"].astype(float).plot(kind="hist", bins=20, ax=ax[1, 0])
    ax[1, 0].set_title('Height')
    # count 'F' and 'M' in sex

    sex_df = pd.DataFrame({'F': perso_data.sex.eq('F').sum(),
                        'M': perso_data.sex.eq('M').sum()}, index=[0, 1])

    sex_df.plot(kind="hist", ax=ax[1, 1])
    ax[1, 1].set_title('Gender')
    fig.savefig(f'../Vitaldb_database_2/figures/characteristics.png')
    plt.close(fig)
