import numpy as np
import pandas as pd
import json
import os
import gzip

def load_paths():
    """
    Load path settings from JSON file.
    """
    with open('path.json', 'r') as file:
        paths = json.load(file)
    return paths

def read_mapping_table(path):
    """
    Load the feature mapping table.
    """
    with open(f'{path}/mapping/mapping_dictionary.json', 'r') as file:
        mapping_table = json.load(file)
    return mapping_table

def dataframe_from_csv(path, header=0, index_col=0, compress='gzip'):
    """ 
    Read a CSV file and return it as a pandas DataFrame. 
    """
    return pd.read_csv(path, header=header, index_col=index_col, compression=compress)

def read_patients_table(mimic3_path):
    """
    Load the patient table.
    """
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv.gz'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    # --- Code Add: GENDER type: 'M','F' -> 'Male', 'Female'
    pats.GENDER = np.where(pats.GENDER == 'M', 'Male', 'Female')
    return pats


def read_admissions_table(mimic3_path):
    """
    Load the admission table.
    """
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv.gz'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits

def read_icustays_table(mimic3_path):
    """
    Load the icustay table.
    """
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv.gz'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    # --- Code change: Convert days to hours.
    stays.LOS = round(stays.LOS * 24, 4)
    return stays

def read_events_table(mimic3_path, remove_null=True):
    """
    Load the labevents table.
    """
    events = dataframe_from_csv(os.path.join(mimic3_path, 'LABEVENTS.csv.gz'))
    if remove_null:
        events = events.loc[events.VALUE.notnull()]
        
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    return events

def remove_icustays_with_transfers(stays):
    """
    Benchmark - Exclusion 1: ICU transfers
    
     - Remove ICU transfers.
    """
    stays = stays.loc[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]

def merge_on_subject_admission(table1, table2):
    """
    Benchmark - Exclusion 2: Preprocessing for "2+ ICU stays per admission"
    
     - Merge the icustays and admission tables.
    """
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

def merge_on_subject(table1, table2):
    """
    Benchmark - Exclusion 2: Preprocessing for "2+ ICU stays per admission"
    
     - Merge the icustays and patient tables.
    """
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])

def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    """
    Benchmark - Exclusion 2: 2+ ICU stays per admission

     - Remove multiple stays per admission.
    """
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep.loc[(to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays

def add_age_to_icustays(stays):
    """
    Benchmark - Exclusion 3: Preprocessing for "Pediatric patients"

     - Calculate the age.
      * Fix for "OverflowError: Overflow in int64 addition" when calculating age using the GitHub code. 
      * The overflow happens because the timestamp calculations exceed int64 limits.
    """
    # stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    stays['AGE'] = (pd.to_datetime(stays.INTIME).astype(int) - pd.to_datetime(stays.DOB).astype(int)) /60./60/24/365/1e9
    stays['AGE'] = stays['AGE'].astype(int)
    stays.loc[stays.AGE < 0, 'AGE'] = 90
    return stays

def add_inhospital_mortality_to_icustays(stays):
    """
    Benchmark - Exclusion 3: Preprocessing for "Pediatric patients"

     - Add in-hospital mortality information.
    """
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    # --- Code change: Mortality type: int(0, 1) -> str('Alive', 'Dead')
    stays['MORTALITY'] = np.where(mortality.astype(int)==1, 'Dead', 'Alive')
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays

def add_inunit_mortality_to_icustays(stays):
    """
    Benchmark - Exclusion 3: Preprocessing for "Pediatric patients"

     - Add in-unit mortality information.
    """
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    # --- Code change: Mortality type: int(0, 1) -> str('Alive', 'Dead')
    stays['MORTALITY_INUNIT'] = np.where(mortality.astype(int)==1, 'Dead', 'Alive')
    return stays

def filter_icustays_on_age(patient, min_age=18, max_age=np.inf):
    """
    Benchmark - Exclusion 3: Pediatric patients

     - Remove patients age < 18 and > max(90).
    """
    patient = patient.loc[(patient.AGE >= min_age) & (patient.AGE <= max_age)]
    return patient


# Add functions
def filter_patients_on_gender(patient, select='Male'):
    """
    Filter on specific gender patients
    """
    # patient = patient[patient['GENDER'] == select]
    if not isinstance(select, list):
        select = [select]
        
    patient = patient[patient['GENDER'].isin(select)]
    return patient

def filter_los_exists(patient):
    """
    Filter based on whether length of stay exists
    """
    patient = patient[patient['LOS'].notnull()]
    return patient

def filter_patients_minimum_los(patient, min_hour=1):
    """
    Filter on ICU stays with at least n hours duration
    """
    patient = patient[patient['LOS'] >= min_hour]
    return patient

def filter_one_unit_stay(patient):
    """
    Filter those having just one stay in unit
    """
    cohort_count = patient.groupby('SUBJECT_ID').size().reset_index(name='icu_count')
    cohort_ids = cohort_count[cohort_count['icu_count'] == 1]['SUBJECT_ID']
    patient = patient[patient['SUBJECT_ID'].isin(cohort_ids)]
    return patient

def filter_patient_lab_exists(patient, lab, map_tb, name=["Hemoglobin [Mass/volume] in Arterial blood"], record_n=1):
    """
    Filter based on exists at least one clinical recrod of any name in the list.
    Names in the list are treated as OR conditions.
    """

    if not isinstance(name, list):
        name = [name]

    # empty list
    all_filter_idx = []
    
    for single_name in name:
        if single_name not in map_tb:
            raise ValueError(f"'{single_name}' does not exist in the mapping_table.")
        
        map_dict = map_tb[single_name]['sources']['mimic3'][0]
        col_name = map_dict['column'].upper()
        value_name = map_dict['idx']
        if not isinstance(value_name, list):
            value_name = [value_name]
    
        # filtering (lab & date & record count)
        filter_lab = pd.merge(lab[(lab[col_name].isin(value_name))], patient[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME']], how='inner', on=['SUBJECT_ID','HADM_ID'])
        filter_idx = filter_lab.query("ADMITTIME <= CHARTTIME <= DISCHTIME").groupby(['SUBJECT_ID','HADM_ID',col_name]).size().reset_index(name='cnt').query("cnt >= @record_n")
        filter_idx2 = filter_idx[['SUBJECT_ID','HADM_ID']].drop_duplicates()
        all_filter_idx.append(filter_idx2)
        
    # Distinct
    all_filter_idx_df = pd.concat(all_filter_idx, ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    # result
    patient = pd.merge(patient, all_filter_idx_df, how='inner', on=['SUBJECT_ID','HADM_ID'])
    return patient


### ----- Add: result check ----- #####
# population sample size
def report_case(population:pd.DataFrame, step:str = 'Start'):
    print("[MIMIC-III database(v1.4)]")
    print(f"\n# ===== {step} ===== #")
    print(f"Patients   : {population.SUBJECT_ID.nunique():,}")
    print(f"Admissions : {population.HADM_ID.nunique():,}")
    print(f"ICU stays  : {population.ICUSTAY_ID.nunique():,}")


def save_to_csv(population, path, filename='file'):
    """
    Save the result to a csv file. (Added code)
    """
    population = population[['SUBJECT_ID','HADM_ID','ICUSTAY_ID','GENDER','AGE','DEATHTIME','MORTALITY_INUNIT','MORTALITY','MORTALITY_INHOSPITAL','LOS']]
    
    print(f"\n# ===== Summary and Save ===== #")
    print(f"Patients     : {population.SUBJECT_ID.nunique():,}")
    print(f"Admissions   : {population.HADM_ID.nunique():,}")
    print(f"ICU stays    : {population.ICUSTAY_ID.nunique():,}\n")
    print(f"[Gender]\n{population.GENDER.value_counts()}\n")
    print(f"[Age]\n{population.AGE.describe()}\n")
    print(f"[Mortality]\n{population.MORTALITY.value_counts()}\n")    
    print(f"[Lenght of Stay (hours)]\n{population.LOS.describe()}\n")    
    
    return population.to_csv(os.path.join(path, f"{filename}.csv"), index=False)