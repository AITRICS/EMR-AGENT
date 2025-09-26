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

def dataframe_from_csv(path, file, compress='infer'):
    """ 
    Read a CSV file and return it as a pandas DataFrame. 
    """
    return pd.read_csv(os.path.join(path, f"{file}.csv.gz"), compression=compress)

def generate_los(patient):
    """ 
    Calculate Length of Stay (LOS, ICU_LOS) for patients based on TimeOfStay. 
    Convert seconds to hours. (unit: s -> h)

    [TimeOfStay] Time of stay(seconds)
      * Comment: Time from primary metavision admission, to last discharge. In cases, where the patient is admitted to ward after surgery, this will include surgery time.
    [ICUOffset] Time of actual ICU admission 
      * Comment: SICdb includes preceding surgery if applicable, this field indicates the first transfer to an intensive or intermediate care ward.
    """
    patient['los'] = round((patient['TimeOfStay'] - patient['ICUOffset']) / 60 / 60, 4)
    patient['los_hospital'] = round(patient['TimeOfStay'] / 60 / 60, 4)
    return patient

def generate_gender(patient):
    """
    Mapping names to ID values from the 'd_reference' table
    """
    patient['gender'] = patient['Sex'].replace({735: 'Male', 736: 'Female'})
    return patient

def generate_mortality(patient):
    """
    Mortality type : ('Alive', 'Dead')
    """
    patient['mortality'] = np.where((~patient['OffsetOfDeath'].isna()).astype(int)==1, 'Dead', 'Alive')
    return patient

# Filter functions
def filter_patients_on_age(patient, min_age=18, max_age=np.inf):
    """
    Filter on specific age patients
    """
    patient = patient.loc[(patient.AgeOnAdmission >= min_age) & (patient.AgeOnAdmission <= max_age)]
    return patient

def filter_patients_on_gender(patient, select='Male'):
    """
    Filter on specific gender patients
    """
    if not isinstance(select, list):
        select = [select]
        
    patient = patient[patient['gender'].isin(select)]
    return patient

def filter_los_exists(patient):
    """
    Filter based on whether length of stay exists
    """
    patient = patient[patient['los'].notnull()]
    return patient

def filter_patients_minimum_los(patient, min_hour=1):
    """
    Filter on ICU stays with at least n hours duration
    """
    patient = patient[patient['los'] >= min_hour]
    return patient

def filter_one_unit_stay(patient):
    """
    Filter those having just one stay in unit
    """
    cohort_count = patient.groupby('PatientID').size().reset_index(name='icu_count')
    cohort_ids = cohort_count[cohort_count['icu_count'] == 1]['PatientID']
    patient = patient[patient['PatientID'].isin(cohort_ids)]
    return patient

def filter_patient_lab_exists(patient, lab, mapping_table, name=["Hemoglobin [Mass/volume] in Arterial blood"], record_n=1):
    """
    Filter based on exists at least one clinical recrod of any name in the list.
    Names in the list are treated as OR conditions.
    """

    if not isinstance(name, list):
        name = [name]

    # empty list
    all_filter_idx = []
    for single_name in name:
        if single_name not in mapping_table:
            raise ValueError(f"'{single_name}' does not exist in the mapping_table.")
        
        map_dict = mapping_table[single_name]['sources']['sicdb'][0]
        col_name = 'LaboratoryID' # Replace with hard-coding: map_dict['column'] = laboratoryid -> LaboratoryID
        value_name = map_dict['idx']
        if not isinstance(value_name, list):
            value_name = [value_name]
    
        # filtering
        pidx = patient.CaseID.unique()
        filter_lab = lab[(lab[col_name].isin(value_name)) & (lab['CaseID'].isin(pidx))]
        filter_idx = filter_lab.groupby(['CaseID',col_name]).size().reset_index(name='cnt').query("cnt >= @record_n").CaseID.unique()
        
        all_filter_idx.extend(filter_idx)
    
    # distinct
    all_filter_idx = list(set(all_filter_idx))

    # result
    patient = patient[patient['CaseID'].isin(all_filter_idx)]
    return patient

### ----- Add: result check ----- #####
# population sample size
def report_case(population:pd.DataFrame, step:str = 'Start'):
    print("[SICdb database(v1.0.8)]")
    print(f"\n# ===== {step} ===== #")
    print(f"Patients   : {population.PatientID.nunique():,}")
    print(f"ICU stays  : {population.CaseID.nunique():,}")

def save_to_csv(population, path, filename='file'):
    """
    Save the result to a csv file. (Added code)
    """
    population = population[['CaseID','PatientID','gender','AgeOnAdmission','mortality','los','los_hospital']]
    
    print(f"\n# ===== Summary and Save ===== #")
    print(f"Patients     : {population.PatientID.nunique():,}")
    print(f"ICU stays    : {population.CaseID.nunique():,}\n")
    print(f"[Gender]\n{population.gender.value_counts()}\n")
    print(f"[Age]\n{population.AgeOnAdmission.describe()}\n")
    print(f"[Mortality]\n{population.mortality.value_counts()}\n")    
    print(f"[Lenght of Stay (hours)]\n{population.los.describe()}\n")    
    
    return population.to_csv(os.path.join(path, f"{filename}.csv"), index=False)
