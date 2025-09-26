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
    ICU's Length of stay (LOS) = Convert minutes to hours. (unit: m -> h)
    """
    patient.loc[:, 'los'] = round(patient['unitdischargeoffset'] / 60, 4)
    return patient

def generate_mortality(patient):
    """
    Mortality type : ('Alive', 'Expired') -> ('Alive', 'Dead')
    """
    # --- Code change: Mortality value: str('Alive', 'Expired') -> str('Alive', 'Dead')
    patient.loc[:, 'mortality'] = np.where(patient['hospitaldischargestatus'].isna(), 'Unknown',
                                           np.where(patient['hospitaldischargestatus']=='Expired', 'Dead', 'Alive'))
    return patient

# Filter functions
def filter_patients_on_age(patient, min_age=18, max_age=89):
    """
    Filter on specific age patients
    """
    patient.loc[patient['age'] == '> 89', 'age'] = 90
    patient[['age']] = patient[['age']].fillna(-1)
    patient[['age']] = patient[['age']].astype(int)
    patient = patient.loc[(patient.age >= min_age) & (patient.age <= max_age)]
    return patient

def filter_patients_on_gender(patient, select='Male'):
    """
    Filter on specific gender patients
    """
    # patient = patient[patient['gender'] == select]

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

def filter_one_unit_stay(patients):
    """
    Filter those having just one stay in unit
    """
    cohort_count = patients.groupby(by='uniquepid').count()
    index_cohort = cohort_count[cohort_count['patientunitstayid'] == 1].index
    patients = patients[patients['uniquepid'].isin(index_cohort)]
    return patients

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
        
        map_dict = mapping_table[single_name]['sources']['eicu'][0]
        col_name = map_dict['column']
        value_name = map_dict['idx']
        if not isinstance(value_name, list):
            value_name = [value_name]
    
        # filtering
        pidx = patient.patientunitstayid.unique()
        filter_lab = lab[(lab[col_name].isin(value_name)) & (lab['patientunitstayid'].isin(pidx))]
        filter_idx = filter_lab.groupby(['patientunitstayid',col_name]).size().reset_index(name='cnt').query("cnt >= @record_n").patientunitstayid.unique()
        all_filter_idx.extend(filter_idx)
    
    # distinct
    all_filter_idx = list(set(all_filter_idx))

    # result
    patient = patient[patient['patientunitstayid'].isin(all_filter_idx)]
    return patient

### ----- Add: result check ----- #####
# population sample size
def report_case(population:pd.DataFrame, step:str = 'Start'):
    print("[eICU database(v2.0)]")
    print(f"\n# ===== {step} ===== #")
    print(f"Patients   : {population.uniquepid.nunique():,}")
    print(f"Admissions : {population.patienthealthsystemstayid.nunique():,}")
    print(f"ICU stays  : {population.patientunitstayid.nunique():,}")

def save_to_csv(patient, path, filename='file'):
    """
    Save the result to a csv file. (Added code)
    """
    patient = patient[['patientunitstayid','patienthealthsystemstayid','uniquepid','gender','age','mortality','los']]
    print(f"\n# ===== Summary and Save ===== #")
    print(f"Patients   : {patient.uniquepid.nunique():,}")
    print(f"Admissions : {patient.patienthealthsystemstayid.nunique():,}")
    print(f"ICU stays : {patient.patientunitstayid.nunique():,}\n")
    print(f"[Gender]\n{patient.gender.value_counts()}\n")
    print(f"[Age]\n{patient.age.describe()}\n")
    print(f"[Mortality]\n{patient.mortality.value_counts()}\n")
    print(f"[Lenght of Stay (hours)]\n{patient.los.describe()}\n")

    return patient.to_csv(f"{path}{filename}.csv", index=False)
