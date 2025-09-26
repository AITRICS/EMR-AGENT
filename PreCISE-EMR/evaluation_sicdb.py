""" SICdb Evaluation
"""
import numpy as np
import pandas as pd
import json
import os
import gzip

from func.sicdb import *

paths=load_paths()
paths

# Save folder setting
os.makedirs(f"{paths['path_evaluation']}/evaluation_set", exist_ok=True, mode=0o777)
path_eval = f"{paths['path_evaluation']}/evaluation_set/"


### [Harmonizability] ###
# Load the data
df_patient = dataframe_from_csv(paths['path_source_sicdb'], 'cases', compress='gzip')

# merge demographic information
df_patient = generate_gender(df_patient)
df_patinet = generate_los(df_patient)
df_patinet = generate_mortality(df_patinet)
report_case(df_patient)

# Laboratory & Feature Mapping data ###
print(f"\nLoading laboratory and feature mapping data...\n")
labs = dataframe_from_csv(paths['path_source_sicdb'], 'laboratory', compress='gzip')
mapping_table=read_mapping_table(paths['path_evaluation'])


## --- [Exclusion criteria 1] Include only Age 19 to 29 and Include only Male and Exclude ICU stays with missing discharge time
patients = filter_patients_on_age(df_patient, min_age=19, max_age=29)
report_case(patients, step='Age: Include Age 19 to 29')
patients = filter_patients_on_gender(patients, select='Male')
report_case(patients, step='Gender: Include only Male')
patients = filter_los_exists(patients)
report_case(patients, step='Filter: Exclude ICU stays with missing discharge time')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_sicdb_criteria1')

## --- [Exclusion criteria 2] Include only Age 61 to 69 and Include only Female and Include only ICU stays with at least 30 hours duration
patients = filter_patients_on_age(df_patient, min_age=61, max_age=69)
report_case(patients, step='Age: Include Age 61 to 69')
patients = filter_patients_on_gender(patients, select='Female')
report_case(patients, step='Gender: Include only Female')
patients = filter_patients_minimum_los(patients, min_hour=30)
report_case(patients, step='Filter: Include only ICU stays with at least 30 hours duration')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_sicdb_criteria2')

## --- [Exclusion criteria 3] Include only Age 70 to 89 and Include only Male and Exclude stay with multiple ICU stays
patients = filter_patients_on_age(df_patient, min_age=70, max_age=89)
report_case(patients, step='Age: Include Age 70 to 89')
patients = filter_patients_on_gender(patients, select='Male')
report_case(patients, step='Gender: Include only Male')
patients = filter_one_unit_stay(patients)
report_case(patients, step='Filter: Exclude stay with multiple ICU stays')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_sicdb_criteria3')

## --- [Exclusion criteria 4] Include only ICU stays from patients aged 20 to 30 and Exclude patient with missing gender information and Include both Female and Male patients
patients = filter_patients_on_age(df_patient, min_age=20, max_age=30)
report_case(patients, step='Age: Include Age 20 to 30')
patients = filter_patients_on_gender(patients, select=['Female','Male'])
report_case(patients, step='Gender: Include gender without missing information')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_sicdb_criteria4')

## --- [Exclusion criteria 5] Include only ICU stays from patients aged 40 to 55 and include ICU stays which contains at least one clinical recrod of {"Hemoglobin [Mass/volume] in Arterial blood"}
patients = filter_patients_on_age(df_patient, min_age=40, max_age=55)
report_case(patients, step='Age: Include Age 40 to 55')
patients = filter_patient_lab_exists(patients, labs, mapping_table, name='Hemoglobin [Mass/volume] in Arterial blood', record_n=1)
report_case(patients, step='Lab: Include at least one clinical record of Hemoglobin [Mass/volume] in Arterial blood')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_sicdb_criteria5')

## --- [Exclusion criteria 6] Include only ICU stays from patients aged 19 to 30 and Include only Male patients and include stays which contains at least 15 clinical recrod of {"Bicarbonate [Moles/volume] in Arterial blood"}
patients = filter_patients_on_age(df_patient, min_age=19, max_age=30)
report_case(patients, step='Age: Include Age 19 to 30')
patients = filter_patients_on_gender(patients, select='Male')
report_case(patients, step='Gender: Include only Male')
patients = filter_patient_lab_exists(patients, labs, mapping_table, name='Bicarbonate [Moles/volume] in Arterial blood', record_n=15)
report_case(patients, step='Lab: Include at least 15 clinical record of Bicarbonate [Moles/volume] in Arterial blood')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_sicdb_criteria6')

## --- [Exclusion criteria 7] Include only ICU stays from patients aged 55 to 70 and include ICU stays which contains at least one clinical recrod of {"Lactate [Mass/volume] in Arterial blood" or "Methemoglobin/Hemoglobin.total in Arterial blood"}
patients = filter_patients_on_age(df_patient, min_age=55, max_age=70)
report_case(patients, step='Age: Include Age 55 to 70')
patients = filter_patient_lab_exists(patients, labs, mapping_table, name=["Lactate [Mass/volume] in Arterial blood", "Methemoglobin/Hemoglobin.total in Arterial blood"], record_n=1)
report_case(patients, step='Lab: Include at least 1 clinical record of Lactate [Mass/volume] in Arterial blood or Methemoglobin/Hemoglobin.total in Arterial blood')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_sicdb_criteria7')

print("Done.")