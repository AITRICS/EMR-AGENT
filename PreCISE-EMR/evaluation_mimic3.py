""" MIMIC-III Evaluation
This code reuses code used in the MIMIC-III Benchmarks.
[GitHub] https://github.com/YerevaNN/mimic3-benchmarks/tree/v1.0.0-alpha

The benchmark uses MIMIC-III dataset version v1.3, which differs in data counts from the currently distributed v1.4.
For consistency in research comparisons, we use the results extracted using the original published GitHub code as reference.

The code reuses only the necessary parts of tasks from the following sources:
 - mimic3benchmark/mimic3csv.py
 - extract_subjects.py
"""
import numpy as np
import pandas as pd
import json
import os
import gzip

from func.mimic3 import *

paths=load_paths()
paths

# Save folder setting
os.makedirs(f"{paths['path_evaluation']}/evaluation_set", exist_ok=True, mode=0o777)
path_eval = f"{paths['path_evaluation']}/evaluation_set/"

### [Reliability] ###
# Load the data
patients = read_patients_table(paths['path_source_mimic3'])
admits = read_admissions_table(paths['path_source_mimic3'])
stays = read_icustays_table(paths['path_source_mimic3'])
report_case(stays)

## --- [Exclusion criteria 1] icu stansfers
stays = remove_icustays_with_transfers(stays)
report_case(stays, step='Exclusion 1: ICU transfers')
# Save the result for evaluation after applying exclusion criteria (1)
save_stays = merge_on_subject_admission(stays, admits)
save_stays = merge_on_subject(save_stays, patients)
save_stays = add_age_to_icustays(save_stays)
save_stays = add_inunit_mortality_to_icustays(save_stays)
save_stays = add_inhospital_mortality_to_icustays(save_stays)
save_to_csv(save_stays, path_eval, filename='reliability_mimic3_cohort1')

## --- [Exclusion criteria 2] 2+ icu stays per admission
stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
stays = filter_admissions_on_nb_icustays(stays)
report_case(stays, step='Exclusion 2: 2+ ICU stays per admission')
# Save the result for evaluation after applying exclusion criteria (2)
save_stays = add_age_to_icustays(stays)
save_stays = add_inunit_mortality_to_icustays(save_stays)
save_stays = add_inhospital_mortality_to_icustays(save_stays)
save_to_csv(save_stays, path_eval, filename='reliability_mimic3_cohort2')

## --- [Exclusion criteria 3] pediatric patients
stays = add_age_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)
report_case(stays, step='Exclusion 3: Pediatric patients (AGE < 18)')
# Save the result for evaluation after applying exclusion criteria (3)
save_to_csv(stays, path_eval, filename='reliability_mimic3_cohort3')

# reset
del patients, admits, stays

### [Harmonizability] ###
# Load the data
patients = read_patients_table(paths['path_source_mimic3'])
admits = read_admissions_table(paths['path_source_mimic3'])
stays = read_icustays_table(paths['path_source_mimic3'])

# merge demographic information
stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
stays = add_age_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
report_case(stays)

# Laboratory & Feature Mapping data ###
print(f"\nLoading laboratory and feature mapping data...\n")
labs = read_events_table(paths['path_source_mimic3'])
mapping_table=read_mapping_table(paths['path_evaluation'])


## --- [Exclusion criteria 1] Include only Age 19 to 29 and Include only Male and Exclude ICU stays with missing discharge time
patients = filter_icustays_on_age(stays, min_age=19, max_age=29)
report_case(patients, step='Age: Include Age 19 to 29')
patients = filter_patients_on_gender(patients, select='Male')
report_case(patients, step='Gender: Include only Male')
patients = filter_los_exists(patients)
report_case(patients, step='Filter: Exclude ICU stays with missing discharge time')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_mimic3_criteria1')

## --- [Exclusion criteria 2] Include only Age 61 to 69 and Include only Female and Include only ICU stays with at least 30 hours duration
patients = filter_icustays_on_age(stays, min_age=61, max_age=69)
report_case(patients, step='Age: Include Age 61 to 69')
patients = filter_patients_on_gender(patients, select='Female')
report_case(patients, step='Gender: Include only Female')
patients = filter_patients_minimum_los(patients, min_hour=30)
report_case(patients, step='Filter: Include only ICU stays with at least 30 hours duration')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_mimic3_criteria2')

## --- [Exclusion criteria 3] Include only Age 70 to 89 and Include only Male and Exclude stay with multiple ICU stays
patients = filter_icustays_on_age(stays, min_age=70, max_age=89)
report_case(patients, step='Age: Include Age 70 to 89')
patients = filter_patients_on_gender(patients, select='Male')
report_case(patients, step='Gender: Include only Male')
patients = filter_one_unit_stay(patients)
report_case(patients, step='Filter: Exclude stay with multiple ICU stays')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_mimic3_criteria3')

## --- [Exclusion criteria 4] Include only ICU stays from patients aged 20 to 30 and Exclude patient with missing gender information and Include both Female and Male patients
patients = filter_icustays_on_age(stays, min_age=20, max_age=30)
report_case(patients, step='Age: Include Age 20 to 30')
patients = filter_patients_on_gender(patients, select=['Female','Male'])
report_case(patients, step='Gender: Include gender without missing information')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_mimic3_criteria4')

## --- [Exclusion criteria 5] Include only ICU stays from patients aged 40 to 55 and include ICU stays which contains at least one clinical recrod of {"Hemoglobin [Mass/volume] in Arterial blood"}
patients = filter_icustays_on_age(stays, min_age=40, max_age=55)
report_case(patients, step='Age: Include Age 40 to 55')
patients = filter_patient_lab_exists(patients, labs, mapping_table, name='Hemoglobin [Mass/volume] in Arterial blood', record_n=1)
report_case(patients, step='Lab: Include at least one clinical record of Hemoglobin [Mass/volume] in Arterial blood')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_mimic3_criteria5')

## --- [Exclusion criteria 6] Include only ICU stays from patients aged 19 to 30 and Include only Male patients and include stays which contains at least 15 clinical recrod of {"Bicarbonate [Moles/volume] in Arterial blood"}
patients = filter_icustays_on_age(stays, min_age=19, max_age=30)
report_case(patients, step='Age: Include Age 19 to 30')
patients = filter_patients_on_gender(patients, select='Male')
report_case(patients, step='Gender: Include only Male')
patients = filter_patient_lab_exists(patients, labs, mapping_table, name='Bicarbonate [Moles/volume] in Arterial blood', record_n=15)
report_case(patients, step='Lab: Include at least 15 clinical record of Bicarbonate [Moles/volume] in Arterial blood')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_mimic3_criteria6')

## --- [Exclusion criteria 7] Include only ICU stays from patients aged 55 to 70 and include ICU stays which contains at least one clinical recrod of {"Lactate [Mass/volume] in Arterial blood" or "Methemoglobin/Hemoglobin.total in Arterial blood"}
patients = filter_icustays_on_age(stays, min_age=55, max_age=70)
report_case(patients, step='Age: Include Age 55 to 70')
patients = filter_patient_lab_exists(patients, labs, mapping_table, name=["Lactate [Mass/volume] in Arterial blood", "Methemoglobin/Hemoglobin.total in Arterial blood"], record_n=1)
report_case(patients, step='Lab: Include at least 1 clinical record of Lactate [Mass/volume] in Arterial blood or Methemoglobin/Hemoglobin.total in Arterial blood')
# Save the result
save_to_csv(patients, path=path_eval, filename='harmonizability_mimic3_criteria7')

print("Done.")