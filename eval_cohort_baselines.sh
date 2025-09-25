#!/bin/bash

folder_name="bs_react_cohort"
framework_name="BS_REACT"
Generated_Cohort_Path="/home/destin/LLM4EMR_DB_TEMP/LLM4EMR_DB_CMA/results/${folder_name}/"
Test_Data_Path="./test_data"

model_name="claude-3-5-sonnet-20240620"
guide=True
obs=True
fb=True
# Piordb_only Piordb_and_manual_and_prior
hint=Piordb_and_manual_and_prior
Generated_Cohorts_Features=(
    ${framework_name}_eicu_criteria1_DBeicu_${hint}_LLM${model_name}
    ${framework_name}_eicu_criteria2_DBeicu_${hint}_LLM${model_name}
    ${framework_name}_eicu_criteria3_DBeicu_${hint}_LLM${model_name}
    ${framework_name}_eicu_criteria4_1_DBeicu_${hint}_LLM${model_name}
    ${framework_name}_eicu_criteria4_2_DBeicu_${hint}_LLM${model_name}
    ${framework_name}_eicu_criteria4_3_DBeicu_${hint}_LLM${model_name}
    ${framework_name}_eicu_criteria4_4_DBeicu_${hint}_LLM${model_name}
    ${framework_name}_mimic3_criteria1_DBmimic3_${hint}_LLM${model_name}
    ${framework_name}_mimic3_criteria2_DBmimic3_${hint}_LLM${model_name}
    ${framework_name}_mimic3_criteria3_DBmimic3_${hint}_LLM${model_name}
    ${framework_name}_mimic3_criteria4_1_DBmimic3_${hint}_LLM${model_name}
    ${framework_name}_mimic3_criteria4_2_DBmimic3_${hint}_LLM${model_name}
    ${framework_name}_mimic3_criteria4_3_DBmimic3_${hint}_LLM${model_name}
    ${framework_name}_mimic3_criteria4_4_DBmimic3_${hint}_LLM${model_name}
    ${framework_name}_sicdb_criteria1_DBsicdb_${hint}_LLM${model_name}
    ${framework_name}_sicdb_criteria2_DBsicdb_${hint}_LLM${model_name}
    ${framework_name}_sicdb_criteria3_DBsicdb_${hint}_LLM${model_name}
    ${framework_name}_sicdb_criteria4_1_DBsicdb_${hint}_LLM${model_name}
    ${framework_name}_sicdb_criteria4_2_DBsicdb_${hint}_LLM${model_name}
    ${framework_name}_sicdb_criteria4_3_DBsicdb_${hint}_LLM${model_name}
    ${framework_name}_sicdb_criteria4_4_DBsicdb_${hint}_LLM${model_name}
)

Test_Data_Files=(
    "eicu_criteria1.csv"
    "eicu_criteria2.csv"
    "eicu_criteria3.csv"
    "eicu_criteria4_1.csv"
    "eicu_criteria4_2.csv"
    "eicu_criteria4_3.csv"
    "eicu_criteria4_4.csv"
    "mimic3_criteria1.csv"
    "mimic3_criteria2.csv"
    "mimic3_criteria3.csv"
    "mimic3_criteria4_1.csv"
    "mimic3_criteria4_2.csv"
    "mimic3_criteria4_3.csv"
    "mimic3_criteria4_4.csv"
    "sicdb_criteria1.csv"
    "sicdb_criteria2.csv"
    "sicdb_criteria3.csv"
    "sicdb_criteria4_1.csv"
    "sicdb_criteria4_2.csv"
    "sicdb_criteria4_3.csv"
    "sicdb_criteria4_4.csv"
)

Column_Set_list=(
    '["patientunitstayid","gender","age","los"]'
    '["patientunitstayid","gender","age","mortality"]'
    '["patientunitstayid","gender","age","mortality"]'
    '["patientunitstayid","gender","age","los"]'
    '["patientunitstayid","gender","age","mortality"]'
    '["patientunitstayid","gender","age","mortality"]'
    '["patientunitstayid","gender","age","mortality"]'
    '["ICUSTAY_ID","GENDER","AGE","LOS"]'
    '["ICUSTAY_ID","GENDER","AGE","MORTALITY"]'
    '["ICUSTAY_ID","GENDER","AGE","MORTALITY"]'
    '["ICUSTAY_ID","GENDER","AGE","LOS"]'
    '["ICUSTAY_ID","GENDER","AGE","MORTALITY"]'
    '["ICUSTAY_ID","GENDER","AGE","MORTALITY"]'
    '["ICUSTAY_ID","GENDER","AGE","MORTALITY"]'
    '["CaseID","gender","AgeOnAdmission","los"]'
    '["CaseID","gender","AgeOnAdmission","mortality"]'
    '["CaseID","gender","AgeOnAdmission","mortality"]'
    '["CaseID","gender","AgeOnAdmission","los"]'
    '["CaseID","gender","AgeOnAdmission","mortality"]'
    '["CaseID","gender","AgeOnAdmission","mortality"]'
    '["CaseID","gender","AgeOnAdmission","mortality"]'
)


for i in "${!Generated_Cohorts_Features[@]}"; do
    Generated_Cohort_Feature="${Generated_Cohorts_Features[$i]}"
    Test_Data_File="${Test_Data_Files[$i]}"
    Column_Set="${Column_Set_list[$i]}"
    python ./cohort_evaluation.py \
        --generated-cohort-path "$Generated_Cohort_Path" \
        --test-data-path "$Test_Data_Path" \
        --generated-cohort-feature "$Generated_Cohort_Feature" \
        --test-data-file "$Test_Data_File" \
        --column-set "$Column_Set"
done

Generated_Cohort_Path="/home/destin/LLM4EMR_DB_TEMP/LLM4EMR_DB_CMA/results/${folder_name}/"
expected_sampple_num=70
DB_Name="mimic3"
python ./cohort_evaluation.py --generated-cohort-path ${Generated_Cohort_Path} --db-name ${DB_Name} --expected-sample-num ${expected_sampple_num}

Generated_Cohort_Path="/home/destin/LLM4EMR_DB_TEMP/LLM4EMR_DB_CMA/results/${folder_name}/"
expected_sampple_num=70
DB_Name="eicu"
python ./cohort_evaluation.py --generated-cohort-path ${Generated_Cohort_Path} --db-name ${DB_Name} --expected-sample-num ${expected_sampple_num}

Generated_Cohort_Path="/home/destin/LLM4EMR_DB_TEMP/LLM4EMR_DB_CMA/results/${folder_name}/"
expected_sampple_num=70
DB_Name="sicdb"
python ./cohort_evaluation.py --generated-cohort-path ${Generated_Cohort_Path} --db-name ${DB_Name} --expected-sample-num ${expected_sampple_num}