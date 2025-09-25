from psycopg2 import connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from typing import List, Dict, Tuple, Optional, Any

import os
import time
import shutil
import logging
import datetime
import ast
import argparse
import pandas as pd
import numpy as np
import pickle
import anthropic

import torch
import torch.nn.functional as F

from utils import PostgreSQLConnector, sql_post_process

def setup_logging(results_dir):
    results_dir = results_dir + f"BS_SEQ_{args.case_name}_DB{args.target_database}_Pior{args.database_knowledge}_LLM{args.llm_model}"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    log_filename = f"{results_dir}/BS_SEQ_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s - %(levelname)s - %(message)s',
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),  # This will also print to console
            logging.StreamHandler()
        ]
    )
    
    # Log initial parameters
    logging.info("=== Starting New Run for baseline SEQ ===")
    logging.info(f"Target Database: {args.target_database}")
    logging.info(f"Trial Number: {args.trial_num}")
    logging.info(f"Case Number: {args.case_number}")
    logging.info(f"Case Name: {args.case_name}")
    logging.info(f"1. Cohort Selection: {args.cohort_selection}")
    logging.info(f"2. Requested Features: {args.requested_features}")
    logging.info(f"=== Ablation Variants ===")
    logging.info(f"LLM Model: {args.llm_model}")
    logging.info(f"Database Knowledge: {args.database_knowledge}")
    logging.info("=========================================")
    return results_dir



def llm_generation(prompt, model="claude-3-5-sonnet-20240620"):
    retry_count=5
    for attempt in range(retry_count):
        try:    
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0.0,
                top_p=1.0,
                messages=[
                    {"role": "user", "content": prompt}
                    ]
            )
            if message:
                break
        except anthropic.RateLimitError:
            print(f"ERROR: Claude API (retry {attempt+1}/{retry_count}) - {e}")
            time.sleep(30)
    return message.content[0].text.replace("\\n", "\n")

def get_cohort(results_dir, target_database, database_knowledge, requested_features, cohort_selection, llm_model):    
    table_schema_path = f"./models/data/db_info/{target_database}_schema.txt"
    question_all = []

    question_information = f"List all {requested_features.strip()} information. Ensure the output in PostgreSQL strictly follows the order and format specified in each () of {requested_features.strip()}."
    question_all.append(question_information)

    cohort_selection = cohort_selection.split('and')
    if type(cohort_selection) == list:
        for question_information in cohort_selection:
            question_information = f"Retrieve only the cases \"{question_information.strip()}\""
            question_all.append(question_information)
    else:
        question_information = f"Retrieve only the cases {question_information}"

    print(f"\nquestion_all: {question_all}")
    overview = None
    prior_knowledge_db_specific = None
    
    with open(table_schema_path, 'r', encoding="UTF-8") as file:
        sql_table_schema = file.read()
    
    if "manual" in database_knowledge:
        print("Get manual information")
        overview_path = f"./models/data/db_info/{target_database}_overview.txt"
        with open(overview_path, 'r', encoding="UTF-8") as file:
            overview = file.read()
    
    if "prior" in database_knowledge:
        print("Get database specific information")
        with open(f"./models/data/db_info/{target_database}_prior_knowledge_cohort.txt", 'r', encoding="UTF-8") as file:
            prior_knowledge_db_specific = file.read()

    pre_llm_api_result = {}
    example_number = args.example_number
    for question_number, question in enumerate(question_all):
        question_number += 1
        base_prompt = f"""Get only one PostgreSQL query as plain text. Do not include code delimiters (e.g., ```sql), comments, or any additional text.
-- Schema information
{sql_table_schema}"""
        if prior_knowledge_db_specific :
            base_prompt += f"-- Prior knowledge\n{prior_knowledge_db_specific}"
        if overview :
                base_prompt += f"-- Overview\n{overview}"
    
        base_prompt += """\n\n-- Post-processing Detail
Please note that:
1. Questions asking whether a specific number falls within a normal range can be formulated as follows and will be changed through the post_processing process.
NLQ: Had the value of result measured during result been normal?
SQL: SELECT COUNT(*)>0 FROM chartevents WHERE chartevents.icustay_id IN (... ) AND chartevents.valuenum BETWEEN sao2_lower AND sao2_upper 
2. Similarly, for questions that require the current time, we will use 'current_time' as a placeholder and adjust it as necessary. For reference, the current time is assumed to be "2105-12-31 23:59:00".Therefore, if there is the expression "this month" means 2105-12.
        
-- SQL-like Rep. Description
PREV_QUERY and PREV_RESULT tokens allow for referencing and reusing the SQL code and results of previous queries in subsequent ones.
The PREV_QUERY token is used to represent the SQL code of the previous query, essentially allowing the new query to build upon it or modify it. SQL queries can also start with the PREV_QUERY token, which enables the duplication and utilization of the previous query in the new one.
The PREV_RESULT token, on the other hand, is used to represent the example of result set from a previous query, rather than the query itself. This is useful when we want to use the results of a previous query directly within a new query.
        
-- TEST_QUESTION"""

        if (question_number-1)>0:
            for pre_question_number in range(question_number-1):
                pre_question_number += 1                
                test_prompt = f"""{str(pre_llm_api_result[pre_question_number].replace("SQL","PREV_QUERY").replace("DATA","PREV_RESULT"))}\nNLQ{str(question_number)}: {str(question)} in PREV_QUERY{question_number-1}.
SQL{str(question_number)}:"""

        else:
            test_prompt = f"""\nNLQ{str(question_number)}:{str(question)}.\nSQL{str(question_number)}:"""
        llm_api_answer = llm_generation(f"{base_prompt}\n{test_prompt}", model=args.llm_model)
        print(f"llm_api_answer: {llm_api_answer}")
        cohort_feature_sql = sql_post_process(llm_api_answer)
        try:
            cohort_feature = db_connector.connect(cohort_feature_sql)
            pre_llm_api_result[question_number] = test_prompt + cohort_feature_sql + f"\nDATA{question_number}:" + str(cohort_feature[:example_number])
            print(str(pre_llm_api_result[question_number]))
            return cohort_feature_sql, cohort_feature, question_number
        except Exception as e:
            print(f"\nError executing SQL : {str(e)}")
            return cohort_feature_sql, None, question_number

def cohort_extraction(args, results_dir):
    target_database = args.target_database
    database_knowledge = args.database_knowledge
    requested_features = args.requested_features
    cohort_selection = args.cohort_selection
    llm_model = args.llm_model
    trial_num = args.trial_num
    start_time = time.time()
    for trial_index in range(trial_num):
        cohort_feature_sql, cohort_feature, api_run_count = get_cohort(results_dir, target_database, database_knowledge, requested_features, cohort_selection, llm_model)
        if cohort_feature is None:
            logging.info("=========== Results ================")
            logging.info(f"Failed to generate valid SQL after maximum retries")
            logging.info(f"cohort feature SQL: {cohort_feature_sql}")
        else:
            logging.info("=========== Results ================")
            logging.info(f"Cohort Feature SQL Generation Completed")
            logging.info(f"Timeseries Feature SQL: {cohort_feature_sql}")
            logging.info(f"api run count: {api_run_count}")
            logging.info(f"cohort feature results: {cohort_feature[:20]}")
            logging.info(f"len(cohort_feature): {len(cohort_feature)}")
        
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Time taken: {duration} seconds")
        
        output_file_path = results_dir + f"/BS_SEQ_cohort_output{trial_index}.pkl"

        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        final_result = {
            "target_database": args.target_database,
            "trial_number": args.trial_num,
            "case_number": args.case_number,
            "cohort_selection": args.cohort_selection,
            "requested_features": args.requested_features,
            "LLM_model": args.llm_model,
            "database_knowledge": args.database_knowledge,
            "api_run_count": api_run_count,
            "cohort_feature_sql": cohort_feature_sql,
            "cohort_feature": cohort_feature,
            "duration": duration,
            "results_dir": results_dir
        }
        with open(output_file_path, 'wb') as f:
            pickle.dump(final_result, f)
        logging.info(f"Requested features saved to: {output_file_path}")
        logging.info("====================================")

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-number', type=int, default=20)
    parser.add_argument('--agent-part', type=str, default="cohort", choices = ["cohort", "feature_mapping", "integration"])
    parser.add_argument('--requested-features', type=str, default="patient id, admission id", help="minimum cohort/task/demographic features to be included in the final output")
    parser.add_argument('--cohort-selection', type=str, default="Exclude patients with transfer between different ICU units or wards", help="cohort selection criteria")
    parser.add_argument('--database-knowledge', type=str, default="db_and_manual_and_prior", choices = ["parametric_knowledge", "db_only", "db_and_manual", "db_and_manual_and_prior"])
    parser.add_argument('--timeseries-feature-selection', type=str, default="Heart rate, Hemoglobin [Mass/volume] in Arterial blood", help="feature selection criteria")

    parser.add_argument('--target-database', type=str, default="mimic3", choices = ["mimic3","eicu","sicdb"])
    parser.add_argument('--trial-num', type=int, default=1) # total trial number
    parser.add_argument('--case-number', type=int, default=1) # index of user input case
    parser.add_argument('--llm-model', type=str, default="claude-3-5-sonnet-20240620")
    parser.add_argument('--case-name', type=str, default="eicu_cohort1")
    parser.add_argument('--results-dir', type=str, default="./results/")

    args = parser.parse_args()
    API_KEY = ""
    os.environ["ANTHROPIC_API_KEY"] = API_KEY

    db_connector = PostgreSQLConnector(target_database=args.target_database, batch_size=500, user="postgres", password="postgres")

    results_dir = setup_logging(args.results_dir)
    if args.agent_part in ["cohort"]:
        cohort_extraction(args, results_dir)
