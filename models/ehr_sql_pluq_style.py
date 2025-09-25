import psycopg2  # conda install psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import os
import re
import time
import shutil
import logging
import argparse
import pickle
import anthropic

from utils import PostgreSQLConnector, sql_post_process

def setup_logging(args):
    results_dir = args.results_dir + f"bs_sql_{args.agent_part}/BS_SQL_{args.case_name}_DB{args.target_database}_Pior{args.database_knowledge}_LLM{args.llm_model}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    log_filename = f"{results_dir}/BS_SQL_log.txt"

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
    logging.info("=== Starting New Run for baseline SQL ===")
    logging.info(f"Target Database: {args.target_database}")
    logging.info(f"Trial Number: {args.trial_num}")
    logging.info(f"Case Number: {args.case_number}")
    logging.info(f"Case Name: {args.case_name}")
    logging.info(f"1. Cohort Selection: {args.cohort_selection}")
    logging.info(f"2. Requested Features: {args.requested_features}")
    logging.info(f"3. Timeseries Features: {args.timeseries_feature}")
    logging.info(f"=== Ablation Variants ===")
    logging.info(f"LLM Model: {args.llm_model}")
    logging.info(f"Database Knowledge: {args.database_knowledge}")
    logging.info("=========================================")
    return results_dir

def llm_api_generation(prompt, model="claude-3-5-sonnet-20240620"):
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
    question = f"List all {requested_features} information that satisfy following [Condition].\n[Condition]: {cohort_selection}\nEnsure the output in PostgreSQL strictly follows the order and format specified in each () of {args.requested_features}."

    with open(table_schema_path, 'r', encoding="UTF-8") as file:
        sql_table_schema = file.read()
    
    extra_knowledge = None
    prior_knowledge_db_specific = None

    if "manual" in database_knowledge:
        print("Get manual information")
        extra_knowledge_path = f"./models/data/db_info/{target_database}_overview.txt"
        with open(extra_knowledge_path, 'r', encoding="UTF-8") as file:
            extra_knowledge = file.read()
    
    if "prior" in database_knowledge:
        print("Get database specific information")
        with open(f"./models/data/db_info/{target_database}_prior_knowledge_cohort.txt", 'r', encoding="UTF-8") as file:
            prior_knowledge_db_specific = file.read()

    db_connector = PostgreSQLConnector(target_database=target_database, batch_size=500, user="postgres", password="postgres")

    prompt = f"""You are given Database information and the Question. Generate the PostgreSQL query for the following question. Note that you should generate 'null' if the question cannot be converted to SQL query given information.
Get only one SQL query as plain text. Do not include code delimiters (e.g., ```sql), comments, or any additional text.
    
[Database schema]
{sql_table_schema}"""
    if prior_knowledge_db_specific :
        prompt += f"\n[Prior knowledge]\n{prior_knowledge_db_specific}"
    if extra_knowledge :
        prompt += f"\n[Extra knowledge]\n{extra_knowledge}"
    
    prompt += f"\nQuestion: {question}\nSQL Query:"
    
    print(question)
    llm_api_answer = llm_api_generation(prompt, model=args.llm_model)
    if llm_api_answer == "null":
        return None, None
    else:
        cohort_feature_sql = sql_post_process(llm_api_answer)
        try:
            cohort_feature = db_connector.connect(cohort_feature_sql)
            print(cohort_feature[:5])
            print(len(cohort_feature))
            return cohort_feature_sql, cohort_feature
        except Exception as e:
            print(f"\nError executing SQL : {str(e)}")
            return cohort_feature_sql, None

def cohort_extraction(args, results_dir):
    target_database = args.target_database
    database_knowledge = args.database_knowledge
    requested_features = args.requested_features
    cohort_selection = args.cohort_selection
    llm_model = args.llm_model
    trial_num = args.trial_num
    start_time = time.time()
    for trial_index in range(trial_num):
        cohort_feature_sql, cohort_feature = get_cohort(results_dir, target_database, database_knowledge, requested_features, cohort_selection, llm_model)
        
        if cohort_feature is None:
            logging.info("============= Results ==================")
            logging.info(f"cohort feature SQL: {cohort_feature_sql}")
            logging.info(f"Failed to generate valid SQL after maximum retries")
        else:
            logging.info("============= Results ==================")
            logging.info(f"Cohort Feature SQL Generation Completed")
            logging.info(f"cohort feature SQL: {cohort_feature_sql}")
            logging.info(f"cohort feature results: {cohort_feature[:20]}")
            logging.info(f"len(cohort_feature): {len(cohort_feature)}")
        
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Time taken: {duration} seconds")
            
        output_file_path = results_dir + f"/BS_SQL_cohort_output{trial_index}.pkl"

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
            "cohort_feature_sql": cohort_feature_sql,
            "cohort_feature": cohort_feature,
            "duration": duration,
            "results_dir": results_dir
        }
        with open(output_file_path, 'wb') as f:
            pickle.dump(final_result, f)
        logging.info(f"Requested features saved to: {output_file_path}")
        logging.info("====================================")


def get_timeseries(results_dir, target_database, database_knowledge, timeseries_feature, llm_model):
    table_schema_path = f"./models/data/db_info/{target_database}_schema.txt"

    with open(table_schema_path, 'r', encoding="UTF-8") as file:
        sql_table_schema = file.read()
    
    extra_knowledge = None
    prior_knowledge_db_specific = None

    if "manual" in database_knowledge:
        print("Get manual information")
        extra_knowledge_path = f"./models/data/db_info/{target_database}_overview.txt"
        with open(extra_knowledge_path, 'r', encoding="UTF-8") as file:
            extra_knowledge = file.read()
    
    if "prior" in database_knowledge:
        print("Get database specific information")
        with open(f"./models/data/db_info/{target_database}_prior_knowledge_feature.txt", 'r', encoding="UTF-8") as file:
            prior_knowledge_db_specific = file.read()

    db_connector = PostgreSQLConnector(target_database=target_database, batch_size=500, user="postgres", password="postgres")

    prompt = """You are given a [Database schema] and a [Feature].
Task: Analyze the information provided below and classify the feature into one of the following categories:
<get schema>: Select this if you can find a column whose name literally matches any part of the given [Feature].
<get definition SQL>: Select this if no such column exists, but you can retrieve the corresponding feature information using an SQL query. The query should return the unique feature identifier, feature name, and unit from the definition table related to the [Feature].
<null>: Select this if neither a matching schema nor an SQL definition can be found.

Instructions:
- If you choose <get schema>, provide the matching table and column in the format: Table_Name.Column_Name
- If you choose <get definition SQL>, provide an SQL query in the format: SELECT unique_feature_identifier, feature_name, unit FROM dbname.Table_A WHERE ..."""
    
    prompt += f"\n[Database schema]\n{sql_table_schema}"
    if prior_knowledge_db_specific :
        prompt += f"\n[Prior knowledge]\n{prior_knowledge_db_specific}"
    if extra_knowledge :
        prompt += f"\n[Extra knowledge]\n{extra_knowledge}"
    
    prompt += f"""\n\n[Feature] : {timeseries_feature}

Output Format:
<classification>
<get schema>, <get mapping SQL>, or <null>  
</classification>

<answer>
[get answer for selected classification formation]
</answer>
"""
    llm_api_answer = llm_api_generation(prompt, model=args.llm_model)
    print("llm_api_answer: ", llm_api_answer)
    
    if "<get schema>" in llm_api_answer:
        llm_api_answer_type = "<get schema>"
    elif "<get mapping SQL>" in llm_api_answer:
        llm_api_answer_type = "<get mapping SQL>"
    else:
        llm_api_answer_type = "<null>"
    llm_api_data = llm_api_answer.split("<answer>")[1].strip().split("</answer>")[0].strip()

    print(f"{timeseries_feature} - llm_api_answer :\n{llm_api_answer}")
    if llm_api_answer_type == "<null>":
        return None, [None]
    elif llm_api_answer_type == "<get schema>":
        timeseries_data = [llm_api_data]
        return None, timeseries_data
    else:
        timeseries_feature_sql = sql_post_process(llm_api_data)
        try:
            timeseries_data = db_connector.connect(timeseries_feature_sql)
            print(len(timeseries_data))
            feature_mapping_result = []
            for timeseries_matching in timeseries_data:
                timeseries_matching = timeseries_matching if len(timeseries_matching) == 3 else (timeseries_matching[0], timeseries_matching[1], None)
                timeseries_matching_key = (int(re.search(r'\d+', str(timeseries_matching[0])).group()), timeseries_matching[1])
                if timeseries_matching_key not in feature_mapping_result:
                    feature_mapping_result.append(timeseries_matching_key)
            return timeseries_feature_sql, feature_mapping_result
        
        except Exception as e:
            print(f"\nError executing SQL : {str(e)}")
            return timeseries_feature_sql, [None]


def timeseries_extraction(args, results_dir):
    target_database = args.target_database
    database_knowledge = args.database_knowledge
    timeseries_feature = args.timeseries_feature
    llm_model = args.llm_model
    start_time = time.time()

    logging.info(f"============= Matching for {timeseries_feature} =============")
    timeseries_feature_sql, feature_mapping_result = get_timeseries(results_dir, target_database, database_knowledge, timeseries_feature, llm_model)
    logging.info("============= Results ==================")
    logging.info(f"Timeseries Feature SQL Generation Completed")
    logging.info(f"timeseries feature: {timeseries_feature}")
    logging.info(f"timeseries mapping feature: {feature_mapping_result}")
    
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Time taken: {duration} seconds")

    target_feature_name = re.sub(r'[^A-Za-z]', '', timeseries_feature)
    output_file_path = results_dir + f"/BS_output{args.trial_num}_{target_feature_name}.pkl"

    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    final_result = {
        "target_database": args.target_database,
        "trial_number": args.trial_num,
        "case_number": args.case_number,
        "requested_features": args.timeseries_feature,
        "LLM_model": args.llm_model,
        "database_knowledge": args.database_knowledge,
        "feature_mapping_result": feature_mapping_result,
        "duration": duration,
        "results_dir": results_dir,
        "api_run_count": 1,
        "mini_api_run_count": 0,
        "number_of_observation": 0,
        "observation_sql_count": 0
    }
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_result, f)
    logging.info(f"Requested features saved to: {output_file_path}")
    logging.info("====================================")
                

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-part', type=str, default="feature_mapping", choices = ["cohort", "feature_mapping", "integration"])
    parser.add_argument('--example-number', type=int, default=20)
    parser.add_argument('--requested-features', type=str, default="", help="minimum cohort/task/demographic features to be included in the final output")
    parser.add_argument('--cohort-selection', type=str, default="", help="cohort selection criteria")
    parser.add_argument('--database-knowledge', type=str, default="db_and_manual_and_prior", choices = ["parametric_knowledge", "db_only", "db_and_manual", "db_and_manual_and_prior"])
    parser.add_argument('--timeseries-feature', type=str, default="Heart rate", help="minimum cohort/task/demographic features to be included in the final output")

    parser.add_argument('--target-database', type=str, default="eicu", choices = ["mimic3","eicu","sicdb"])
    parser.add_argument('--trial-num', type=int, default=1) # total trial number
    parser.add_argument('--case-number', type=int, default=1) # index of user input case
    parser.add_argument('--llm-model', type=str, default="claude-3-5-sonnet-20240620")
    parser.add_argument('--case-name', type=str, default="")
    parser.add_argument('--results-dir', type=str, default="./results/")

    args = parser.parse_args()
    API_KEY = ""
    os.environ["ANTHROPIC_API_KEY"] = API_KEY

    results_dir = setup_logging(args)
    if args.agent_part in ["cohort"]:
        cohort_extraction(args,results_dir)
    if args.agent_part in ["feature_mapping"]: 
        timeseries_extraction(args,results_dir)