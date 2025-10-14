import os
import re
import sys
import logging
import json
import time
import math
import shutil
import openai
import datetime
import ast
import argparse
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from decimal import Decimal

import torch.nn.functional as F

from data.data_util import unit_dict, all_ts_features
from utils import *
from utils import schema_knowledge_prompt, llm_api_generation, sql_post_process
from utils import candidate_feature_sql_prompt_maker, column_or_row_prompt_maker, additional_schema_guideline_prompt_maker, selected_schema_analysis_prompt_maker

def cohort_setup_logging(args):
    results_dir = args.results_dir + f"/cohort{args.case_name}_DB{args.target_database}_Pior{args.database_knowledge}_LLM{args.llm_model}_sg{args.schema_guideline}_sge{args.schema_guideline_edit}_obs{args.sql_observation}_fb{args.sql_feedback}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    log_filename = f"{results_dir}/log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),  # This will also print to console
            logging.StreamHandler()
        ]
    )
    
    # Log initial parameters
    logging.info("=== Starting New Run ===")
    logging.info(f"Target Database: {args.target_database}")
    logging.info(f"Trial Number: {args.trial_num}")
    logging.info(f"Case Name: {args.case_name}")
    logging.info(f"1. Cohort Selection: {args.cohort_selection}")
    logging.info(f"2. Requested Features: {args.requested_features}")
    logging.info(f"=== Ablation Variants ===")
    logging.info(f"LLM Model: {args.llm_model}")
    logging.info(f"Database Knowledge: {args.database_knowledge}")
    logging.info(f"Schema Guideline: {args.schema_guideline}")
    logging.info(f"Schema Guideline Edit: {args.schema_guideline_edit}")
    logging.info(f"SQL Error Feedback: {args.sql_feedback}")
    logging.info(f"SQL-based Observation: {args.sql_observation}")
    logging.info("==========================")
    return results_dir

def feature_map_setup_logging(args):
    results_dir = args.results_dir + f"/feature_map_DB{args.target_database}_Pior{args.database_knowledge}_LLM{args.llm_model}_sg{args.schema_guideline}_obs{args.sql_observation}_fb{args.sql_feedback}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    log_filename = f"{results_dir}/log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),  
            logging.StreamHandler()
        ]
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    # Log initial parameters
    logging.info("=== Starting New Run ===")
    logging.info(f"Target Database: {args.target_database}")
    logging.info(f"Trial Number: {args.trial_num}")
    logging.info(f"Requested Features: {args.timeseries_feature}")
    logging.info(f"=== Ablation Variants ===")
    logging.info(f"LLM Model: {args.llm_model}")
    logging.info(f"Database Knowledge: {args.database_knowledge}")
    logging.info(f"Schema Guideline: {args.schema_guideline}")
    logging.info(f"SQL-based Observation: {args.sql_observation}")
    logging.info(f"SQL Candidates Feedback: {args.sql_feedback}")
    logging.info("==========================")
    return results_dir

def base_database_information(args):
    example_cell_size = args.example_cell_size
    text_cell_maxlen = args.text_cell_maxlen
    database_knowledge = args.database_knowledge
    target_database = args.target_database
    agent_part = args.agent_part
    
    if "prior" in database_knowledge:
        if agent_part == "cohort":
            with open(f"./models/data/db_info/{target_database}_prior_knowledge_cohort.txt", 'r', encoding="UTF-8") as file:
                prior_knowledge_db_specific = file.read()
        elif agent_part == "feature_mapping":
            with open(f"./models/data/db_info/{target_database}_prior_knowledge_feature.txt", 'r', encoding="UTF-8") as file:
                prior_knowledge_db_specific = file.read()
        else:
            with open(f"./models/data/db_info/{target_database}_prior_knowledge_cohort.txt", 'r', encoding="UTF-8") as file:
                prior_knowledge_db_specific = file.read()
            with open(f"./models/data/db_info/{target_database}_prior_knowledge_feature.txt", 'r', encoding="UTF-8") as file:
                prior_knowledge_db_specific += file.read()
       
    # Schema Extraction
    schema_link = db_connector.connect(schema_extract_query)
    schema_link = tab2col_dictionary_join(schema_link, ['schema', 'table', 'column'])
    db_schema = "\n".join(schema_link)
    # Table Column and example values & Schema name extraction 
    example_values = db_connector.connect(example_value_extact_query.replace("number_of_values", str(example_cell_size)))
    table_column_values_list = []
    for example_value in example_values:
        table_column_values = db_connector.connect(example_value[0])      
        table_column_values_list.append(table_column_values[0])
    schema_raw = table_column_values_list[:][:]
    schema_names = list(set([i[0] for i in schema_raw]))
    schema_name = " or ".join(schema_names)

    # Foreign Key Extraction
    for i in schema_names:
        foreign_key = db_connector.connect(foreignkey_extact_query.replace("Schema_Name", i))
        foreign_key = fk_dictionary_join(foreign_key)
    
    if len(foreign_key)==0:
        foreign_key_file_path = f"./models/data/db_info/{target_database}_foreign_key_schema.txt"  
        foreign_key_schema = "Relationships Between Tables:\n"

        with open(foreign_key_file_path, 'r', encoding="UTF-8") as file:
            foreign_key = file.read()
    
    foreign_key_schema = f"Relationships Between Tables:\n{foreign_key}"

    table_column_values_dict = {}
    schema_names_dict = {}
    primary_key = {}
    for schema_n, table_name, column_name, example_value in table_column_values_list:
        if table_name not in schema_names_dict:
            schema_names_dict[table_name] = schema_n
        if table_name not in table_column_values_dict:
            table_column_values_dict[table_name] = {} 
        table_column_values_dict[table_name][column_name] = example_value  
        unique_values = set(example_value)
        if len(unique_values) == len(example_value):
            if table_name not in primary_key:
                primary_key[table_name] = []
            primary_key[table_name].append(column_name)
    primary_key_columns = set()
    for table in primary_key:
        columns = primary_key[table]
        for column in columns:
            primary_key_columns.add((table, column))
    schema_link_with_values = "\n".join(f"Table: {schema_names_dict[table_name]}.{table_name}: " + ", ".join(
                                            f"(Column: {col}, {'(primary_key), ' if (table_name, col) in primary_key_columns else ''}Values: [{', '.join([str(' '.join(str(v).split()))[:text_cell_maxlen] if isinstance(v, str) and len(' '.join(str(v).split())) > text_cell_maxlen else ' '.join(str(v).split()) for v in values])}])"
                                            for col, values in columns.items())
                                        for table_name, columns in table_column_values_dict.items())
    parsed_schema = defaultdict(list)
    for line in db_schema.strip().splitlines():
        table_match = re.match(r"Table:\s+([\w\.]+),\s+Columns:\s+\((.*?)\)", line)
        if table_match:
            table = table_match.group(1)
            columns = [col.strip() for col in table_match.group(2).split(',')]
            table_short = table.split('.')[-1]
            for col in columns:
                col_display = f"{col} (primary_key)" if (table_short, col) in primary_key_columns else col
                parsed_schema[table].append(col_display)
    annotated_lines = []
    for table, columns in parsed_schema.items():
        col_str = ", ".join(columns)
        line = f"Table: {table}, Columns: ({col_str})"
        annotated_lines.append(line)
    db_schema = "\n".join(annotated_lines)
    manual_path = f"./models/data/db_info/{target_database}_overview.txt"
    prior_knowledge_dict = {} # prior knowledge source
    global base_instruction
    if database_knowledge == "parametric_knowledge":
        base_instruction = parametric_instruction.replace("{Database Name}", str(schema_name))
        prior_knowledge_dict["schema_link_with_values"] = ""  
        prior_knowledge_dict["db_schema"] = ""
    else:
        base_instruction = database_instruction.replace("{Database Name}", str(schema_name))
        prior_knowledge_dict["foreign_key"] = foreign_key
        prior_knowledge_dict["schema_link_with_values"] = schema_link_with_values
        prior_knowledge_dict["db_schema"] = db_schema
        if database_knowledge == "db_and_manual":
            prior_knowledge_dict["manual_path"] = manual_path
            prior_knowledge_dict["manual_instruction"] = manual_instruction
        elif database_knowledge == "db_and_prior":
            prior_knowledge_dict["prior_knowledge_db_specific"] = prior_knowledge_db_specific
        elif database_knowledge == "db_and_manual_and_prior":
            prior_knowledge_dict["prior_knowledge_db_specific"] = prior_knowledge_db_specific
            prior_knowledge_dict["manual_path"] = manual_path
            prior_knowledge_dict["manual_instruction"] = manual_instruction
    if "prior_knowledge_db_specific" in prior_knowledge_dict:
        base_instruction = base_instruction + prior_instruction + str(prior_knowledge_dict['prior_knowledge_db_specific']) + "\n"
    # print("1. DB Name:", schema_name)
    # print("2. database_knowledge: ", database_knowledge)
    # print("3. DB schema: ", db_schema)
    # print("4. schema_link_with_values: ", schema_link_with_values)
    # print("5. Primary_Key: ", primary_key)
    # print("6. foreign_key: ", foreign_key)
    return schema_name, foreign_key_schema, base_instruction, prior_knowledge_dict

def cohort_feature_agent(args, schema_name, foreign_key_schema, base_instruction, prior_knowledge_dict, final_matching_result_dict = None):
    results_dir = cohort_setup_logging(args)    
    start_time = time.time()

    # Cohort Selection
    requested_features = args.requested_features
    cohort_selection = f"{args.cohort_selection}."

    ### Variables for LLM Framework
    max_retries = args.max_retries if args.sql_feedback else 1

    trial_index = args.trial_num
    # 1. Definition Schema Selection
    definition_schema = llm_api_generation(
        schema_knowledge_prompt(
            initial_prompt=base_instruction.replace("{schema type}", "definition ").replace("{feature}", "[Features] and [Cohort Selection]") + mapping_instruction_cohort,
            target_knowledge="definition_features", 
            prior_knowledge=prior_knowledge_dict,
            requested_features=requested_features,
            cohort_selection=cohort_selection),
            model=args.llm_model)
    if not args.schema_guideline:
        definition_schema = definition_schema.split("[Schema Guideline]:")[0].strip() + "\n[Schema Guideline]: "
    logging.info(f"\n===Definition Schema===: \n{definition_schema}")
    # 2. Cohort/Task and Demographic Schema
    retry_count = 0
    api_run_count = 1 # definition schema is already counted
    skip_to_sql_generation = False
    cohort_feature_sql, pre_schema, pre_sql, error_msg, error_think, error_class = None, None, None, None, None, None
    while (cohort_feature_sql is None) and (retry_count < max_retries) and (not skip_to_sql_generation):
        logging.info(f"Trial {trial_index} - Cohort/Task and Demographic Schema - Retry {retry_count}")
        if error_class in ["<wrong_schema>", None]:
            if api_run_count >= args.max_apirun - 1:
                skip_to_sql_generation = True
            if not skip_to_sql_generation:
                number_of_observation = 0
                requested_feature_schema = llm_api_generation(schema_knowledge_prompt(
                        initial_prompt=base_instruction.replace("{schema type}", "").replace("{feature}", "[Features] and [Cohort Selection]") + requested_feature_instruction,
                        target_knowledge="cohort_features",
                        prior_knowledge=prior_knowledge_dict,
                        requested_features=requested_features,
                        cohort_selection=cohort_selection,
                        pre_schema=pre_schema, 
                        pre_sql=pre_sql,
                        error_msg=error_msg,
                        error_think=error_think,
                        mapping_table=definition_schema), model=args.llm_model)
                api_run_count += 1
                
                if not args.schema_guideline:
                    requested_feature_schema = requested_feature_schema.split("[Schema Guideline]:")[0].strip() + "\n[Schema Guideline]: "
                logging.info(f"\n===Schema Linking and Guideline===: \n{requested_feature_schema}")
            
            sql_observation = args.sql_observation
            observation_count = 0
            extra_schema_information_total_dict = {}
            thought_count = 0
            obs_temperature = 0.0
            while (sql_observation) and (observation_count < args.max_sql_search_retries) and (not skip_to_sql_generation):
                extra_schema_information_dict = {}
                if observation_count > args.max_obs_count or retry_count > 1:
                    obs_temperature += 0.1
                    obs_temperature = min(obs_temperature, 1.0)
                if api_run_count >= args.max_apirun - 1:
                    skip_to_sql_generation = True
                if not skip_to_sql_generation:
                    schema_analysis_result = llm_api_generation(
                        selected_schema_analysis_prompt_maker(
                            schema=prior_knowledge_dict["schema_link_with_values"],
                            selected_schema=requested_feature_schema,
                            requested_features=requested_features,
                            cohort_selection=cohort_selection,
                            mapping_table=definition_schema,
                            foreign_key=foreign_key_schema,
                            error_think=error_think,
                            max_sql_search_at_once=args.max_sql_search_at_once,
                            previous_observation=extra_schema_information_total_dict), model=args.llm_model, temperature=obs_temperature)
                    observation_count += 1
                    api_run_count += 1
                    thought_count += 1
                    print(f"schema_analysis_result: {schema_analysis_result}")
                    # think = schema_analysis_result.split("<think>")[1].strip().split("</think>")[0].strip()
                    schema_analysis_report = schema_analysis_result.split("<output>")[1].strip().split("</output>")[0].strip()
                    # logging.info(f"\n\nThought {thought_count}: {think}")
                    logging.info(f"Action {thought_count}: {schema_analysis_report}")
                    if schema_analysis_report == "<need_more_information>":
                        analysis_sqls = schema_analysis_result.split("<SQL_queries>")[1].strip().split("</SQL_queries>")[0].strip().split("||")
                        if len(analysis_sqls) > args.max_sql_search_at_once:
                            analysis_sqls = analysis_sqls[:args.max_sql_search_at_once]
                        for analysis_sql in analysis_sqls:
                            number_of_observation += 1
                            try:
                                observation = db_connector.connect(sql_post_process(analysis_sql), max_row_n=args.max_obsoutput_len)
                                if not observation:
                                    raise ValueError("Result is empty or none")
                                if len(observation) > args.max_obsoutput_len:
                                    observation = observation[:args.max_obsoutput_len]
                                extra_schema_information_dict[analysis_sql] = observation
                            except Exception as e:
                                extra_schema_information_dict[analysis_sql] = str(e)
                        obs_sql_count = 0
                        for analysis_sql in extra_schema_information_dict:
                            obs_sql_count += 1
                            logging.info(f"\n\nobs_sql {thought_count}-{obs_sql_count}: {analysis_sql}")
                            logging.info(f"obs_result {thought_count}-{obs_sql_count}: {extra_schema_information_dict[analysis_sql]}")
                        if api_run_count >= args.max_apirun - 1:
                            skip_to_sql_generation = True
                        if not skip_to_sql_generation:
                            additional_schema_guideline = llm_api_generation(
                                additional_schema_guideline_prompt_maker(
                                    schema=prior_knowledge_dict["schema_link_with_values"],
                                    selected_schema=requested_feature_schema,
                                    requested_features=requested_features,
                                    cohort_selection=cohort_selection,
                                    observation_dict=extra_schema_information_dict,
                                    pre_observation_dict=extra_schema_information_total_dict
                                ), model=args.llm_model)
                            api_run_count += 1            
                            if extra_schema_information_dict is not None:
                                max_entries = args.max_obs_history  # Maximum number of entries to keep
                                if len(extra_schema_information_total_dict) >= max_entries:
                                    keys_to_remove = list(extra_schema_information_total_dict.keys())[:len(extra_schema_information_dict)]
                                    for key in keys_to_remove:
                                        extra_schema_information_total_dict.pop(key, None)
                                extra_schema_information_total_dict.update(extra_schema_information_dict)
                            
                            additional_schema_bool = additional_schema_guideline.split("<output>")[1].strip().split("</output>")[0].strip()
                            logging.info(f"\n\nObservation {thought_count}: {additional_schema_bool}")
                            if additional_schema_bool == "<add_information>":
                                additional_schema_guideline = additional_schema_guideline.split("<additional_information>")[1].strip().split("</additional_information>")[0].strip()
                                logging.info(f"Gained Information {thought_count}: {additional_schema_guideline}")
                                
                                if args.schema_guideline_edit:
                                    schema_only = requested_feature_schema.split("[Schema Guideline]:")[0].strip()
                                    guideline_only = requested_feature_schema.split("[Schema Guideline]:")[1].strip()
                                    if api_run_count >= args.max_apirun - 1:
                                        skip_to_sql_generation = True
                                    if not skip_to_sql_generation:
                                        new_schema = llm_api_generation(
                                            edit_schema_guideline_prompt_maker(
                                                cohort_selection=cohort_selection,
                                                requested_features=requested_features,
                                                schema=schema_only,
                                                guideline=guideline_only,
                                                additional_information=additional_schema_guideline
                                            ), model=args.llm_model)
                                        api_run_count += 1
                                        new_schema_guideline = new_schema.split("<edited_schema_guideline>")[1].strip().split("</edited_schema_guideline>")[0].strip()
                                        new_schema = new_schema.split("<edited_schema>")[1].strip().split("</edited_schema>")[0].strip()
                                        
                                        if not args.schema_guideline:
                                            requested_feature_schema = new_schema + "\n[Schema Guideline]: "
                                        else:
                                            requested_feature_schema = new_schema + "\n[Schema Guideline]: " + new_schema_guideline
                                        # logging.info(f"New Schema {thought_count}: {new_schema}")
                                        # logging.info(f"New Schema Guideline {thought_count}: {new_schema_guideline}")
                                    logging.info(f"New Schema and Schema Guideline {thought_count}: {requested_feature_schema}")
                                else:
                                    requested_feature_schema = requested_feature_schema + "\n[Additional Information]: " + additional_schema_guideline
                    else:
                        sql_observation = False    

        cohort_feature_sql = None
        try:
            output = llm_api_generation(
                Cohort_selection_sql_maker(
                requested_features=requested_features, 
                schema_name=schema_name, 
                schema=requested_feature_schema, 
                relation_information=foreign_key_schema, 
                cohort_selection=cohort_selection,
                pre_sql=pre_sql,
                error_msg=error_msg,
                ), model=args.llm_model)
            api_run_count += 1
            
            cohort_feature_sql = output.split("<SQL_query>")[1].strip().split("</SQL_query>")[0].strip()
            cohort_feature_sql = sql_post_process(cohort_feature_sql)
            # logging.info(f"\n\nSQL generation thought: {think}")
            logging.info(f"\n\nSQL query: {cohort_feature_sql}")
            cohort_feature = db_connector.connect(cohort_feature_sql, limit_row_n=False)
            if not cohort_feature:
                raise ValueError("Result is empty or none")
            else:
                cohort_feature = sorted(set(cohort_feature))
                print(f"requested_feature_schema : {requested_feature_schema}")
                cohort_feature = [tuple(float(x) if isinstance(x, Decimal) else x for x in tup) for tup in cohort_feature]
        except Exception as e:
            error_msg = str(e)            
            error_anaylsis = llm_api_generation(
                error_feedback_prompt_maker(
                    error_msg=error_msg,
                    schema_information=requested_feature_schema,
                    failed_sql=cohort_feature_sql,
                    target=requested_features), model=args.llm_model)
            api_run_count += 1
            error_think = error_anaylsis.split("<think>")[1].strip().split("</think>")[0].strip()
            error_class = error_anaylsis.split("<error_class>")[1].strip().split("</error_class>")[0].strip()
            logging.info(f"\n\nError Message: {error_msg}")
            # logging.info(f"Error Feedback: {error_think}")
            logging.info(f"Error Type: {error_class}")
            pre_sql = cohort_feature_sql
            pre_schema = requested_feature_schema
            cohort_feature_sql = None
            retry_count += 1
            time.sleep(2)  
            if api_run_count >= args.max_apirun:
                print("Failed to generate valid SQL after maximum retries")
                break
    
    if cohort_feature_sql is None:
        raise ValueError("Failed to generate valid SQL after maximum retries")
    else:
        lines = requested_feature_schema.strip().split("[Schema Guideline]:")
        requested_feature_schema = lines[0].strip()
        cohort_schema_guideline = lines[1].strip()

        logging.info("=========== Results ================")
        logging.info(f"Cohort Feature SQL Generation Completed")
        logging.info(f"api run count: {api_run_count}")
        logging.info(f"number of observation: {number_of_observation}")
        logging.info(f"cohort feature results: {cohort_feature[:20]}")
        logging.info(f"len(cohort_feature): {len(cohort_feature)}")
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Time taken: {duration} seconds")
        
        output_file_path = results_dir + f"/agent_output{trial_index}.pkl"

        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        final_result = {
            "requested_feature_schema": requested_feature_schema,
            "cohort_schema_guideline": cohort_schema_guideline,
            "target_database": args.target_database,
            "trial_number": trial_index,
            "case_name": args.case_name,
            "cohort_selection": args.cohort_selection,
            "requested_features": args.requested_features,
            "LLM_model": args.llm_model,
            "database_knowledge": args.database_knowledge,
            "schema_guideline": args.schema_guideline,
            "schema_guideline_edit": args.schema_guideline_edit,
            "sql_observation": args.sql_observation,
            "api_run_count": api_run_count,
            "number_of_observation": number_of_observation,
            "cohort_feature_sql": cohort_feature_sql,
            "cohort_feature": cohort_feature,
            "duration": duration,
            "results_dir": results_dir
        }
        with open(output_file_path, 'wb') as f:
            pickle.dump(final_result, f)
        logging.info(f"Requested features saved to: {output_file_path}")
        logging.info("====================================")
    return requested_feature_schema, cohort_schema_guideline, cohort_feature_sql
    
def feature_mapping_agent(args, schema_name, foreign_key_schema, base_instruction, prior_knowledge_dict):
    if args.agent_part == "feature_mapping":
        results_dir = feature_map_setup_logging(args)
    start_time = time.time()
    api_run_count = 0
    mini_api_run_count = 0    
    number_of_observation = 0 # we dont need observation for feature mapping
    final_matching_list = None
    timeseries_feature = args.timeseries_feature
    max_retries = args.max_retries
    similarity_threshold = args.similarity_threshold
    target_feature_name = re.sub(r'[^A-Za-z]', '', timeseries_feature)
    retry_count = 0
    row_feature_list = []
    column_feature_list = []
    definition_schema_dict = {} 
    list_per_definition_table = {}
    final_matching_result_dict = {"column": None, "row": {}}
    output_file_path = results_dir + f"/agent_output{args.trial_num}_{target_feature_name}.pkl"
    if os.path.exists(output_file_path):
        return  -1

    # 1. Definition Schema Selection
    definition_schema = llm_api_generation(
        schema_knowledge_prompt(
            initial_prompt="Definition table search: " + base_instruction.replace("{schema type}", "").replace("{feature}", "[Feature]") + mapping_instruction_ts,
            target_knowledge="timeseries_feature_definition_features", 
            prior_knowledge=prior_knowledge_dict,
            requested_features=timeseries_feature), model=args.llm_model)
    api_run_count += 1
    logging.info(f"\n===Definition Schema===: \n{definition_schema}")
    definition_information = definition_schema.split("[Schema Guideline]:")
    definition_items = definition_information[0].strip().split("Mapping Table: ")[1:]
    if not args.schema_guideline:
        definition_schema = definition_schema.split("[Schema Guideline]")[0]
    definition_schema_dict = {
        item.split(", Column:")[0].replace("Table Name:", "").strip():
        [col.split(", Values:")[0].strip() for col in item.split(", Column:")[1:]]
        for item in definition_items if item.strip()
    }        

    # 2. Time-series Feature Schema Linking
    timeseries_schema = llm_api_generation(
        schema_knowledge_prompt(
            initial_prompt="Time-series feature schema linking: " + base_instruction.replace("{schema type}", "").replace("{feature}", "[Feature]") + requested_timeseries_instruction,
            target_knowledge="timeseries_feature",
            prior_knowledge=prior_knowledge_dict,
            requested_features=timeseries_feature), model=args.llm_model)
    api_run_count += 1
    if not args.schema_guideline:
        schema_guideline = ""
    else:
        schema_guideline = timeseries_schema.split("<schema_guideline>")[1].strip().split("</schema_guideline>")[0].strip()
    selected_schema = timeseries_schema.split("<selected_schema>")[1].strip().split("</selected_schema>")[0].strip()
    logging.info(f"\n===Schema Linking and Guideline===")
    logging.info(f"selected_schema: {selected_schema}")
    logging.info(f"schema_guideline: {schema_guideline}")
    timeseries_feature_name = re.sub(r'\s*\(.*?\)', '', timeseries_feature)
    
    if args.sql_feedback:    
        column_or_row = llm_api_generation(
            column_or_row_prompt_maker(
                selected_schema=selected_schema,
                schema_guideline=schema_guideline,
                requested_feature=timeseries_feature_name), model=args.llm_model)
        api_run_count += 1
        column_feature = column_or_row.split("<feature_column>")[1].strip().split("</feature_column>")[0].strip().split("||")
        logging.info(f"\n===Column or Row===: \n{column_or_row}")
        if "None" not in column_feature: # column
            column_feature = [".".join(item.split(".")[1:]) for item in column_feature]
            column_feature_list.append(column_feature)
            final_matching_result_dict["column"] = column_feature
        else: # row
            row_feature_list = [{"table": line.split(", Column:")[0].replace("Table Name:", "").strip(), "columns": [col.split(", Values:")[0].strip() for col in line.split(", Column:")[1:]] } for line in selected_schema.splitlines() if line.strip()]

        logging.info(f"\n1-1 row_feature_list: {row_feature_list}")
        logging.info(f"\n1-2 column_feature_list: {column_feature_list}")
        logging.info(f"\n1-3 definition_schema_dict: {definition_schema_dict}")

        if row_feature_list:
            row_feature_definition_dict = {}
            for feature in row_feature_list:
                table_name = feature['table']
                if table_name in definition_schema_dict:
                    logging.info(f"table_name: {table_name}")
                    logging.info(f"definition_schema_dict[table_name]: {definition_schema_dict[table_name]}")
                    row_feature_definition_dict[table_name] = definition_schema_dict[table_name]

            logging.info(f"##### Part B-1. Candidate Features Listing for Row Features in {row_feature_definition_dict} #####")
            for definition_table in row_feature_definition_dict:
                sql_history = {}
                target_code_sql, error_msg = None, None
                pass_flag = False
                retry_count = 0
                while target_code_sql is None and retry_count < max_retries:
                    try:
                        output = llm_api_generation(candidate_feature_sql_prompt_maker(
                            definition_schema=definition_schema,
                            definition_table=definition_table,
                            foreign_key_schema=foreign_key_schema,
                            ts_schema=selected_schema,
                            sql_history=sql_history,
                            schema_guideline=schema_guideline), model=args.llm_model)
                        api_run_count += 1
                        target_code_sql = output.split("<SQL_queries>")[1].strip().split("</SQL_queries>")[0].strip()
                        target_code_sql = sql_post_process(target_code_sql)
                        logging.info(f"\ntarget_code_sql:{target_code_sql}")
                        candidates_list = db_connector.connect(target_code_sql, max_row_n=100000)
                        candidates_list = [tuple(float(x) if isinstance(x, Decimal) else x for x in tup) for tup in candidates_list]
                        if not candidates_list:
                            raise ValueError("Result is empty or none")
                        if len(candidates_list) == 100000:
                            continue
                        pass_flag = True
                    except Exception as e:
                        logging.info(f"\nError executing SQL (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                        error_msg = str(e)
                        sql_history[target_code_sql] = error_msg
                        target_code_sql = None
                        retry_count += 1
                        time.sleep(2)
                        continue
                    logging.info(f"candidates_list: {list(set(candidates_list))[:20]}")
                if not pass_flag:
                    continue
                
                candidates_list = list(set(candidates_list))
                logging.info(f"2-1 candidates_list length: {len(candidates_list)}")
                list_per_definition_table[definition_table] = candidates_list
                
                logging.info("\n##### Part B-2. Candidate Features and Target Feature Matching")
                feature_search_len = 100
                llm_api_batch_size = 20
                matching_result_dict = {}

                similarity_prob_list = []
                candidate_features = list(set(list_per_definition_table[definition_table])) # candidate features
                candidate_features_chunks = [candidate_features[i:i + feature_search_len] for i in range(0, len(candidate_features), feature_search_len)]
                mini_api_run_count = len(candidate_features_chunks)
                logging.info(f" - Checking similar features of {timeseries_feature} in {definition_table}!")
                start = time.time() # time check
                batch_llm_api_count = 0
            
                try: 
                    multi_filtered_prompt_all = [matching_noprob_prompt_maker(str(candidate_features_chunk), timeseries_feature, similarity_threshold) for candidate_features_chunk in candidate_features_chunks]
                    for prompt_size in range(0, len(multi_filtered_prompt_all), llm_api_batch_size):
                        similarity_prob_set = asyncio.run(Claude_long_generation(multi_filtered_prompt_all[prompt_size : prompt_size + llm_api_batch_size]))
                        similarity_prob_list += similarity_prob_set
                        batch_llm_api_count += 1
                        logging.info(f"Batch {batch_llm_api_count}...")
                except openai.error.Timeout as e:
                    time.sleep(3)
                
                if len(similarity_prob_list) == 0:
                    continue
                # logging.info("**** 1st similarity_prob_list:", similarity_prob_list)
                for similarity_prob_list_chunk in similarity_prob_list:
                    for similarity_probs in similarity_prob_list_chunk.split("};"):
                        try:
                            key, def_strs = similarity_probs.split(": ", 1)
                            for one_synonym in def_strs.strip("}{").split("|| "):
                                syn_tuple, prob = one_synonym.split(": ")
                                if key not in matching_result_dict:
                                    matching_result_dict[key] = []                     
                                matching_result_dict[key].append(syn_tuple.strip())
                        except:
                            pass                   
                matching_result_dict = dict(sorted(matching_result_dict.items()))
                
                if not matching_result_dict: # no matching result
                    continue
                batch_llm_api_count = 0
                final_matching_list = llm_api_generation(matching_prob_prompt_maker(", ".join(matching_result_dict[timeseries_feature]), timeseries_feature, similarity_threshold), model=args.llm_model)
                api_run_count += 1
                logging.info(f"**** 2nd similarity_prob_list:{final_matching_list}")
                if final_matching_list:
                    logging.info(f"{time.time()-start:.4f} sec")
                    # final_matching_list = final_matching_list.split(": {")[1].strip(" ").strip("}").strip("{}").split("||")
                    # final_matching_list = [i.split(": ")[0] for i in final_matching_list if int(i.split(": ")[1]) >= max(similarity_threshold, 90)]
                    content = final_matching_list.split(": {", 1)[1].rstrip("}").strip()
                    items = content.split(" || ")
                    final_matching_list = [
                        item.split(": ")[0].strip()
                        for item in items
                        if item.split(": ")[-1].isdigit() and int(item.split(": ")[-1]) >= max(similarity_threshold, 90)
                    ]
                    logging.info(f"**** final_matching_list: {final_matching_list}")
                    final_matching_result_dict["row"][definition_table] = final_matching_list
                else:
                    final_matching_list = None
    else:
        column_or_row = llm_api_generation(
            column_or_row_prompt_maker_no_sql_search(
                selected_schema=selected_schema,
                requested_feature=timeseries_feature_name), model=args.llm_model)
        api_run_count += 1
        feature = column_or_row.split("<feature_column>")[1].strip().split("</feature_column>")[0].strip().split("||")
        logging.info(f"\n===Column or Row===: \n{column_or_row}")
        if "\"," not in feature: # column
            column_feature = [".".join(item.split(".")[1:]) for item in feature]
            column_feature_list.append(column_feature)
            final_matching_result_dict["column"] = column_feature
        else: # row
            final_matching_result_dict["column"] = feature
    
    if final_matching_result_dict["column"]:
        final_prediction = final_matching_result_dict["column"]
    elif final_matching_result_dict["row"]:
        matched_codes = final_matching_result_dict["row"]
        matched_codes = list(set(sum(matched_codes.values(), [])))
        final_prediction = [ast.literal_eval(item.strip()) for item in matched_codes]
        final_prediction = list(set(final_prediction))
        final_prediction = [item[:2] if len(item) >= 3 else item[1] for item in final_prediction]
    else: # No matching result
        final_prediction = [None]
    
    observation_sql_count = 0
    logging.info(f"final_prediction : {final_prediction}")

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Time taken: {duration} seconds")
    logging.info(f"output_file_path: {output_file_path}")
    final_result = {
        "target_database": args.target_database,
        "trial_number": args.trial_num,
        "requested_features": args.timeseries_feature,
        "LLM_model": args.llm_model,
        "database_knowledge": args.database_knowledge,
        "schema_guideline": args.schema_guideline,
        "sql_feedback": args.sql_feedback, # candidates feedback
        "sql_observation": args.sql_observation,
        "feature_mapping_result": final_prediction,
        "api_run_count": api_run_count,
        "mini_api_run_count": mini_api_run_count,
        "number_of_observation": number_of_observation,
        "observation_sql_count": observation_sql_count,
        "2nd_matching_results": final_matching_list,
        "duration": duration,
        "results_dir": results_dir
    }
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_result, f)
    logging.info(f"Requested features saved to: {output_file_path}")
    logging.info("====================================")

    return final_matching_result_dict

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-part', type=str, default="integration", choices = ["cohort", "feature_mapping", "cohort_with_timeseries", "integration"])
    parser.add_argument('--target-database', type=str, default="eicu", choices = ["mimic3","eicu","sicdb"])
    parser.add_argument('--trial-num', type=int, default=1) # total trial number
    parser.add_argument('--case-number', type=int, default=1) # index of user input case
    parser.add_argument('--llm-model', type=str, default="claude-3-5-sonnet-20240620")
    parser.add_argument('--case-name', type=str, default="eicu_cohort1")

    parser.add_argument('--sql-observation', default=False, action='store_true')
    parser.add_argument('--sql-feedback', default=False, action='store_true')
    parser.add_argument('--schema-guideline', default=False, action='store_true')

    parser.add_argument('--results-dir', type=str, default="./results")
    parser.add_argument('--matching-method', type=str, default="multi_prob_ths") 
    parser.add_argument('--similarity-threshold', type=int, default=80)     
    parser.add_argument('--max-retries',type=int, default=5)  # limit feedback analysis 
    parser.add_argument('--max-sql-search-retries',type=int, default=10)  
    parser.add_argument('--max-sql-search-at-once',type=int, default=5)  
    parser.add_argument('--observation-per-feature',type=int, default=3)  

    parser.add_argument('--max-apirun',type=int, default=50)  
    parser.add_argument('--max-obs-count',type=int, default=5)  
    parser.add_argument('--max-obs-history',type=int, default=20)  
    parser.add_argument('--max-obsoutput-len',type=int, default=20)  
    parser.add_argument('--schema-guideline-edit',type=bool, default=True) # only true yet
    parser.add_argument('--database-batch-size',type=int, default=500) 
    
    ##### selection methods of database-knowledge
    parser.add_argument('--database-knowledge', type=str, default="db_and_manual_and_prior", choices = ["parametric_knowledge", "db_only", "db_and_manual", "db_and_manual_and_prior"])
    parser.add_argument('--example-cell-size',type=int, default=10)  # example cell size
    parser.add_argument('--text-cell-maxlen',type=int, default=50)  # max length of 'text' cell value   
    
    # 'age', 'gender', 'length of stay', 'mortality status', 'length of stay'
    parser.add_argument('--requested-features', type=str, default="ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'length of stay (hours, rounded to 4 decimals in float format)'", help="minimum cohort/task/demographic features to be included in the final output")     
    parser.add_argument('--cohort-selection', type=str, default="Include only ICU stays from patients aged 19 to 29 and Include only Male and Exclude ICU stays with missing discharge time", help="cohort selection criteria")
    parser.add_argument('--final-output-columns', type=str, default="Feature name, Feature value, Feature unit, Feature measurment time")
    ### User Control
    # Prediction input/output control         
    # parser.add_argument('--timeseries-feature-list', type=str, default="Hemoglobin [Mass/volume] in Arterial blood")     
    parser.add_argument('--timeseries-feature', type=str, default="Hemoglobin [Mass/volume] in Arterial blood")     
    parser.add_argument('--target-time-range', type=str, default="48hour", choices = ["24hour", "48hour", "72hour", "all time"])     
    parser.add_argument('--api-key', type=str, default=None)
    args = parser.parse_args()

    db_connector = PostgreSQLConnector(target_database=args.target_database, batch_size=args.database_batch_size, user="postgres", password="")
    final_matching_result_dict = None
    
    os.environ["ANTHROPIC_API_KEY"] = args.api_key
    
    schema_name, foreign_key_schema, base_instruction, prior_knowledge_dict = base_database_information(args)
    if args.agent_part in ["cohort_with_timeseries", "feature_mapping"]:
        final_matching_result_dict = feature_mapping_agent(args, schema_name, foreign_key_schema, base_instruction, prior_knowledge_dict)
    
    if args.agent_part in ["cohort_with_timeseries", "cohort"]:
        requested_feature_schema, cohort_schema_guideline, cohort_feature_sql = cohort_feature_agent(args, schema_name, foreign_key_schema, base_instruction, prior_knowledge_dict, final_matching_result_dict)
    
    
