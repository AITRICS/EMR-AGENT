import google.generativeai as genai
import os
import re
import sys
import json
import time
import math
import openai
import datetime
import ast
import argparse
import pandas as pd
import numpy as np
import pickle
import asyncio
import anthropic
import torch
import torch.nn.functional as F

import tiktoken  # pip install --upgrade tiktoken
import psycopg2  # conda install psycopg2
from psycopg2 import connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict

from data.data_util import unit_dict
from typing import List, Dict, Tuple, Optional, Any


class PostgreSQLConnector:
    def __init__(self, 
    target_database: str, 
    batch_size: int, 
    user: str = "postgres", 
    password: str = "postgres"
    ):
        self.target_database: str = target_database
        self.user: str = user
        self.password: str = password
        self.dbname: str = target_database
        self.host: Optional[str] = None
        self.port: Optional[str] = None
        self._set_db_params()
        self.batch_size: int = batch_size
        self.conn = None
        self.cursor = None

    def _set_db_params(self) -> None:
        database_config: Dict[str, Dict[str, str]] = {
            "mimic3": {"dbname": "mimic", "port": "5432", "host": "192.168.90.10"},
            "mimic4": {"dbname": "mimiciv", "port": "5432", "host": "192.168.90.24"},
            "eicu": {"port": "5433", "host": "192.168.90.10"},
            "hirid": {"port": "5433", "host": "192.168.90.24"},
            "sicdb": {"port": "5434", "host": "192.168.90.24"}
        }

        if self.target_database in database_config:
            config = database_config[self.target_database]
            self.dbname = config.get("dbname", self.dbname)
            self.port = config["port"]
            self.host = config["host"]
        else:
            raise ValueError(f"Do not support {self.target_database} database.")

    def connect(self, query: str, limit_row_n: bool = True, max_row_n: int = 1000000) -> List[tuple]:
        self.conn = connect(dbname=self.dbname, user=self.user, password=self.password, host=self.host, port=self.port)
        self.schema_extract: List[tuple] = []
        with self.conn.cursor() as self.setup_cursor:
            self.setup_cursor.execute("SET enable_nestloop = off;")
            self.setup_cursor.execute("SET enable_mergejoin = off;")

        with self.conn.cursor(name='server_side_cursor') as self.cursor:
            self.cursor.itersize = self.batch_size
            self.cursor.execute(query)

            while True:
                rows = self.cursor.fetchmany(self.batch_size)  # get data in batch size
                if not rows:
                    break
                self.schema_extract.extend(rows)
                if limit_row_n and len(self.schema_extract) >= max_row_n:
                    break

        # Reset server source
        with self.conn.cursor() as self.cleanup_cursor:
            self.cleanup_cursor.execute("RESET ALL;")     # reset all session 
            self.cleanup_cursor.execute("DISCARD TEMP;")  # delete sub table/cash
            self.cleanup_cursor.execute("DISCARD PLANS;") # delete prepared plan

        return self.schema_extract

    def close(self) -> None:
        """
        End the database connection.
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """
        Justify context manager approach action.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def format_schema_for_llm(text):
    return "\n\n".join(
        f"Table Name: {line.split(', Column:')[0].replace('Table Name:', '').replace(',,', '').strip()}: {{\n" +
        "\n".join(f"Column:{col.strip()}" for col in line.split(', Column:')[1:]) + "\n}}"
        for line in text.strip().splitlines() if line.strip()
    )

def parse_time_series_data(text: str) -> Dict[str, str]:

    data_dict: Dict[str, str] = {}
    entries = re.split(r'\n- ', text.strip())
    entries = entries[1:]
    
    for entry in entries:
        lines = entry.split('\n')
        key = lines[0].strip()
        value = " ".join(line.strip() for line in lines[1:])
        data_dict[key] = value
    
    return data_dict

def tab2col_dictionary_join(schema_data: List[tuple], columns: List[str]) -> List[str]:

    tables: Dict[str, List[str]] = {}
    df = pd.DataFrame(schema_data, columns=columns)
    
    for _, row in df.iterrows():
        schema_name = row['schema']
        table_name = row['table']
        column_name = row['column']
        full_table_name = f"{schema_name}.{table_name}"
        
        if full_table_name not in tables:
            tables[full_table_name] = []
        tables[full_table_name].append(column_name)
    
    output: List[str] = []
    for table, cols in tables.items():
        columns_str = ', '.join(cols)
        output.append(f"Table: {table}, Columns: ({columns_str})")
    
    return output

def fk_dictionary_join(schema_data: List[tuple]) -> str:

    fk_df = pd.DataFrame(schema_data, columns=['schema_A', 'table_A', 'column_A', 'schema_B', 'table_B', 'column_B'])
    
    output = "\n".join(
        f"{row['table_A']}.{row['column_A']} = {row['table_B']}.{row['column_B']}"
        for _, row in fk_df.iterrows()
    )
    
    return output #if output else "Foreign key don't exist. Check your schema information."

def validate_sql(sql):
    if not sql.strip().upper().startswith(("SELECT","WITH")):
        raise ValueError("SQL must start with SELECT or WITH")
    return sql

def sql_post_process(sql):
    sql = sql.replace('\n', ' ')
    sql = re.sub('[ ]+', ' ', sql)
    if "sql" in sql:
        sql = sql.split("```sql")[1].strip().split("```")[0].strip()
    else:
        sql = sql.replace("```","").strip()
    return validate_sql(sql)

schema_extract_query = """
SELECT 
    table_schema AS schema_name,
    table_name,
    column_name
FROM 
    information_schema.columns
WHERE 
    table_schema NOT IN ('information_schema', 'pg_catalog')
ORDER BY 
    table_schema,
    table_name,
    ordinal_position;
"""

primarykey_extact_query = """
SELECT 
    tc.table_schema,
    tc.table_name,
    kcu.column_name
FROM 
    information_schema.table_constraints AS tc 
JOIN 
    information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
    AND tc.table_name = kcu.table_name
WHERE 
    tc.constraint_type = 'PRIMARY KEY'
    AND tc.table_schema = 'Schema_Name'
ORDER BY 
    tc.table_schema, 
    tc.table_name,
    kcu.ordinal_position;
"""

foreignkey_extact_query = """
SELECT
    tc.table_schema                           AS table_schema,
    tc.table_name                             AS table_name,
    kcu.column_name                           AS column_name,
    ccu.table_schema                          AS referenced_table_schema,
    ccu.table_name                            AS referenced_table,
    ccu.column_name                           AS referenced_column
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
   AND tc.table_schema = kcu.table_schema
JOIN information_schema.referential_constraints AS rc
    ON rc.constraint_name = tc.constraint_name
   AND rc.constraint_schema = tc.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON rc.unique_constraint_name = ccu.constraint_name
   AND rc.unique_constraint_schema = ccu.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = 'Schema_Name'
ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position;
"""   

example_value_extact_query = """
SELECT 
    'SELECT ''' || table_schema || ''' AS schema_name, ''' || table_name || ''' AS table_name, ''' || column_name || ''' AS column_name, ' ||
    'ARRAY(SELECT ' || quote_ident(column_name) || ' FROM ' || quote_ident(table_schema) || '.' || quote_ident(table_name) || ' LIMIT number_of_values) AS sample_values;'
FROM 
    information_schema.columns
WHERE 
    table_schema NOT IN ('information_schema', 'pg_catalog')
ORDER BY 
    table_schema,
    table_name,
    ordinal_position;
"""
#----------------------------------------------------SQL_Maker_prompts-----------------------------------------------
nondef_timeseries_feature_prompt = """
    [example timeseries feature] : heart rate
    [example Schema_name]: PatientDatabase
    [example Schema]:
    {'table': 'PatientDatabase.apacheapsvar', 'columns': ['subject_id', 'heartrate']}, {'table': 'PatientDatabase.vitalperiodic', 'columns': ['subject_id', 'observationoffset', 'heartrate']}
    [SQL]: SELECT vs.subject_id, vs.observationoffset, vs.heartrate
    FROM PatientDatabase.vitalsign vs 
    JOIN PatientDatabase.admissions e ON pat.subject_id = e.subject_id
    """

#-------------------------------------------------prior knowledge_prompt-------------------------------------
parametric_instruction = """Using your internal knowledge about Database {Database Name}, select all {schema type}schema that are necessary to extact {feature}.
Follow the exact "Format" under [Notes] without any extra symbols, code delimiters (e.g., ```sql), comments, or additional text.
"""

database_instruction = """Using [Database schema information] from Database {Database Name}, select all {schema type}schema that are necessary to extact {feature}.
Please select exact table name(s) and column name(s) in [Database schema information]. 
Follow the exact "Format" under [Notes] without any extra symbols, code delimiters (e.g., ```sql), comments, or additional text.
"""

manual_instruction = """\n\n[Database Manual] is about information of each tables and columns. So consider [Database Manual] to select columns that satisfied necessary details.
"""

prior_instruction = """**IMPORTANT** PRIORITIZE following information more than any other considerations: \n"""

requested_feature_instruction =""" 
[Notes]:
- Get all schema (tables, columns) related to each element in [Features] and [Cohort Selection].
- After listing the schema for each element in [Features] and [Cohort Selection], provide a [Schema Guideline] in a paragraph of no more than 15 sentences, 
explaining the details of the selected schema's columns (such as type or example values) 
and how to generate SQL to obtain patients from [Cohort Selection] with each [Features] 
and what it is missing to get the correct result.
- If necessary, utilize [Foreign Key] and [Mapping Table] when generating [Schema Guideline] for [Cohort Selection].
- Get patient's related year, date, time information such as admission date, birth date, etc.
- [Feature name] must be exactly same with [Feature].
- "Format": 
- [Feature name] 
    Table Name: dbname.Table_A , Columns: Column_a, Column_b, Column_c
    Table Name: dbname.Table_B , Columns: Column_1, Column_2, Column_3
- [Feature name] 
    Table Name: dbname.Table_A , Columns: Column_a, Column_b, Column_c
    Table Name: dbname.Table_B , Columns: Column_1, Column_2, Column_3
[Schema Guideline]: (paragraph of no more than 15 sentences)
"""

requested_timeseries_instruction ="""
[Notes]:
- Select all schema (tables, columns, and 10 sample values for each column) related to extract [Feature], including definition table(s) and measurement table(s) of [Feature].
- The selected schema must include tables such as definition table(s) and measurement table(s) of [Feature].
- Provide a [Schema Guideline] in a paragraph of no more than 5 sentences, explaining the details of the columns (such as type or how to interpret the values).

Output Format: Provide exactly:

<selected_schema>
Table Name: dbname.Table_A , Column: Column_a, Values: [value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9, value_10], Column: Column_b, Values: [value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9, value_10]
Table Name: dbname.Table_B , Column: Column_1, Values: [value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9, value_10]
</selected_schema>

<schema_guideline>
[Schema Guideline in a paragraph of no more than 5 sentences]
</schema_guideline>
"""


mapping_instruction_cohort="""
[Notes]:
- Identify only definition schema with mapping information that can be used to extract patients of [Cohort Selection] with [Features].
- Exclude all tables that have actual numeric measurement (vital sign or lab test) columns.
- If identified tables have measurement unit information (not results), get all columns without result information.
- After listing the schema for each feature, provide a [Schema Guideline] in a paragraph of no more than 10 sentences, explaining the details of the columns (such as type or how to interpret the values).
- Output Format:
    Mapping Table: dbname.Table_A , Columns: Column_a, Column_b, Column_c
    Mapping Table: dbname.Table_B , Columns: Column_1, Column_2, Column_3
[Schema Guideline]: (paragraph of no more than 10 sentences)"
""" 

mapping_instruction_ts ="""
[Notes]:
- Identify only definition schema (table(s), column(s), and 3 sample values for each column) related to [Feature].
- Exclude columns that have actual measurement values (vital sign or lab test).
- *Important* The columns of definition sceham must include [Feature]'s information such as code, item number, name, abbreviation, etc.
- If identified definition table(s) have measurement unit information (not measurement value), get all the columns without actual measurement value information.
- After listing the schema for each feature, provide a [Schema Guideline] in a paragraph of no more than 10 sentences, explaining the details of the columns (such as type or how to interpret the values).
- Output Format:
    Mapping Table: dbname.Table_A , Column: Column_a, Values: [value_1, value_2, value_3], Column: Column_b, Values: [value_1, value_2, value_3], Column: Column_c, Values: [value_1, value_2, value_3]
    Mapping Table: dbname.Table_B , Column: Column_a, Values: [value_1, value_2, value_3], Column: Column_b, Values: [value_1, value_2, value_3], Column: Column_c, Values: [value_1, value_2, value_3]
[Schema Guideline]: (paragraph of no more than 10 sentences)
""" 

#----------------------------------------------------Step_prompt-------------------------------------
def schema_knowledge_prompt(
    initial_prompt : str = None,
    target_knowledge : str = None,
    prior_knowledge : Dict[str, str] = None,
    requested_features : List[str] = None,
    cohort_selection: str = None,
    pre_sql: str = None,
    pre_schema: str = None,
    error_msg: str = None,
    error_think: str = None,
    mapping_table: str = None,
    ) -> str :
    prompt = initial_prompt
    if requested_features:
        if "timeseries_feature" in target_knowledge:
            prompt = prompt + "\n[Feature]: " + str(requested_features) 
        else:
            prompt = prompt + "\n[Features]: " + str(requested_features)
            
    if cohort_selection:
        prompt = prompt + "\n[Cohort Selection]: " + cohort_selection

    if target_knowledge == "cohort_features":
        prompt += "\n\n[Foreign Key]: " + str(prior_knowledge.get("foreign_key", "").replace("\n\n", "\n"))
        prompt += "\n[Mapping Table]: " + mapping_table + "\n"
    
    if "manual_path" in prior_knowledge:
        with open(prior_knowledge["manual_path"], "r", encoding="UTF-8") as file:
            database_manual = file.read()
        prompt += prior_knowledge["manual_instruction"]
        prompt += "\n[Database Manual]:\n" + database_manual
    
    prompt += "\n[Database schema information]:\n" + str(prior_knowledge.get("schema_link_with_values", ""))

    if pre_schema is not None:
        Feed_back_prompt = """**IMPORTANT**\nFeedback Note: [Previous Failed SQL] and [Error Feedback] are failed SQL and error feedback when generated using [Previously Selected Schema]. 
Carefully exmaine [Previously Failed Selected Schema], [Previous Failed SQL], [Error Feedback], [Error Message] to generate correct [Selected Schema Informations].
If [Error Message] is 'Result is empty or none', it is possible that you are incorrectly assumes specific categorical values in the database without verifying their actual presence, leading to potentially invalid SQL conditions.
You might need to check the actual values in the database using [Mapping Table] and [Foreign Key].\n"""
        prompt = prompt + Feed_back_prompt + "\n[Previously Failed Selected Schema]: " + pre_schema + "\n[Previous Failed SQL]: " + pre_sql + "\n[Error Message]: " + error_msg + "\n[Error Feedback]: " + error_think
    return prompt + "\n[Selected Schema Informations]: "

def column_or_row_prompt_maker(
    selected_schema: str,
    schema_guideline: str,
    requested_feature: str,
) -> str:
    instruction = """
Classify whether the [Feature] name is literally present in any column name(s) of [Selected Schema]. If necessary, use [Schema Guideline] to help you classify whether the [Feature] name is literally present in any column name(s) of [Selected Schema].
If the [Feature] name is literally present in any column name(s) (e.g., [Feature]: 'chris', and [Selected Schema] has column names 'Destin', 'tom', 'CHRIS'), return it as Schema_Name.Table_Name.Column_Name between <feature_column> and </feature_column>.
    - Matching should be case-insensitive, space-insensitive, and symbol-insensitive. Reasonable abbreviations are also accepted.
    - Do not match semantic or contextual similarity. Only match if [Feature] name is a literal substring of the column name after removing case, space, and symbol differences.
    - If more than one column name is present in [Selected Schema], return all of them in <feature_column> separated by || as Schema_Name.Table_Name.Column_Name||Schema_Name.Table_Name.Column_Name||....
    - Never match based on content or examples of values in the column - **only use column names**.
If the [Feature] name is not literally present in any column name(s) (e.g., [Feature]: 'chris', and [Selected Schema] has column names 'name', 'tom', 'Andy'), output <feature_column>None</feature_column>.
    - If the [Feature] name only matches semantically or through contextual similarity, but not literally, output <feature_column>None</feature_column>.

Output Format: Provide exactly:
<think>
[Explain your thought process clearly and concisely in no more than 5 sentences.]
</think>

<feature_column>
[Schema_Name.Table_Name.Column_Name if [Feature] name is literally present in any column name(s) from [Selected Schema], 
or None if not.]
</feature_column>
"""
    selected_schema = format_schema_for_llm(selected_schema)
    prompt = instruction + "\n[Selected Schema]: " + selected_schema + "\n[Feature]: " + requested_feature + "\n[Schema Guideline]: " + schema_guideline
    return prompt

def column_or_row_prompt_maker_no_sql_search(
    selected_schema: str,
    requested_feature: str,
) -> str:
    instruction = """
Classify whether the [Feature] name is literally present in any column name(s) of [Selected Schema]. If necessary, use [Schema Guideline] to help you classify whether the [Feature] name is literally present in any column name(s) of [Selected Schema].
If the [Feature] name is literally present in any column name(s) (e.g., [Feature]: 'chris', and [Selected Schema] has column names 'Destin', 'tom', 'CHRIS'), return it as Schema_Name.Table_Name.Column_Name between <feature_column> and </feature_column>.
    - Matching should be case-insensitive, space-insensitive, and symbol-insensitive. Reasonable abbreviations are also accepted.
    - Do not match semantic or contextual similarity. Only match if [Feature] name is a literal substring of the column name after removing case, space, and symbol differences.
    - If more than one column name is present in [Selected Schema], return all of them in <feature_column> separated by || as Schema_Name.Table_Name.Column_Name||Schema_Name.Table_Name.Column_Name||....
    - Never match based on content or examples of values in the column - **only use column names**.
If the [Feature] name is not literally present in any column name(s), return ("unique_feature_identifier_code", "feature_name", "unit") between <feature_column> and </feature_column>.
    - If more than one ("unique_feature_identifier_code", "feature_name", "unit") can be returned, return all of them in <feature_column> separated by || as ("unique_feature_identifier_code", "feature_name", "unit")||("unique_feature_identifier_code", "feature_name", "unit")||....

Output Format: Provide exactly:
<think>
[Explain your thought process clearly and concisely in no more than 5 sentences.]
</think>

<feature_column>
[Schema_Name.Table_Name.Column_Name if [Feature] name is literally present in any column name(s) from [Selected Schema], 
or ("unique_feature_identifier_code", "feature_name", "unit") if not.]
</feature_column>
"""
    selected_schema = format_schema_for_llm(selected_schema)
    prompt = instruction + "\n[Selected Schema]: " + selected_schema + "\n[Feature]: " + requested_feature
    return prompt

def error_feedback_prompt_maker(
    error_msg: str,
    schema_information: str,
    failed_sql: str,
    target: str,
) -> str:
    instruction = """
You are an assistant that classifies SQL execution errors.

Given:
- Failed SQL: The query that failed.
- Selected Schema: Schema used to generate the query.
- Target: Intended data to extract.
- Error Feedback: Database error message.

Task: Analyze the provided information and classify the error as one of the following:
<syntax_error>: SQL syntax is incorrect (e.g., missing keywords, misplaced clauses, invalid syntax).
<wrong_schema>: Schema-related issue (e.g., referencing non-existent tables or columns, incorrect schema usage).

Output Format: Provide your response exactly as below, without additional commentary or text:
<think>
[Explain your thought process clearly and concisely in less than 6 sentences, highlighting why you chose this classification and exactly what factors caused the error.]
</think>

<error_class>
<syntax_error> or <wrong_schema>
</error_class>
"""
    prompt = instruction + "\n[Selected Schema]: " + schema_information + "\n[Target]: " + str(target) + "\n[Failed SQL]: " + failed_sql + "\n\n[[Error Feedback]]:\n" + error_msg + "\n\nOutput:"
    return prompt

def edit_schema_guideline_prompt_maker(
    schema: str,
    guideline: str,
    additional_information: str,
    requested_features: List[str],
    cohort_selection: str
) -> str:
    instruction = """
You are an assistant tasked with editing the [Schema Guideline] and [Schema] based on newly obtained [Additional Information].
Carefully review the following components:
- [Schema]: The original schema for [Target Features] and [Cohort Selection].
- [Schema Guideline]: The original schema guideline for [Target Features] and [Cohort Selection].
- [Additional Information]: New information gained from SQL Observation(s).
- [Target Features]: Specific features required to extract for each patient.
- [Cohort Selection]: specifications for the configuration of patients to extract.

Task:
- *IMPORTANT* Make sure to update both [Schema Guideline] and [Schema] based on [Additional Information].
- Update the [Schema Guideline] and [Schema] based on [Additional Information] to support SQL query generation for extracting [Target Features] from the [Cohort Selection] patients.
- If [Additional Information] resolves previously unknown parts in [Schema Guideline], update them accordingly in [Schema Guideline].
- Provide the updated [Schema Guideline] no more than 15 sentences between <edited_schema_guideline> and </edited_schema_guideline>.
- Provide the updated [Schema] between <edited_schema> and </edited_schema> with the same format as the original [Schema] but with updated information such as column name, column type, column value (you can even add value examples), etc.
- If there is no need to update, provide the original [Schema Guideline] and [Schema].

Output Format: Provide your response exactly as below, without additional commentary or text:
<think>
[Explain your thought process clearly and concisely in no more than 5 sentences.]
</think>

<edited_schema_guideline>
[Edited Schema Guideline no more than 15 sentences]
</edited_schema_guideline>

<edited_schema>
[Edited Schema]
</edited_schema>
"""
    prompt = instruction + "\n[Schema]: " + schema + "\n[Schema Guideline]: " + guideline + "\n[Additional Information]: " + additional_information + "\n[Target Features]: " + str(requested_features) + "\n[Cohort Selection]: " + cohort_selection
    return prompt

def candidates_feedback_prompt_maker(
    candidates: str,
    definition_schema: str,
    selected_schema: str,
    foreign_key_schema: str
) -> str:
    instruction = """
    You are an assistant that evaluates whether the given [Candidates] tuples' first and second indices correctly represent 
    'unique feature identifier code' and 'feature name' based solely on the values provided in [Candidates].

    Structure:
    - Each element in [Candidates] is a tuple with the format: (identifier_code, feature_name, unit).
    - The first index should be the 'unique feature identifier code'.
    - The second index should be the 'feature name' and must belong to either vital signs or lab-test features.
    - The third index, if present, can be a 'unit' column which may contain NULL values if the feature inherently lacks a unit.

    Important:
    - A valid 'identifier code' should follow a consistent format, typically alphanumeric or a specific pattern (e.g., 'ID_123').
    - A valid 'feature name' must explicitly represent a vital sign or a lab-test feature, matching common clinical terminology.
    - A valid 'unit' column may include NULL values, but the non-null values should be consistent with the expected unit type.

    Task: Analyze the given [Candidates] and classify as follows:
    <correct>: [Candidates]'s first and second indices contain valid 'unique feature identifier code' and 'feature name'.
    <incorrect>: [Candidates]'s first and second indices do not contain valid 'unique feature identifier code' and 'feature name'.
    - If <incorrect>, provide a brief reason or suggestion for improvement in no more than 3 sentences.

    Output Format:
    <think>
    [Briefly explain your reasoning in no more than 5 sentences.]
    </think>

    <output>
    [<correct> or <incorrect>]
    </output>

    <error_msg>
    [If <incorrect>, provide a suggested improvement.]
    </error_msg>
    """
    prompt = (
        instruction + "\n[Candidates]: " + str(candidates) +
        "\n[Definition Schema]: " + definition_schema +
        "\n[Selected Schema]: " + selected_schema +
        "\n[Foreign Key Schema]: " + foreign_key_schema
    )
    return prompt

def additional_schema_guideline_prompt_maker(
    schema: str,
    selected_schema: str,
    requested_features: str,
    cohort_selection: str,
    observation_dict: dict,
    pre_observation_dict: dict
) -> str:
    instruction = """
You are an assistant to observe [SQL Observation] and find extra information to add to [Schema Guideline] to assist when generating SQL query for [Cohort Selection] patients with each of [Target Features].
Carefully review the following components:
- Original Schema: Includes tables, columns, and associated values before Schema Linking.
- Selected Schema: The schema and its guidelines chosen to extract patients according to [Cohort Selection] with specified features [Target Features].
- Target Features: Specific features required to extract for each patient.
- Cohort Selection: specifications for the configuration of patients to extract.
- SQL Observation: Results from executed SQL queries, provided as a dictionary (query-output pairs), offering further insights into [Original Schema] and possibly suggest more information to add to [Selected Schema]. The output could be an error message if the SQL query is failed. Keep in mind that the length of [SQL Observation] is limited to 20.
- Pre-Observation: Previously observed information.

Task: 
- Select one of the below two options:
<add_information>: If you found something valuable information from [SQL Observation]
<no_information>: If you found nothing valuable information from [SQL Observation]
- *IMPORTANT* If you selected <add_information>, provide the gained information from [SQL Observation] in less than 5 sentences between <additional_information> and </additional_information>. The gained information should improve the [Schema Guideline] to extract the [Patients] according to [Cohort Selection] with [Target Features].

Output Format: Provide your response exactly as below, without additional commentary or text:
<think>
[Clearly and concisely explain your reasoning behind the classifications based on the given information in less than 5 sentences.]
</think>

<output>
<add_information> or <no_information>
</output>

<additional_information>
[Do not include information that is already in [Selected Schema] and [Schema Guideline]. Provide the gained information from [SQL Observation] in less than 6 sentences. This should be helpful to improve the [Schema Guideline] to extract the [Patients] according to [Cohort Selection] with [Target Features].] 
</additional_information>
    """
    pre_observation = ""
    for sql in pre_observation_dict:
        pre_observation += f"\n[Observation SQL {sql}]: {pre_observation_dict[sql]}"
    observation = ""
    for sql in observation_dict:
        observation += f"\n[Observation SQL {sql}]: {observation_dict[sql]}"
    prompt = instruction + "\n[Original Schema]: " + schema + "\n[Selected Schema]: " + selected_schema + "\n[Target Features]: " + str(requested_features) + "\n[Cohort Selection]: " + cohort_selection + "\n[Pre-Observation]: " + pre_observation + "\n[SQL Observation]: " + observation
    return prompt

def filtering_decision_prompt_maker(
    target_feature: str,
    feature_map: str,
    selected_schema: str,
    sql_history: dict,
    max_sql_search_at_once: int,
) -> str:
    instruction = f"""
You are an assistant that evaluates whether the [Candidate Feature] is truly the [Target Feature] using your knowledge of the [Target Map], [Selected Schema], and previous [Observation History] (if provided).
You may generate SQL queries to inspect actual measurement values or relevant metadata of the [Candidate Feature] using the [Selected Schema].
Note: The output of each SQL query is limited to 20 rows.

Task:
- Assess whether the [Candidate Feature] matches the intended [Target Feature].
- Provide your decision between <output> and </output> using one of:
  - <correct>: The [Candidate Feature] is truly the [Target Feature].
  - <incorrect>: The [Candidate Feature] is not the [Target Feature].
  - <need_observation>: You are unsure and need to generate observation SQL to get more information from the database.

Output Format (strict):
<think>
[Explain your reasoning clearly in less than 5 sentences.]
</think>

<output>
<correct> or <incorrect> or <need_observation>
</output>

<observation_sql>
[If <need_observation>, generate up to {max_sql_search_at_once} SQL queries using [Selected Schema] to inspect measurement values or related metadata of the [Candidate Feature], separated by || if needed. Otherwise, leave this empty.]
</observation_sql>
"""
    pre_observation = ""
    for sql in sql_history:
        pre_observation += f"\n[Observation SQL {sql}]: {sql_history[sql]}"
    prompt = instruction + "\n[Target Feature]: " + target_feature + "\n[Candidate Feature]: " + feature_map + "\n[Selected Schema]: " + selected_schema + "\n[Observation History]:" + pre_observation    
    return prompt

def selected_schema_analysis_prompt_maker(
    schema: str,
    selected_schema: str,
    requested_features: str,
    cohort_selection: str,
    mapping_table: str,
    foreign_key: str,
    error_think: str,
    max_sql_search_at_once: int,
    previous_observation: dict
) -> str:
    instruction = """
You are an assistant tasked with evaluating the provided schema and guideline to determine if they are sufficient to support data extraction requirements.
Carefully review the following components:
- Original Schema: The schema before Schema Linking.
- Selected Schema: The schema and its guideline to assist to extract patients according to [Cohort Selection] with specified features [Target Features].
- Target Features: Specific features required for each patient. Note that names in [Target Features] are not always same in [Schema], do not assume value in schema.
- Cohort Selection: specifications for the configuration of patients to extract.
- Mapping Table: A table(s) and column(s) that contain mapping information of certain features, indicating details/definitions of certain features.
- Foreign Key: A foreign keys of the original schema.
- Error Feedback: If available, feedback from previously generated SQL queries indicating errors.
- Previous Observation (if provided): Previously observed information through SQL queries. Do not generate any SQL query that is already in [Previous Observation].

Task: 
- Assess whether the current [Selected Schema] and associated [Schema Guideline] are ENOUGH to extract the [Patients] according to [Cohort Selection] with [Target Features]. Classify your evaluation clearly into one of the following:

<need_more_information>: The [Selected Schema] and [Schema Guideline] are insufficient or require clarification.
- ***IMPORTANT*** Do not simply assume the names in [Target Features] and [Cohort Selection] are in [Selected Schema] and [Schema Guideline]. If you are not sure about the values, you need to first check or ask for the actual values that exist in the column (e.g., via `SELECT DISTINCT column FROM table`) before using them.
- Only use a specific value in WHERE clauses if it is explicitly observed in the schema or query result, otherwise keep you position as <need_more_information>.
- If [Error Feedback] exists and indicates issues, generate additional SQL queries to retrieve missing details.

<correct>: The provided [Selected Schema] and [Schema Guideline] are sufficient.

If you classified the schema as <need_more_information>, 
- Generate SQL queries to retrieve the necessary additional details.
- If multiple queries are needed, separate each with ||.
- Do not generate SQL queries that retrieve entire tables — focus only on concise, targeted retrievals.
- If you need to use [Mapping Table] and [Foreign Key], please use them in the SQL queries.

Output Format: Provide your response exactly as below, without additional commentary or text:
<think>
[Clearly and concisely explain your reasoning behind the classifications based on the given information.]
</think>

<output>
<need_more_information> or <correct>
</output>

<SQL_queries>
[If you classified the schema as <need_more_information>, based on you think process, [Schema Guideline] and [Additional Information], provide SQL queries to retrieve additional details from the schema using [Original Schema], [Mapping Table] and [Foreign Key]. 
If multiple queries are needed, separate each with ||. Note that the number of SQL queries should not exceed [Max SQL Search At Once]. Do not include any SQL query that is already in [Previous Observation].]
</SQL_queries>
"""
    instruction = instruction.replace("[Max SQL Search At Once]", str(max_sql_search_at_once))
    pre_observation = ""
    for sql in previous_observation:
        pre_observation += f"\n[Observation SQL {sql}]: {previous_observation[sql]}"
    
    if error_think is not None:
        prompt = instruction + "\n[Original Schema]: " + schema + "\n[Selected Schema]: " + selected_schema + "\n[Target Features]: " + str(requested_features) + "\n[Cohort Selection]: " + cohort_selection + "\n[Mapping Table]: " + mapping_table + "\n[Foreign Key]: " + foreign_key + "\n[Previous Observation]: " + pre_observation + "\n[Error Feedback]: " + error_think
    else:
        prompt = instruction + "\n[Original Schema]: " + schema + "\n[Selected Schema]: " + selected_schema + "\n[Target Features]: " + str(requested_features) + "\n[Cohort Selection]: " + cohort_selection + "\n[Mapping Table]: " + mapping_table + "\n[Foreign Key]: " + foreign_key + "\n[Previous Observation]: " + pre_observation
    
    return prompt

def selected_features_analysis_prompt_maker(
    selected_feature: str,
    target_feature: str,
    original_schema: str,
    selected_schema: str,
    schema_guideline: str,
    foreign_key: str,
    previous_observation: dict
) -> str:
    instruction = """
You are an assistant to evaluate [Selected Feature] is truly the [Target Feature].
Carefully review the [Selected Schema], [Schema Guideline], [Previous Observations] to evaluate the [Selected Feature].

Step 1:
- Assess whether the [Selected Feature] is truly the [Target Feature].

Step 2:
- Make you decision by providing one of [<need_observation> or <correct> or <incorrect>] between <output> and </output>:
<need_observation>: If you need to generate observation SQL query(ies) to check things such as the actual measurement values or other related information of the [Selected Feature], you can generate the observation SQL query(ies) between <observation_sql> and </observation_sql> using [Selected Schema], [Schema Guideline], [Previous Observations], [Original Schema], and [Foreign Key].
Please note that no more than 20 results can be retrieved from the database. Generate the SQL query accordingly.
<correct>: [Selected Feature] is truly the [Target Feature].
<incorrect>: [Selected Feature] is not the [Target Feature].

Output Format: Provide your response exactly as below, without additional commentary or text:
<think>
[Clearly and concisely explain your reasoning behind the classifications in less than 5 sentences.]
</think>

<output>
[<need_observation> or <correct> or <incorrect>]
</output>

<observation_sql>
[If you need to generate observation SQL query(ies) to check things such as the actual measurement values or other related information of the [Selected Feature], 
you can generate the observation SQL query(ies) between <observation_sql> and </observation_sql> using [Selected Schema], [Schema Guideline], [Previous Observations].
Do not include any SQL query that is already in [Previous Observations]. Please note that no more than 20 results can be retrieved from the database. Generate the SQL query accordingly.]
</observation_sql>
"""
    prompt = instruction + "\n[Target Feature]: " + target_feature + "\n[Selected Feature]" + selected_feature + "\n[Original Schema]: " + original_schema + "\n[Selected Schema]: " + selected_schema + "\n[Schema Guideline]: " + schema_guideline + "\n[Foreign Key]: " + foreign_key + "\n[Previous Observation]: " + str(previous_observation)
    return prompt

def Cohort_selection_sql_maker(
    requested_features: List[str],
    schema_name: str,
    schema: str,
    relation_information: str,
    cohort_selection: str,
    pre_sql: Optional[str],
    error_msg: Optional[str],
    ) -> str:
    
    instruction = """\n\nQ: Using the provided [Schema] with tables and columns and [Schema Guideline], write a PostgreSQL query to extract patients according to [Cohort Selection] with specified features [Target Features].
Output is only the SQL query as plain text. Do not include code delimiters (e.g., ```sql), comments, or any additional text.

Follow these steps:
1. Select appropriate foreign keys(columns) provided in [Relation Information] to connect identified tables.
2. If necessary, use selected foreign key to make "JOIN". Do not use any other columns.
3. Ensure that each column referenced in the SELECT clause is present in the table alias used.
4. Use [Requested Features] to follow the sequence and format of '()' in [Requested Features] to generate the SQL query.
5. If some values are not visually understandable due to mapping code, add 'CASE' and 'WHEN' to replace the values with understandable values.
6. When writing WHERE conditions involving categorical values (e.g., gender, status), Do not assume specific values.
7. Only use a specific value in WHERE clauses if it is explicitly observed in the schema or query result.
8. When applying multiple inclusion/exclusion criteria, ensure that logically dependent conditions are ordered correctly.
    - Do not reorder or drop dependent conditions; maintain logical dependencies when translating natural language criteria into SQL.
9. For all float values in the SQL output, cast them to ::float in the SELECT clause. If rounding is applied, first cast to numeric for ROUND(..., n) to work, then cast the result back to ::float if a float output is desired.
10. In SQL WHERE clauses, string comparison is case-sensitive. Use LOWER(), UPPER(), or adjust collation if you need case-insensitive matching.

SQL generate rule: 
Ensure that the SQL query only applies numeric comparisons (such as BETWEEN) on values that are safely converted to integers, thereby preventing type conversion errors.
**IMPORTANT** Always extract the Patient ID as-is (without deduplication, filtering, or counting) for the first column, exactly as it appears in the database.

Output Format: Provide your response exactly as below, without additional commentary or text:

<think>
[Clearly and concisely explain your reasoning behind the sql generation based on the given information.]
</think>

<SQL_query>
[Write a PostgreSQL query to extract requested features of patients according to [Cohort Selection] with [Requested Features]]
</SQL_query>\n"""
    
    if pre_sql is None and error_msg is None:
        prompt = instruction + "[Cohort Selection]: " + cohort_selection + "\n[Requested Features]: " + str(requested_features) + "\n[Schema_name]: " + schema_name + "\n[Schema]:\n" + schema + "\n[Relation information]:\n" + relation_information
    else:
        instruction = instruction + "**IMPORTANT**\nFeedback Note: [Previous Failed SQL] and [Error Feedback] are failed SQL and error feedback. Carefully exmaine [Error Feedback] and avoid [Previous Failed SQL] to generate correct SQL.\n"
        prompt = instruction + "[Cohort Selection]: " + cohort_selection + "\n[Requested Features]: " + str(requested_features) + "\n[Schema_name]: " + schema_name + "\n[Schema]:\n" + schema + "\n[Relation information]:\n" + relation_information + "\n[Previous Failed SQL]:\n" + pre_sql  + "\n[Error Feedback]:\n" + error_msg
    return prompt

def nondef_timeseries_feature_extractor_sql_maker(
    nondef_timeseries_feature: List[str],
    foreign_key_schema: str,
    schema_name: str,
    nondef_timeseries_feature_schema: Dict[str, List[Dict[str, List[str]]]],
    schema_guideline_per_timeseries_feature: str,
    pre_sql: Optional[str],
    error_msg: Optional[str]
    ) -> str:

    instruction = """\n\nQ: Using [Schema] and [Schema Guideline], write a PostgreSQL query to extract [Timeseries Feature] data **that includes related information for the timeseries feature** (e.g., ICU Stay ID, time, and value).
Output is only the SQL query as plain text. Do not include code delimiters (e.g., ```sql), comments, or any additional text.

Follow these steps:
1. Select appropriate foreign keys(columns) provided in [Relation Information] to connect identified tables.
2. Ensure that each column referenced in the SELECT clause is present in the table alias used.

Output Format: Provide your response exactly as below, without additional commentary or text:

<think>
[Clearly and concisely explain your reasoning behind the sql generation based on the given information.]
</think>

<SQL_query>
[Write a PostgreSQL query to extract [Timeseries Feature] (e.g., ICU Stay ID, time, value, etc.) with [Schema], [Relation Information] and [Schema Guideline]]
</SQL_query>
"""
    if pre_sql is None:
        prompt =  instruction + "\n[Timeseries Feature] " + str(nondef_timeseries_feature) + "\n[Schema_name]: " + schema_name + "\n[Schema]:\n" + str(nondef_timeseries_feature_schema) + "\n[Schema Guideline]:" + schema_guideline_per_timeseries_feature + "\n[Relation information]:\n" + foreign_key_schema
    else:
        instruction = instruction + "**IMPORTANT**\nFeedback Note: [Previous Failed SQL] and [Error Feedback] are failed SQL and error feedback. Carefully exmaine [Error Feedback] and avoid [Previous Failed SQL] to generate correct SQL.\n"
        prompt =  instruction  + "\n[Timeseries Feature] " + str(nondef_timeseries_feature) + "\n[Schema_name]: " + schema_name + "\n[Schema]:\n" + str(nondef_timeseries_feature_schema) + "\n[Schema Guideline]:" + schema_guideline_per_timeseries_feature + "\n[Relation information]:\n" + foreign_key_schema + "\n[Previous SQL]:\n" + pre_sql  + "\n[Error Feedback]:\n" + error_msg
        
    return prompt #+ nondef_timeseries_feature_prompt

def candidate_feature_sql_prompt_maker(
    definition_schema : dict, 
    definition_table : str, 
    foreign_key_schema : str, 
    ts_schema : str,
    schema_guideline: str,
    sql_history: dict,
    ) -> str :
    instruction = """\nQ: Using the provided Schema (tables, columns, values) and [Schema Guideline], generate a single PostgreSQL query to obtain columns 'unique feature identifier code (if exists)', 'feature name' and 'unit' from [Definition table].

To make a SQL query, follow these steps:
1. Identify tables that appear both in the [Feature Schema] and [Definition Schema]. Avoid using tables that are not in 'both' [Definition Schema] and [Feature Schema].
2. For identified table, ensure to obtain columns in the order of 'unique feature identifier code (if exists)','feature name' and 'unit'.
    - **Important** Obtain 'unique feature identifier code' that represents feature types or items, but not row-level event-level IDs.
    - **Important** Do NOT include the actual measurement value column.
3. If the table does not have a column about unit, look up the [Relation information] and [Feature Schema] to find any table that could provide the unit information via a foreign key relationship (e.g.,measurement_id, machine_id, etc.). Then JOIN that table to retrieve the correct unit column.
4. Use consistent aliasing for each table (e.g.,table AS alias) and ensure all aliases used in the SELECT clause are defined in the FROM clause.
5. Only use JOIN when necessary.
    - **Important** Do not JOIN between each tables in [Definition Schema]. 
    - Use JOIN only when there is a connection (foreign key) in [Relation Information].
6. Ensure the 'feature name' represents name of vital sign or lab test but not type or code number.
7. The order of the columns in the SELECT clause must be 'feature code number', 'feature name', and 'unit'.

Note for SQL formation:
1. Your final answer for each query must start from 'SELECT' (do not include any code fences or explanation).
2. **Important** Use DISTINCT to eliminate duplicate feature names. Return only one row per unique feature name.
3. When [Failed SQL] exists, carefully review the [Failed SQL] and [Error Feedback] to identify the cause of the failure and avoid the same mistake in the next SQL generation.

Output Format: Provide exactly:
<think>
[Clearly and concisely explain your reasoning behind your SQL query generation.]
</think>

<SQL_queries>
[SQL_QUERY_HERE]
</SQL_queries>
"""
    ts_schema = format_schema_for_llm(ts_schema)
    prompt = instruction + "\n[Definition table]\n" + definition_table + "\n\n[Definition Schema]\n" + definition_schema + "\n\n[Feature schema]:\n" + ts_schema + "\n[Schema Guideline]:" + schema_guideline + "\n\n[Relation information]:\n" + foreign_key_schema
    if bool(sql_history):
        sql_history_prompt = ""
        for i, sql in enumerate(sql_history):
            sql_history_prompt += f"\n\n[Failed SQL]-{i+1}: {sql}\n[Error Feedback]-{i+1}: {sql_history[sql]}"
        prompt += ("""\n*Important* Each [Failed SQL] shows a previously incorrect SQL query, and the corresponding [Error Feedback] explains why it failed. Review them carefully to avoid repeating the same mistakes when generating the next SQL.\n""")
        prompt += "\n" + sql_history_prompt
        # print(f"{sql_history_prompt}")
    # exit()
    return prompt

def find_synonyms_prompt_maker(feature_synonyms_pairs_list:str, targeting_feature:str):
    instruction = """Identify tuples about [Targeting Feature] within [Feature synonyms]. Compare all [Feature synonyms] one by one with [Targeting Feature],reflecting medical knowledge.
Response Guidelines:
Return only the matching tuple(s) for each measurement.
If a measurement has no matching tuples, return "None" instead of an empty list.
###Output Format : follow the formate of [Feature synonyms]
Do not include any additional text, explanations, metadata, or formatting beyond the required tuples.\n"""
    prompt = instruction + "[Feature synonyms]\n " + feature_synonyms_pairs_list + "\n[Targeting Feature]\n " + str(targeting_feature) + "\n[Selected Feature Synonyms Tuple]\n"
    return prompt

def matching_prob_prompt_maker(feature_synonyms_pairs_list, targeting_feature, threshold):
    instruction = """Compare the [Targeting Feature] with each tuple in [Synonyms] using medical knowledge. 
Assign probabilities within a range of 0 to 100 for each pair, ensuring the comparisons reflect how strongly each Synonym belongs to the same category or type as the specific Targeting Feature. 
Only include similarity probabilities that is equal or higher than the specified threshold.

Formatting Requirements:
1. Targeting Features: Each result must begin with the name of the Targeting Feature, followed by a colon (:).
2. Synonyms and Probabilities: After the colon, include a dictionary where:
   - Each Synonym is a key (tuple format).
   - The assigned probability (0 to 100) reflects how strongly the Synonym belongs to the same category or type as the Targeting Feature.
   - Only unique synonym tuples should be included (i.e., do not repeat the same synonym multiple times).
3. Key-Value Separators: Use double-pipes (||) to separate key-value pairs inside the dictionary.
4. No Additional Text: The output must strictly adhere to this format, and do not include code delimiters (e.g., ```sql), comments, or any additional text.

Example 1:
[Targeting Features]: CRP
[Synonyms]: ('C-reactive protein',), ('Pulse',), ('Serum Glucose',), ('SBP',)
Similarity Threshold: 1
[Similarity Probabilities]: CRP: {('C-reactive protein',): 99|| ('SBP',): 3}

Example 2:
[Targeting Features]: Heart Rate
[Synonyms]: ('C-reactive protein',), ('Pulse',), ('Serum Glucose',), ('SBP',)
Similarity Threshold: 80
[Similarity Probabilities]: Heart Rate: {('Pulse',): 95}

Example 3:
[Targeting Features]: SBP (mmHg)
[Synonyms]: ('Systolic Blood Pressure', 'mmHg'), ('Diastolic Blood Pressure', 'mmHg'), ('Heart Rate', 'bpm'), ('Serum Glucose', 'mg/dL')
Similarity Threshold: 50
[Similarity Probabilities]: SBP: {('Systolic Blood Pressure', 'mmHg'): 98 || ('Diastolic Blood Pressure', 'mmHg'): 60}\n\n"""
    prompt = instruction + "[Targeting Features]: " + str(targeting_feature) + "\n[Synonyms]: " + feature_synonyms_pairs_list + "\nSimilarity Threshold: " + str(threshold) + "\n[Similarity Probabilities]: "
    return prompt

def matching_noprob_prompt_maker(candidate_features_list, targeting_features, threshold):
    instruction = """Compare each of the [Targeting Features] with each tuple from [Candidate Features] using your medical knowledge. 
Assign similarity probabilities within a range of 0 to 100 for each pair, ensuring the comparisons reflect the degree to which each Candidate Feature aligns with the specific Targeting Feature. 
Only include [Candidate Features] tuple(s) with similarity probabilities that is equal or higher than the specified Similarity Threshold.

Formatting Requirements:
1. Targeting Features: Each result must begin with the name of the Targeting Feature, followed by a colon (:).
2. Candidate Features and Probabilities: After the colon, include a dictionary where:
   - Each Candidate Feature is a key (tuple format).
   - The assigned probability (0 to 100 integer only) reflects how strongly the Candidate Feature belongs to the same category or type as the Targeting Feature.
   - Only unique candidate feature tuples should be included (i.e., do not repeat the same candidate feature multiple times).
3. Key-Value Separators: Use double-pipes || to separate key-value pairs inside the dictionary.
4. Separator: Use a semicolon (;) to separate results for each Targeting Feature.
5. No Additional Text: The output must strictly adhere to this format, and do not include code delimiters (e.g., ```sql), comments, or any additional text.

Example Input:
[Targeting Features]:
['CRP', 'Heart Rate', 'Blood Glucose', 'Systolic Blood Pressure']
[Candidate Features]:
[('C-reactive protein',), ('Pulse',), ('Serum Glucose',), ('SBP',)]
Similarity Threshold: 10
[Similarity Probabilities]:
CRP: {('C-reactive protein',): 95}; Heart Rate: {('C-reactive protein',): 10|| ('Pulse',): 90|| ('Serum Glucose',): 10|| ('SBP',): 15}; Blood Glucose: {('C-reactive protein',): 15|| ('Serum Glucose',): 90|| ('SBP',): 20}; Systolic Blood Pressure: {('C-reactive protein',): 10|| ('Pulse',): 10|| ('Serum Glucose',): 20|| ('SBP',): 95};\n\n"""
    prompt = instruction + "[Targeting Features]:\n " + str(targeting_features) + "\n[Candidate Features]:\n " + candidate_features_list + "\nSimilarity Threshold: " + str(threshold) + "\n[Similarity Probabilities]:\n"
    return prompt

def multi_synonyms_probs_prompt_maker(synonyms_pairs_list, targeting_features):
    instruction = """Compare each of the [Targeting Features] with each tuple in [Synonyms] using medical knowledge. Assign similarity probabilities within a range of 0 to 100 for each pair, ensuring the comparisons reflect the degree to which each Synonym aligns with the specific Targeting Feature. 

Formatting Requirements:
1. Targeting Features: Each result must begin with the name of the Targeting Feature, followed by a colon (:).
2. Synonyms and Probabilities: After the colon, include a dictionary where:
   - Each Synonym is a key (tuple format).
   - The assigned probability (0 to 100) reflects how strongly the Synonym belongs to the same category or type as the Targeting Feature.
   - Only unique synonym tuples should be included (i.e., do not repeat the same synonym multiple times).
3. Key-Value Separators: Use double-pipes || to separate key-value pairs inside the dictionary.
4. Result Separator: Use a semicolon (;) to separate results for each Targeting Feature.
5. No Additional Text: The output must strictly adhere to this format, and do not include code delimiters (e.g., ```sql), comments, or any additional text.

Example Input:
[Targeting Features]:
['CRP', 'Heart Rate', 'Blood Glucose', 'Systolic Blood Pressure']
[Synonyms]:
[('C-reactive protein',), ('Pulse',), ('Serum Glucose',), ('SBP',)]
[Similarity Probabilities]:
CRP: {(‘C-reactive protein’,)=100||(‘Pulse’,)=0||(‘Serum Glucose’,)=0||(‘SBP’,)=0}; Heart Rate: {(‘C-reactive protein’,)=0||(‘Pulse’,)=95||(‘Serum Glucose’,)=0||(‘SBP’,)=0}; Blood Glucose: {(‘C-reactive protein’,)=0||(‘Pulse’,)=0||(‘Serum Glucose’,)=100||(‘SBP’,)=0}; Systolic Blood Pressure: {(‘C-reactive protein’,)=0||(‘Pulse’,)=0||(‘Serum Glucose’,)=0||(‘SBP’,)=100};\n\n"""
    prompt = instruction + "[Targeting Features]:\n " + str(targeting_features) + "\n[Synonyms]:\n " + synonyms_pairs_list + "\n[Similarity Probabilities]:\n"
    return prompt

def unit_changer(main_unit, sub_unit, feature_name):
    unit_prompt = """The same feature is provided under different names. 
The [Sub unit] has a similar feature name, unit information, and similarity score of the feature. 
The [Main unit] contains standard unit information for each feature. Using this standard unit information, I want to convert the unit information of [Sub unit].
If [Main unit] or [Sub unit] doesn't have unit information, just get "x" in formula that means use same value. 
If an additional conversion formula is required, consider the conversion formula expressed with "x" and reflect it in the output.
Please output the result in the following format. And do not change (sub unit set) from [Sub unit].
No Additional Text: The output must strictly adhere to this format, and do not include code delimiters (e.g., ```sql), comments, or any additional text.

Format: "{(sub unit set): (formula), (sub unit set): (formula)}" """
    prompt = unit_prompt+'\n[Feature]'+feature_name+'\n[Main unit]'+str(main_unit)+'\n[Sub unit]'+str(sub_unit)
    return prompt

def def_timeseries_feature_extractor_sql_maker(
    matching_feature : List[Any],
    definition_table: str, 
    foreign_key_schema: str,
    schema_name: str,
    def_timeseries_feature_schema: str,
    schema_guideline_per_timeseries_feature: str,
    pre_sql: Optional[str],
    error_msg: Optional[str]):
    instruction = """
Q: Using [Schema_name], [Selected Schema], [Definition Schema], and [Matching feature], write a PostgreSQL query to retrieve the 'item name', 'unit', 'sampled time' and 'value' of the [Matching feature] feature. 
Additionally, consider the [Schema Guideline] to use appropriate tables and columns in [Selected Schema] to make correct SQL query.
Output is only the SQL query as plain text. Do not include code delimiters (e.g., ```sql), comments, or any additional text.

Note:
- Make SQL from [Definition Schema].
- [Matching feature] means feature name and unit information, use all information to make 'WHERE'.
- [Matching feature] is only value in [Definition Schema].
- [Selected Schema] is the schema that contains the value and sampled time information and possible item name or code number.
- [Definition Schema] and [Selected DB Schema] can be related through foreign key [Relation Information] or same. If same, you can directly use [Matching feature] in [Selected DB Schema]."""
    if len(matching_feature) > 1:
        instruction = instruction + "\n    - If [Matching feature] has more than one feature. Make SQL for each [Matching feature], then integrate into one SQL."
    if pre_sql is None:
        prompt = instruction + "\n[Matching feature]:" + str(matching_feature) + "\n[Schema_name]: " + schema_name + "\n[Selected Schema]:\n" + def_timeseries_feature_schema + "\n[Schema Guideline]:" + schema_guideline_per_timeseries_feature + "\n[Definition Schema]:\n" + definition_table + "\n[Relation information]:\n" + foreign_key_schema + "\n[SQL]: "
    else:
        instruction = instruction + "Feedback Note : [Previous SQL] and [Error Feedback] are failed cases of SQL and error feedback. Carefully exmaine [Error Feedback] and avoid [Previous SQL] to generate correct SQL.\n"
        prompt = instruction + "\n[Matching feature]:" + str(matching_feature) + "\n[Schema_name]: " + schema_name + "\n[Selected Schema]:\n" + def_timeseries_feature_schema + "\n[Schema Guideline]:" + schema_guideline_per_timeseries_feature + "\n[Definition Schema]:\n" + definition_table + "\n[Relation information]:\n" + foreign_key_schema + "\n[Previous SQL]:\n" + pre_sql  + "\n[Error Feedback]:\n" + error_msg + "\n[SQL]: "
    return prompt

def integrate_demo_target_timeseries_sql(
    requested_timeseries_schema_guideline: str,
    demographic_schema_guideline: str,
    demographic_feature_sql: str, 
    timeseries_feature_sql: str,
    target_time_range : str,
    unit_translate_formula = None ,
    standard_unit = None, 
    pre_sql = None,
    error_msg = None):
    
    instruction = """\nQ: Integrate three SQL codes [Demography feature SQL], [Target feature SQL], [Timeseries feature SQL] into one SQL. Make sure to satisfy [Note] to make optimized query for Large dataset.
[Note]
- Do not chage column name or alis, just use same SELECT information.
- Avoid Redundant Joins
- Use CTEs for Clarity & Indexing
- Push Filters Earlier

Also integrate three schema guidelines [Selected Schema Guideline], [Demographic Schema Guideline], [Target Schema Guideline] in order to integrate the three SQL queries and generate final correct SQL.
Use early reduction of data volume to optimize SQL query short. Before using JOIN, apply WHERE limit condition to get data faster. As possible, Place filter conditions at the top of the subquery.
Output is only one SQL query as plain text according to the output format. Do not include code delimiters (e.g., ```sql), comments, or any additional text.

Only select values that statisfy [Time condition] that means interval time between 'feature observation' and 'ICU admission' time. Do not select values that don't have time information."""
    
    output_format = """
Present your final output in the output format:
<think>
[Explain your thought process here, discussing how you considered the three schema guidelines and three SQL queries to generate SQL.]
</think>

<sql_query>
[The final generated SQL query to extract demographic, target, and timeseries features of patients]
</sql_query>
Do not add any explanations or additional text outside of the specified output format.
    """
    
    if unit_translate_formula:
        instruction += "\nModeify [Timeseries feature sql] to change SQL output resul's unit information to [Standard unit]. So change value by [Unit translate formula]." 
        instruction += "\n[Demography feature sql]\n" + str(demographic_feature_sql) + "\n[Timeseries feature sql]\n" + str(timeseries_feature_sql) + "\n[Time condition] :" + target_time_range + "\n[Unit translate formula]" + str(unit_translate_formula) + "\n[Standard unit]" + str(standard_unit)
    else:
        instruction += "\n[Demography feature sql]\n" + str(demographic_feature_sql) + "\n[Timeseries feature sql]\n" + str(timeseries_feature_sql) + "\n[Time condition] :" + target_time_range 
    
    if pre_sql is None:
        prompt = instruction + "\n\n[Selected Schema Guideline]" + str(requested_timeseries_schema_guideline) + "\n[Demographic Schema Guideline]" + " ".join(demographic_schema_guideline) + output_format
    else:
        prompt = instruction + "\n\n[Selected Schema Guideline]" + str(requested_timeseries_schema_guideline) + "\n[Demographic Schema Guideline]" + " ".join(demographic_schema_guideline) + "\nFeedback Note : [Previous SQL] and [Error Feedback] are failed cases of SQL and error feedback. Carefully exmaine [Error Feedback] and avoid [Previous SQL] to generate correct SQL.\n" + "\n\n[Previous SQL]:\n" + pre_sql  + "\n\n[Error Feedback]:\n" + error_msg + output_format
    return prompt

def llm_api_generation(prompt, model="claude-3-5-sonnet-20240620", temperature=0.0):
    retry_count=4
    # claude-3-5-sonnet-20240620
    # claude-3-7-sonnet-20250219
    for attempt in range(retry_count):
        print(f"Attempt {attempt+1}/{retry_count}")
        try:    
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=temperature,
                top_p=0.9,
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

async def Claude_long_generation(prompt_all, model="claude-3-5-sonnet-20240620"):
    max_retries = 3
    throttle_delay = 0.2  

    client = anthropic.Anthropic(
        api_key=""
    )

    async def call_claude(prompt):
        for attempt in range(max_retries):
            try:
                print(f"[Claude] Prompt: {prompt[:50]}... (attempt {attempt+1})")
                message = await asyncio.to_thread(
                    client.messages.create,
                    model=model,
                    max_tokens=2048,
                    temperature=0.0,
                    top_p=0.9,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text.replace("\\n", "\n")
            except anthropic.RateLimitError as e:
                print(f"Rate limit error (attempt {attempt+1}): {e}")
                await asyncio.sleep(30)
            except Exception as e:
                print(f"Claude API Error (attempt {attempt+1}): {e}")
                await asyncio.sleep(3)
        raise RuntimeError(f"Failed to generate response for prompt: {prompt[:50]}...")

    tasks = []
    for prompt in prompt_all:
        tasks.append(call_claude(prompt))
        await asyncio.sleep(throttle_delay)

    return await asyncio.gather(*tasks)

def search_walk(info):
    searched_list = []
    root_path = info.get('path')
    extension = info.get('extension')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extension:
                list_file = ('%s/%s' % (path, filename))
                searched_list.append(list_file)

    if searched_list:
        return searched_list
    else:
        return False