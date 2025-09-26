import os

# Database configurations
DATABASE_CONFIGS = {
    "mimic3": {
        "dbname": "mimic",
        "port": "5432",
        "host": "43.246.152.140",
        "manual_path": "./data/mimic3_overview.txt",
        "primary_key_path": "./data/mimic3_primary_keys.txt"
    },
    "mimic4": {
        "dbname": "mimiciv",
        "port": "5432",
        "host": "43.246.152.154",
        "manual_path": "./data/mimic4_overview.txt",
        "primary_key_path": "./data/mimic4_primary_keys.txt"
    },
    "eicu": {
        "dbname": "eicu",
        "port": "5433",
        "host": "43.246.152.140",
        "manual_path": "./data/eicucrd_overview.txt",
        "primary_key_path": "./data/eicu_primary_keys.txt"
    },
    "hirid": {
        "dbname": "hirid",
        "port": "5433",
        "host": "43.246.152.154",
        "manual_path": "./data/hirid_overview.txt",
        "primary_key_path": "./data/hirid_primary_keys.txt"
    },
    "sicdb": {
        "dbname": "sicdb",
        "port": "5434",
        "host": "43.246.152.154",
        "manual_path": "./data/sicdb_overview.txt",
        "primary_key_path": "./data/sicdb_primary_keys.txt"
    }
}

# SQL queries
SQL_QUERIES = {
    "schema_extract": """
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
        """,
    
    "primary_key_extract": """
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
        """,
    
    "foreign_key_extract": """
        SELECT 
            kcu.column_name,
            tc.table_name,
            ccu.table_name
        FROM 
            information_schema.table_constraints AS tc 
        JOIN 
            information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN 
            information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE 
            tc.table_schema = 'Schema_Name'
            AND tc.constraint_type = 'FOREIGN KEY';
        """
}

# Default parameters
DEFAULT_PARAMS = {
    "batch_size": 500,
    "example_cell_size": 10,
    "text_cell_maxlen": 50,
    "max_retries": 5,
    "retry_delay": 2,
    "similarity_threshold": 85
}
