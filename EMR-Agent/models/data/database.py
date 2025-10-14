from psycopg2 import connect
from typing import List, Dict, Any, Optional
from config import DATABASE_CONFIGS

class DatabaseManager:
    def __init__(self, target_database: str, batch_size: int, user: str = "postgres", password: str = ""):
        """Initialize database connection manager.
        
        Args:
            target_database: Name of the target database
            batch_size: Number of rows to fetch at a time
            user: Database username
            password: Database password
        """
        self.target_database = target_database
        self.user = user
        self.password = password
        self.batch_size = batch_size
        
        # Get database configuration
        db_config = DATABASE_CONFIGS.get(target_database)
        if not db_config:
            raise ValueError(f"Unsupported database: {target_database}")
            
        self.dbname = db_config["dbname"]
        self.host = db_config["host"]
        self.port = db_config["port"]
        self.conn = None
        self.cursor = None

    def connect(self, query: str, limit_row_n: bool = True, row_limit: int = 100000) -> List[Any]:
        """Execute query and fetch results.
        
        Args:
            query: SQL query to execute
            limit_row_n: Whether to limit number of rows returned
            row_limit: Maximum number of rows to return if limit_row_n is True
            
        Returns:
            List of query results
        """
        self.conn = connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        
        results = []
        with self.conn.cursor(name='server_side_cursor') as self.cursor:
            self.cursor.itersize = self.batch_size
            self.cursor.execute(query)
            
            while True:
                rows = self.cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                results.extend(rows)
                if limit_row_n and len(results) >= row_limit:
                    break
                    
        return results

    def close(self):
        """Close database connection and cursor."""
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def execute_with_retry(self, query: str, max_retries: int = 3, retry_delay: int = 2) -> Optional[List[Any]]:
        """Execute query with retry logic.
        
        Args:
            query: SQL query to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            Query results if successful, None if all retries fail
        """
        import time
        
        for attempt in range(max_retries):
            try:
                results = self.connect(query)
                return results
            except Exception as e:
                print(f"Error executing query (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise 