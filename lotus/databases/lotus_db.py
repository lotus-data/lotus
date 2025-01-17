from typing import Optional

import pandas as pd

import lotus


class LotusDB:
    def __init__(self, connector):
        self.connector = connector

    def query(
        self, query: str, db_type: str = "sql", db_name: Optional[str] = None, collection_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Executes query and returns a pandas dataframe

        Args:
            query (str): The query to execute
            db_type (str, optional): The type of database to use. Defaults to 'sql'.

        Returns:
            pd.DataFrame: The result of the query

        """
        try:
            if db_type == "sql":
                if not isinstance(query, str):
                    raise ValueError("Query must be a string")
                lotus.logger.debug("Executing SQL Query")
                return pd.read_sql(query, self.connector.sql_engine)
            elif db_type == "nosql":
                if not collection_name or not db_name:
                    raise ValueError("Collection name and database is required for NoSQL database")
                collection = self.connector.nosql_client[db_name][collection_name]
                results = collection.find(query)
                return pd.DataFrame(list(results))
            else:
                raise ValueError("Invalid database type")

        except Exception as e:
            lotus.logger.error(f"Error executing query: {e}")
            raise
