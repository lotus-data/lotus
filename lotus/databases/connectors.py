import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError


class DatabaseConnector:
    @staticmethod
    def load_from_db(connection_url: str, query: str) -> pd.DataFrame:
        """
        Executes SQl query and returns a pandas dataframe

        Args:
            connection_url (str): The connection url for the database
            query (str): The query to execute

        Returns:
            pd.DataFrame: The result of the query
        """
        try:
            engine = create_engine(connection_url)
            with engine.connect() as conn:
                return pd.read_sql(query, conn)
        except OperationalError as e:
            raise ValueError(f"Error connecting to database: {e}")
