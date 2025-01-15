from pymongo import MongoClient
from sqlalchemy import create_engine


class DatabaseConnector:
    def __init__(self):
        self.sql_engine = None
        self.nosql_client = None

    def connect_sql(self, connection_url: str):
        """Connect to SQL database"""
        try:
            self.sql_engine = create_engine(connection_url)
            return self
        except Exception as e:
            raise ConnectionError(f"Error connecting to SQL database: {e}")

    def connect_nosql(self, connection_url: str):
        """Connect to MongoDB NoSQL database"""
        try:
            self.nosql_client = MongoClient(connection_url)
            return self
        except Exception as e:
            raise ConnectionError(f"Error connecting to NoSQL database: {e}")

    def close_connections(self):
        """Close SQL and NoSQL connections"""
        if self.sql_engine:
            self.sql_engine.dispose()
        if self.nosql_client:
            self.nosql_client.close()
