from io import BytesIO, StringIO

import boto3
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError


class DataConnector:
    @staticmethod
    def load_from_db(connection_url: str, query: str) -> pd.DataFrame:
        """
        Executes SQl query from supported databases on SQlAlchemy and returns a pandas dataframe

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

    @staticmethod
    def load_from_s3(
        aws_access_key: str, aws_secret_key: str, region: str, bucket: str, file_path: str
    ) -> pd.DataFrame:
        """
        Loads a pandas DataFrame from an S3 object.

        Args:
            aws_access_key (str): The AWS access key
            aws_secret_key (str): The AWS secret key
            region (str): The AWS region
            bucket (str): The S3 bucket
            file_path (str): The path to the file in S3

        Returns:
            pd.DataFrame: The loaded DataFrame

        """
        try:
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region,
            )
        except Exception as e:
            raise ValueError(f"Error creating boto3 session: {e}")

        s3 = session.resource("s3")
        s3_obj = s3.Bucket(bucket).Object(file_path)
        data = s3_obj.get()["Body"].read()

        file_type = file_path.split(".")[-1].lower()

        file_mapping = {
            "csv": lambda data: pd.read_csv(StringIO(data.decode("utf-8"))),
            "json": lambda data: pd.read_json(StringIO(data.decode("utf-8"))),
            "parquet": lambda data: pd.read_parquet(BytesIO(data)),
            "xlsx": lambda data: pd.read_excel(BytesIO(data)),
        }

        try:
            return file_mapping[file_type](data)
        except KeyError:
            raise ValueError(f"Unsupported file type: {file_type}")
