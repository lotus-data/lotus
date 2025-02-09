from io import BytesIO, StringIO
from typing import Optional

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
        aws_access_key: Optional[str],
        aws_secret_key: Optional[str],
        region: str,
        bucket: str,
        file_path: str,
        endpoint_url: Optional[str] = None,
        protocol: str = "s3",
    ) -> pd.DataFrame:
        """
        Loads a pandas DataFrame from an S3-compatible service.

        Args:
            aws_access_key (str): The AWS access key (None for Public Access)
            aws_secret_key (str): The AWS secret key (None for Public Access)
            region (str): The AWS region
            bucket (str): The S3 bucket
            file_path (str): The path to the file in S3
            endpoint_url (str): The Minio endpoint URL. Default is None for AWS s3
            protocol (str): The protocol to use (http for Minio and https for R2). Default is "s3"

        Returns:
            pd.DataFrame: The loaded DataFrame

        """
        try:
            if aws_access_key is None and aws_secret_key is None:
                session = boto3.Session(region_name=region)
            else:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=region if protocol == "s3" and endpoint_url is None else None,
                )
        except Exception as e:
            raise ValueError(f"Error creating boto3 session: {e}")

        s3 = session.resource("s3", endpoint_url=endpoint_url)
        s3_obj = s3.Bucket(bucket).Object(file_path)
        data = s3_obj.get()["Body"].read()

        file_type = file_path.split(".")[-1].lower()

        file_mapping = {
            "csv": lambda data: pd.read_csv(StringIO(data.decode("utf-8"))),
            "json": lambda data: pd.read_json(StringIO(data.decode("utf-8"))),
            "parquet": lambda data: pd.read_parquet(BytesIO(data)),
            "xlsx": lambda data: pd.read_excel(BytesIO(data)),
            "txt": lambda data: pd.read_csv(StringIO(data.decode("utf-8")), sep="\t"),
        }

        try:
            return file_mapping[file_type](data)
        except KeyError:
            raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Error loading from S3-compatible service: {e}")
