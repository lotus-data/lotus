import boto3
import pandas as pd

import lotus
from lotus.data_connectors import DataConnector
from lotus.models import LM

lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# Service configurations
service_configs = {
    "aws": {
        "aws_access_key": "your_aws_access_key",
        "aws_secret_key": "your_aws_secret_key",
        "region": "us-east-1",
        "bucket": "your-aws-bucket",
        "file_path": "data/test.csv",
        "protocol": "s3",
        "endpoint_url": None,
    },
    "minio": {
        "aws_access_key": "accesskey",
        "aws_secret_key": "secretkey",
        "region": None,
        "bucket": "test-bucket",
        "file_path": "data/test.csv",
        "protocol": "http",
        "endpoint_url": "http://localhost:9000",
    },
    "cloudfare_R2": {
        "aws_access_key": "your_r2_access_key",
        "aws_secret_key": "your_r2_secret_key",
        "region": None,
        "bucket": "your-r2-bucket",
        "file_path": "data/test.csv",
        "protocol": "https",
        "endpoint_url": "https://<account_id>.r2.cloudflarestorage.com",
    },
}

# Get configuration for selected service
service = "minio"
service_config = service_configs[service]

# Create Test Data
test_data = pd.DataFrame(
    {
        "title": ["The Matrix", "The Godfather", "Inception", "Parasite", "Interstellar", "Titanic"],
        "description": [
            "A hacker discovers the reality is simulated.",
            "The rise and fall of a powerful mafia family.",
            "A thief enters dreams to steal secrets.",
            "A poor family schemes to infiltrate a rich household.",
            "A team travels through a wormhole to save humanity.",
            "A love story set during the Titanic tragedy.",
        ],
    }
)
csv_data = test_data.to_csv(index=False)

# Connect to s3 Service and load data
try:
    session = boto3.Session(
        aws_access_key_id=service_config["aws_access_key"],
        aws_secret_access_key=service_config["aws_secret_key"],
        region_name=service_config["region"],
    )
    s3 = session.resource("s3", endpoint_url=service_config["endpoint_url"])

    try:
        s3.create_bucket(Bucket=service_config["bucket"])
        print(f"Bucket '{service_config['bucket']}' created successfully.")
    except s3.meta.client.exceptions.BucketAlreadyOwnedByYou:
        print(f"Bucket '{service_config['bucket']}' already exists.")
    except Exception as e:
        print(f"Error creating bucket: {e}")

    s3.Bucket(service_config["bucket"]).put_object(Key=service_config["file_path"], Body=csv_data)
except Exception as e:
    print(f"Error connecting to s3 service: {e}")


# loading data from s3
df = DataConnector.load_from_s3(
    aws_access_key=(service_config["aws_access_key"]),
    aws_secret_key=(service_config["aws_secret_key"]),
    region=str(service_config["region"]),
    bucket=str(service_config["bucket"]),
    file_path=str(service_config["file_path"]),
    endpoint_url=(service_config["endpoint_url"]),
    protocol=str(service_config["protocol"]),
)
df = df.sem_filter("the {title} is science fiction")
print(df)
