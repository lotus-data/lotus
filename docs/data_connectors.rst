data_connectors
=================

Overview
---------
LOTUS provides additional methods to load your data if your data is not originally in a pandas df format.
Current data connections include most SQL databases supported by SQLalchemy and any s3 serivice.

Motivation
-----------
Data connectors offer additional ways to import data to be used by LOTUS's semantic Operators

DB Example
-----------
.. code-block:: python

    import lotus
    from lotus.data_connectors import DataConnector
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    query = "SELECT * FROM movies"
    df = DataConnector.load_from_db("sqlite:///example_movies.db", query=query)

    user_instruction = "{title} that are science fiction"
    df = df.sem_filter(user_instruction)
    print(df)

Output:

+----+--------------+-------------------+----------+----------------+-----------------------------------------------------+
|    | title        | director          | rating   | release_year   | description                                         |
+====+==============+===================+==========+================+=====================================================+
|  0 | The Matrix   | Wachowskis        | 8.7      | 1999           | A hacker discovers the reality is simulated.        |
+----+--------------+-------------------+----------+----------------+-----------------------------------------------------+
|  1 | Inception    | Christopher Nolan | 8.8      | 2010           | A thief enters dreams to steal secrets.             |
+----+--------------+-------------------+----------+----------------+-----------------------------------------------------+
|  2 | Interstellar | Christopher Nolan | 8.6      | 2014           | A team travels through a wormhole to save humanity. |
+----+--------------+-------------------+----------+----------------+-----------------------------------------------------+


s3 Example
-----------
.. code-block:: python

    import lotus
    from lotus.data_connectors import DataConnector
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    service_configs = {
        "minio": {
            "aws_access_key": "accesskey",
            "aws_secret_key": "secretkey",
            "region": None,
            "bucket": "test-bucket",
            "file_path": "data/test.csv",
            "protocol": "http",
            "endpoint_url": "http://localhost:9000",
        }
    }

    # Get configuration for selected service
    service = "minio"
    service_config = service_configs[service]

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
    user_instruction = "{title} is science fiction movie"
    df = df.sem_filter(user_instruction)
    print(df)

Output:

+----+--------------+-------------------+----------+----------------+-----------------------------------------------------+
|    | title        | director          | rating   | release_year   | description                                         |
+====+==============+===================+==========+================+=====================================================+
|  0 | The Matrix   | Wachowskis        | 8.7      | 1999           | A hacker discovers the reality is simulated.        |
+----+--------------+-------------------+----------+----------------+-----------------------------------------------------+
|  1 | Inception    | Christopher Nolan | 8.8      | 2010           | A thief enters dreams to steal secrets.             |
+----+--------------+-------------------+----------+----------------+-----------------------------------------------------+
|  2 | Interstellar | Christopher Nolan | 8.6      | 2014           | A team travels through a wormhole to save humanity. |
+----+--------------+-------------------+----------+----------------+-----------------------------------------------------+

Required DB Parameters
------------------------
- **connection_url** : The connection url for the database
- **query** : The query to execute

Required s3 Paramaters
-----------------------
- **aws_access_key** : The AWS access key (None for Public Access)
- **aws_secret_key** : The AWS secret key (None for Public Access)
- **region** : The AWS region
- **bucket** : The S3 bucket
- **file_path** : The path to the file in S3
- **endpoint_url** : The Minio endpoint URL. Default is None for AWS s3
- **protocol** : The protocol to use (http for Minio and https for R2). Default is "s3"

