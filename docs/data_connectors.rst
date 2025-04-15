Database Connectors
=================

Overview
---------
LOTUS' data connectors let you seamlessly load data from external stores (e.g. a SQL database) so that you can run LOTUS programs over them.
Current data connections include SQL databases supported by `SQLAlchemy`_ and any S3 serivice.


.. _SQLAlchemy: https://docs.sqlalchemy.org/en/14/dialects/

Intstallation
--------
To get started, you will need to install the lotus submodule as follows::

    pip install lotus[data_connectors]


Example: Loading from SQLite
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

Example: Loading from Postgres
------------
.. code-block:: python

    import lotus
    from lotus.data_connectors import DataConnector
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    query = "SELECT * FROM movies WHERE rating > 5.0"
    df = DataConnector.load_from_db("postgresql+psycopg2://user:password@host:port/database", query=query)

    user_instruction = "{title} that are science fiction"
    df = df.sem_filter(user_instruction)
    print(df)

Example: Loading from Snowflake
---------------
.. code-block:: python

    import lotus
    from lotus.data_connectors import DataConnector
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    query = "SELECT * FROM movies WHERE genre = 'Horror'"
    df = DataConnector.load_from_db("snowflake://<user>:<password>@<account>/<database>/<schema>?warehouse=<warehouse>&role=<role>", query=query)

    user_instruction = "{title} that are science fiction"
    df = df.sem_filter(user_instruction)
    print(df)

Example: Loading from Google Big Query
--------------------------
.. code-block:: python

    import lotus
    from lotus.data_connectors import DataConnector
    from lotus.models import LM

    lm = LM(model="gpt-4o-mini")
    lotus.settings.configure(lm=lm)

    query = "SELECT date, MAX(title) as title, AVG(rating) as rating FROM movies GROUPBY date ORDERBY rating desc"
    df = DataConnector.load_from_db("bigquery://my-gcp-project/my_dataset", query=query)

    user_instruction = "{title} that are science fiction"
    df = df.sem_filter(user_instruction)
    print(df)

Example: Loading from S3
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

