from examples.op_examples.openhouse_connector import OpenHouse
from openhouse_config import connection_params

openhouse = OpenHouse(connection_params)
df = openhouse.table("openhouse.u_openhouse.lotus_test")
print(df.head())  # Now connects and fetches data
print(df.describe())  # Uses cached data from previous fetch