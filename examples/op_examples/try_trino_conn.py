import trino
import pandas as pd

#TODO: Fix the connection params to make it work
conn=trino.dbapi.connect(
    host='******.corp.******.com',
    port=***,
    user='lesun',
    catalog='openhouse',
    http_scheme='https',
    # TODO: Can we avoid using the password here? 
    auth=trino.auth.BasicAuthentication("", ""),
)
cur = conn.cursor()
cur.execute("SELECT * from openhouse.db.table")

columns = [desc[0] for desc in cur.description]
# Convert results to DataFrame
df = pd.DataFrame(cur.fetchall(), columns=columns)
# Now you can work with the DataFrame
print(df)
