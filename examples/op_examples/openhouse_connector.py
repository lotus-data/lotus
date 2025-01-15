from typing import Optional, Any
import pandas as pd
import trino

class LazyTable:
    def __init__(self, table_name: str, connection_params: dict):
        self._table_name = table_name
        self._connection_params = connection_params
        self._connection = None
        self._query = f"SELECT * FROM {table_name}"
        self._df: Optional[pd.DataFrame] = None

    def _ensure_connection(self):
        if self._connection is None:
            self._connection = trino.dbapi.connect(**self._connection_params)

    def _ensure_data(self):
        if self._df is None:
            self._ensure_connection()
            cur = self._connection.cursor()
            cur.execute(self._query)
            columns = [desc[0] for desc in cur.description]
            # Convert results to DataFrame
            self._df = pd.DataFrame(cur.fetchall(), columns=columns)
        return self._df

    # Implement pandas DataFrame methods that will trigger evaluation
    def head(self, n: int = 5) -> pd.DataFrame:
        return self._ensure_data().head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        return self._ensure_data().tail(n)
    
    def describe(self) -> pd.DataFrame:
        return self._ensure_data().describe()

    # You can override other pandas methods as needed
    def __getattr__(self, name: str) -> Any:
        # This will be called for any attribute not explicitly defined
        def method(*args, **kwargs):
            # Ensure we have the data and then call the pandas method
            df = self._ensure_data()
            if hasattr(df, name):
                return getattr(df, name)(*args, **kwargs)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return method

class OpenHouse:
    def __init__(self, connection_params: dict):
        self._connection_params = connection_params

    # format of the table_name is like: openhouse.u_openhouse.lotus_test
    def table(self, table_name: str) -> LazyTable:
        return LazyTable(table_name, self._connection_params)
