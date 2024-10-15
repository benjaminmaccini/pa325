import pandas as pd
import sqlite3

def sqlite2df(conn: sqlite3.Connection, table: str, nan_replace: bool = True, nan_val: float = -9999.) -> pd.DataFrame:
    # Read the SQLite table into a pandas DataFrame
    query = f"SELECT * FROM {table}"
    df = pd.read_sql_query(query, conn)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Replace all values matching nan_val with pandas nan
    if nan_replace:
        df = df.replace(nan_val, pd.NA)

    return df
