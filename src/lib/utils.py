from typing_extensions import Optional
import pandas as pd
import sqlite3

from src.lib.constants import FIELD_NAMES, TIMESTAMP_KEY

def get_reach_fields(reach: int) -> list[str]:
    return [f for f in FIELD_NAMES if f"R{reach}" in f]

def sqlite2df(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    # Read the SQLite table into a pandas DataFrame
    query = f"SELECT * FROM {table}"
    df = pd.read_sql_query(query, conn)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    return df

def huc2df(files: list[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f, header=0)
        df[TIMESTAMP_KEY] = pd.to_datetime(df['Date_Text'], format='%Y%m%d')
        all_dfs.append(df)

    mean_precip = sum([df["mean_precip"] for df in all_dfs])

    df = all_dfs[0][[TIMESTAMP_KEY]].assign(mean_precip=mean_precip)

    # Filter to time range if needed
    if start:
        df = df[(df[TIMESTAMP_KEY] >= pd.to_datetime(start))]

    if end:
        df = df[(df[TIMESTAMP_KEY] <= pd.to_datetime(end))]

    df.set_index(TIMESTAMP_KEY, inplace=True)
    df.sort_index(inplace=True)

    return df
