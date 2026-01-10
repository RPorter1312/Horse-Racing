import sqlalchemy
import pandas as pd
import os
from typing import Literal


def upload_csv_to_pg(engine: sqlalchemy.Engine, csv_file_path: str | os.PathLike, sql: str):
    with open(csv_file_path, "r") as f:
        next(f) # Skip over header
        conn = engine.raw_connection()
        cursor = conn.cursor()
        cursor.copy_expert(sql, f)
        conn.commit()


def upload_df_to_pg(
        engine: sqlalchemy.Engine,
        df: pd.DataFrame,
        table_name: str,
        if_exists: Literal["fail", "replace", "append"]='fail'
    ):
    with engine.connect() as conn:
        df.to_sql(name=table_name, con=conn, if_exists=if_exists)


def import_pg_to_df(engine: sqlalchemy.engine, table_name: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_table(table_name=table_name, con=conn)
    

def get_max_hist_date(engine: sqlalchemy.engine) -> str:
    conn = engine.raw_connection()
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM racing_history")
    
    return cur.fetchone()[0].strftime('%Y-%m-%d')