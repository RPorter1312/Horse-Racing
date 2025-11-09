import sqlalchemy
import pandas as pd
import logging
import os
import argparse
from pathlib import Path

logging.getLogger(__name__)

env_vars = {
    "user": os.environ.get("hruser"),
    "pass": os.environ.get("hrpass"),
    "db_name": os.environ.get("hrdatabase"),
}
engine = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{env_vars['user']}:{env_vars['pass']}@localhost/{env_vars['db_name']}"
)


def upload_csv_to_pg(engine: sqlalchemy.Engine, csv_file_path: str | os.PathLike, sql: str):
    with open(csv_file_path, "r") as f:
        next(f) # Skip over header
        conn = engine.raw_connection()
        cursor = conn.cursor()
        cursor.copy_expert(sql, f)
        conn.commit()


def upload_df_to_pg(engine: sqlalchemy.Engine, df: pd.DataFrame, table_name: str):
    df.to_sql(table_name, engine, if_exists='replace')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()

    root_dir = Path(__file__).parents[2]
    racing_hist_imp_sql_path = Path(
        root_dir,
        'racingmodel',
        'db',
        'sql', 
        'racing_history_csv_import.sql'
    )

    with open(racing_hist_imp_sql_path) as f:
        sql = f.read()
        upload_csv_to_pg(
            engine,
            csv_file_path=args.csv_file,
            sql=sql
        )