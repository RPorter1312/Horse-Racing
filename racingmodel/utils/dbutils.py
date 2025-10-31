import sqlalchemy
import pandas as pd
import logging
import os

logging.getLogger(__name__)

env_vars = {
    "user": os.environ.get("hruser"),
    "pass": os.environ.get("hrpass"),
    "db_name": os.environ.get("hrdatabase"),
}
engine = sqlalchemy.create_engine(
    f"postgresql+psycopg2://{env_vars['user']}:{env_vars['pass']}@localhost/{env_vars['db_name']}"
)


def upload_csv_to_pg(engine: sqlalchemy.Engine, csv_file_path: str | os.pathlike, cmd: str):
    with open(csv_file_path, "r") as f:
        cursor = engine.cursor
        cursor.copy_expert(cmd, f)
        engine.commit()


def upload_df_to_pg(engine: sqlalchemy.Engine, df: pd.DataFrame, table_name: str):
    df.to_sql(table_name, engine, if_exists='replace')