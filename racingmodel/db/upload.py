from .dbutils import upload_csv_to_pg, upload_df_to_pg
from ..utils.parsing import parse_racecards

import sqlalchemy
import os
import argparse
import logging
import json
from pathlib import Path

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]

    env_vars = {
        "user": os.environ.get("hruser"),
        "pass": os.environ.get("hrpass"),
        "db_name": os.environ.get("hrdatabase"),
    }

    engine = sqlalchemy.create_engine(
        f"postgresql+psycopg2://{env_vars['user']}:{env_vars['pass']}@localhost/{env_vars['db_name']}"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--type", type=str, choices=["history", "racecards"], required=True
    )
    parser.add_argument("-f", "--file", type=str, required=True)
    args = parser.parse_args()

    if args.type == "history":
        racing_hist_imp_sql_path = Path(
            root_dir, "racingmodel", "db", "sql", "racing_history_csv_import.sql"
        )
        logging.info("Uploading race history data to database...")

        with open(racing_hist_imp_sql_path) as f:
            sql = f.read()
            upload_csv_to_pg(engine=engine, file=args.file, sql=sql)

        logging.info("History updated successfully")
    elif args.type == "racecards":
        logging.info("Uploading latest racecards data to database...")

        racecards_parsed = parse_racecards(json_file=args.file, regions=["GB", "IRE"])
        upload_df_to_pg(engine=engine, df=racecards_parsed, table_name="latest_racecards", if_exists="replace")

        logging.info("Racecards uploaded successfully")
