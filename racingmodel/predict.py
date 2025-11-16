from model import RacingPredictor
from db.dbutils import (
    upload_df_to_pg,
    import_pg_to_df
)

import numpy as np
import pandas as pd
import sqlalchemy
import os
import logging
import argparse
from datetime import datetime
from pathlib import Path

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    root_dir = Path(__file__).parents[1]

    env_vars = {
        "user": os.environ.get("hruser"),
        "pass": os.environ.get("hrpass"),
        "db_name": os.environ.get("hrdatabase"),
    }

    engine = sqlalchemy.create_engine(
        f"postgresql+psycopg2://{env_vars['user']}:{env_vars['pass']}@localhost/{env_vars['db_name']}"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()

    model = RacingPredictor.load(args.model)

    df_hist = import_pg_to_df(engine=engine, table_name="racing_history")

    df_rc = import_pg_to_df(engine=engine, table_name="latest_racecards")

    # Need to combine upcoming race data with historical race data
    # for preprocessing
    missing_cols = df_hist.columns.difference(df_rc.columns)
    df_rc[missing_cols] = np.nan
    df_rc = df_rc[df_hist.columns]
    df_total = pd.concat({"hist": df_hist, "racecards": df_rc})

    df_processed = model.preprocess(df_total)

    # Separate upcoming races once preprocessing is done
    df_rc_processed = df_processed.loc["racecards"]
    predictions = model.predict(df_rc_processed)

    upload_df_to_pg(
        engine=engine,
        df=predictions,
        table_name='latest_predictions',
        if_exists='replace'
    )

    upload_df_to_pg(
        engine=engine,
        df=predictions,
        table_name='prediction_history',
        if_exists='append'
    )