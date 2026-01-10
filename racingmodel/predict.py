from .model import RacingPredictor
from .db.dbutils import upload_df_to_pg, import_pg_to_df, get_max_hist_date

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

if __name__ == "__main__":
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
    parser.add_argument("-wm", "--win_model", type=str, required=True)
    parser.add_argument("-pm", "--place_model", type=str, required=True)
    args = parser.parse_args()

    win_model = args.win_model
    place_model = args.place_model

    win_model = RacingPredictor.load(args.win_model)
    place_model = RacingPredictor.load(args.place_model)

    df_hist = import_pg_to_df(engine=engine, table_name="racing_history")

    # Use a smaller daatset for testing

    df_hist = df_hist[df_hist["date"] >= '2025-09-01']

    df_rc = import_pg_to_df(engine=engine, table_name="latest_racecards")

    # Check if 'racing_history' table is up to date
    current_date = datetime.today().strftime("%Y-%m-%d")
    max_hist_date = get_max_hist_date(engine=engine)

    if current_date != max_hist_date:
        # Log a warning that table is not up to date. In future,
        # have it so that the most recent data is pulled and uploaded
        # here, before making predictions.
        logging.warning(
            "Racing history table is not up to date, " \
            "predictions may be made using incomplete information"
        )

    # Need to combine upcoming race data with historical race data
    # for preprocessing
    missing_cols = df_hist.columns.difference(df_rc.columns)
    df_rc[missing_cols] = np.nan
    df_rc = df_rc[df_hist.columns]
    df_total = pd.concat({"hist": df_hist, "racecards": df_rc})

    logging.info("Preprocessing data:")

    df_processed = win_model.preprocess(df_total)
    df_processed = df_processed.drop(columns=["entry_id", "date", "win", "place"])
    logging.info("Data has been preprocessed")

    # Separate upcoming races once preprocessing is done
    df_rc_processed = df_processed.loc["racecards"]
    win_predictions = win_model.predict(df_rc_processed)
    place_predictions = place_model.predict(df_rc_processed)

    all_predictions = pd.concat([win_predictions, place_predictions], axis=1)

    for table, update_method in zip(
        ("latest_predictions", "prediction_history"), ("replace", "append")
    ):
        upload_df_to_pg(
            engine=engine,
            df=all_predictions,
            table_name=table,
            if_exists=update_method,
        )

    logging.info("Predictions uploaded to database")
