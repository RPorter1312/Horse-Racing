from model import RacingPredictor

import logging
import argparse
import joblib
import yaml
from yamlcore import CoreLoader
from pathlib import Path
import pandas as pd

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    root_dir = Path(__file__).parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--race_type', type=str, 
                        required=True, choices=['all', 'flat', 'jumps'],
                        help="The type of race being modelled")
    parser.add_argument('-t' ,'--target', type=str,
                        required=True, choices=['win', 'place'],
                        help="The target outcome being modelled")
    parser.add_argument('-trsd', '--train_start_date', type=str,
                        required=True, help="Start date of training data")
    parser.add_argument('-tred', '--train_end_date', type=str,
                        required=True, help="End date of training data")
    parser.add_argument('-tsed', '--test_end_date', type=str,
                        required=True, help="End date of test data")
    parser.add_argument('-s', '--save', required=False, action='store_true',
                        help='Optionally exports preprocessed data to excel file')
    args = parser.parse_args()

    race_type = args.race_type
    target = args.target
    train_start_date = args.train_start_date
    train_end_date = args.train_end_date
    test_end_date = args.test_end_date

    with open(Path(root_dir, 'racingmodel', 'config', 'features.yaml')) as f:
        feat_lists = yaml.load(f, Loader=CoreLoader)

    logging.info(f"Now training model for race_type: {race_type} and target: {target}")

    data = pd.read_csv(Path(root_dir, 'data', 'raceform.csv'))

    logging.info("Data has been read in")

    racing_model = RacingPredictor(
        target=target,
        race_type=race_type,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_end_date=test_end_date,
        excluded_features=feat_lists['non_training_features'],
        horse_windows=[3, 5, 7],
        jockey_windows=[25, 50, 100],
        trainer_windows=[50, 100, 150],
        date_windows=[30, 90, 150, 365],
        dist_bins=[1, 5, 7, 9, 11, 13, 15, 18, 21, 24, 27, 30, 100],
        include_lifetime=True
    )

    data = racing_model.preprocess(
        data=data,
    )

    logging.info(f"Data has been preprocessed. Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    if args.save:
        min_date = data['date'].min().strftime('%Y-%m-%d')
        max_date = data['date'].max().strftime('%Y-%m-%d')
        joblib.dump(
            value=data,
            filename=Path(
                root_dir,
                'data',
                f'preprocessed_data_{min_date}_{max_date}.joblib',
            )
        )

        logging.info("Preprocessed data has been saved as a python object")

    X_train, X_test, y_train, y_test = racing_model.split_data(
        data=data,
        apply_train_test_split=True
    )

    racing_model.fit(
        X=X_train,
        y=y_train,
        train=True,
        calibrate=True,
        tune=False
    )

    logging.info("Model has been fit")

    metrics = racing_model.performance(
        X=X_test,
        y_true=y_test,
        plot_calibration_curve=True
    )

    logging.info("Model performance:")
    for k, v in metrics.items():
        logging.info(f"{k}: {v}")

    racing_model.save(
        path=Path(
            root_dir,
            'artifacts',
            'models',
            f'model_{target}_{race_type}.joblib'
        )
    )

    logging.info("Model has been saved")