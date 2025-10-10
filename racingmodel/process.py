from racingmodel.model import RacingPredictor
import pandas as pd
import logging
import yaml
import yamlcore
import argparse
import joblib
from pathlib import Path

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    root_dir = Path(__file__).parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--start_date', type=str, required=True)
    parser.add_argument('-ed', '--end_date', type=str, required=True)
    parser.add_argument('-mp', '--model_path', type=str, required=True)
    args = parser.parse_args()

    data_all = pd.read_csv(Path(root_dir, 'data', 'raceform.csv'))

    model = RacingPredictor.load(args.model_path)

    data_preprocessed = model.preprocess(data_all)

    joblib.dump(
        data_preprocessed,
        filename=(
            Path(
                root_dir,
                'data',
                f'data_preprocessed_{args.start_date}_{args.end_date}'
            )
        )
    )


