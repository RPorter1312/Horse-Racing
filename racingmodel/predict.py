from model.model import RacingPredictor

import logging
import argparse
import pandas as pd
from pathlib import Path

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    root_dir = Path(__file__).parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()

    model = RacingPredictor.load(args.model)

    data = pd.read_csv(Path(root_dir, 'data', 'raceform.csv'))

    proc_data = model.preprocess(data)

    predictions = model.predict(proc_data)