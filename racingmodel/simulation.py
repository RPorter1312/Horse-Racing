from model import RacingPredictor
from utils.betting import get_decimal_odds, calculate_ev, evaluate_bet_type, kelly_bet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import joblib
import argparse
from pathlib import Path

logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class SimDateException(Exception):
    """Raise an error if the simulation includes training data"""


class ModelMismatchException(Exception):
    """Raise an error if the win and place models were
    trained on different data"""


class RacingSimulator:
    def __init__(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        starting_balance: float,
        margin: float = 0.1,
        max_horses: int | None = None,
        allow_ew_bets: bool = True,
        betting_strategy: str = "fixed",
        fixed_bet_amount: float = 10.0,
        kelly_frac: float = 1.0,
        max_bet_size: float = 100,
    ):
        if betting_strategy.lower() not in ["fixed", "kelly"]:
            raise ValueError("'betting_strategy' must be either 'fixed' or 'kelly'")
        if margin < 0:
            logging.warning(
                "Warning: 'margin' is set below 0, bets with a negative edge"
                "may be placed."
            )

        self.data = data.copy().sort_values(["date", "off", "race_id"])
        self.start_date = start_date
        self.end_date = end_date
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.ROI = None
        self.margin = margin
        self.max_horses = max_horses
        self.allow_ew_bets = allow_ew_bets
        self.betting_strategy = betting_strategy
        self.fixed_bet_amount = fixed_bet_amount
        self.kelly_frac = kelly_frac
        self.max_bet_size = max_bet_size
        self.betting_history = None

    def simulate(
        self,
        runners_col: str,
        win_res_col: str,
        place_res_col: str,
        win_odds_col: str,
        implied_prob_col: str,
        win_pred_prob_col: str,
        place_pred_prob_col: str | None = None,
        place_terms_col: str | None = None,
    ):
        bets = []
        for row in self.data.itertuples():
            runners = getattr(row, runners_col)
            win_res = getattr(row, win_res_col)
            place_res = getattr(row, place_res_col)
            win_odds = getattr(row, win_odds_col)
            implied_prob = getattr(row, implied_prob_col)
            win_prob = getattr(row, win_pred_prob_col)
            win_edge = win_prob - implied_prob

            if win_odds <= 1 or np.isnan(win_odds):
                continue
            if self.allow_ew_bets:
                if place_pred_prob_col is None or place_terms_col is None:
                    raise ValueError(
                        (
                            "If each-way bets are allowed, both"
                            "'place_pred_prob_col' and 'place_terms_col'"
                            "must be provided"
                        )
                    )
                place_prob = getattr(row, place_pred_prob_col)
                place_terms = getattr(row, place_terms_col)
                place_edge = place_prob - implied_prob

                if win_edge >= self.margin and place_edge >= self.margin:
                    win_ev = calculate_ev(
                        win_odds=win_odds, win_prob=win_prob, is_ew=False
                    )

                    ew_ev = calculate_ev(
                        win_odds=win_odds,
                        win_prob=win_prob,
                        is_ew=True,
                        place_prob=place_prob,
                        place_terms=place_terms,
                    )

                    bet_type = evaluate_bet_type(win_ev=win_ev, ew_ev=ew_ev)

                elif win_edge >= self.margin and place_edge < self.margin:
                    bet_type = "win"
                elif win_edge < self.margin and place_edge >= self.margin:
                    bet_type = "ew"
                else:
                    continue
                if self.betting_strategy == "fixed":
                    stake = self.fixed_bet_amount
                elif self.betting_strategy == "kelly":
                    if bet_type == "win":
                        stake = min(
                            kelly_bet(
                                balance=self.balance,
                                win_prob=win_prob,
                                win_odds=win_odds,
                                win_or_ew="win",
                                kelly_frac=self.kelly_frac,
                                default_bet_amount=self.fixed_bet_amount,
                            ),
                            self.max_bet_size,
                        )
                    elif bet_type == "ew":
                        stake = min(
                            kelly_bet(
                                balance=self.balance,
                                win_prob=win_prob,
                                win_odds=win_odds,
                                win_or_ew="ew",
                                place_prob=place_prob,
                                place_terms=place_terms,
                                kelly_frac=self.kelly_frac,
                                default_bet_amount=self.fixed_bet_amount,
                            ),
                            self.max_bet_size,
                        )
            else:
                if win_edge < self.margin:
                    continue
                bet_type = "win"
                if self.betting_strategy == "fixed":
                    stake = self.fixed_bet_amount
                elif self.betting_strategy == "kelly":
                    stake = min(
                        kelly_bet(
                            balance=self.balance,
                            win_prob=win_prob,
                            odds=win_odds,
                            win_or_ew="win",
                            frac=self.kelly_frac,
                            default_bet_amount=self.fixed_bet_amount,
                        ),
                        self.max_bet_size,
                    )
                stake = round(stake, 2)

            if bet_type == "win":
                payout = (stake * win_odds * win_res) - stake
            elif bet_type == "ew":
                payout = (
                    (stake * win_odds * win_res / 2)
                    + (stake * (win_odds * place_terms) * place_res / 2)
                    - stake
                )

            self.balance += payout

            logging.debug(
                f"Bet placed on {row.horse} on {row.date}. Bet type: {bet_type}, Stake: {stake}, Payout: {payout}."
            )

            bets.append(
                {
                    "date": row.date,
                    "race_id": row.race_id,
                    "race_name": row.race_name,
                    "runners": runners,
                    "horse": row.horse,
                    "bet_type": bet_type,
                    "stake": stake,
                    "payout": payout,
                    "balance": self.balance,
                    "win_prob": win_prob,
                    "place_prob": place_prob,
                    "odds": win_odds,
                    "implied_prob": implied_prob,
                    "win": win_res,
                    "place": place_res,
                }
            )

        self.ROI = self.balance / self.starting_balance

        self.betting_history = pd.DataFrame(bets)

    def evaluate_results(
        self,
    ):
        if self.betting_history is None:
            raise ValueError(
                "Simulation has not been performed yet, or no bets were placed"
            )

        sns.lineplot(data=self.betting_history, x="race_id", y="balance")

    def save(self, path: str | os.PathLike):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | os.PathLike):
        return joblib.load(path)


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("-sb", "--starting_balance", type=float, required=True)
    parser.add_argument("-ma", "--margin", type=float, required=True)
    parser.add_argument(
        "-rt", "--race_type", type=str, required=True, choices=["all", "flat", "jumps"]
    )
    parser.add_argument("-ew", "--allow_ew_bets", action="store_true", required=False)
    parser.add_argument(
        "-b", "--betting_strategy", type=str, choices=["fixed", "kelly"], required=True
    )
    parser.add_argument("-k", "--kelly_frac", type=float, required=False)
    parser.add_argument("-sd", "--sim_start_date", type=str, required=True)
    parser.add_argument("-ed", "--sim_end_date", type=str, required=True)
    parser.add_argument("-d", "--data", type=str, required=False)
    args = parser.parse_args()

    starting_balance = args.starting_balance
    margin = args.margin
    race_type = args.race_type
    allow_ew_bets = args.allow_ew_bets
    betting_strategy = args.betting_strategy
    sim_start_date = args.sim_start_date
    sim_end_date = args.sim_end_date

    if args.kelly_frac:
        kelly_frac = args.kelly_frac
    else:
        kelly_frac = 1

    if sim_start_date >= sim_end_date:
        raise ValueError("'sim_start_date' must be before 'sim_end_date'")

    win_model = RacingPredictor.load(
        Path(root_dir, "artifacts", "models", f"model_win_{race_type}.joblib")
    )

    logging.info("'Win' model loaded")

    if sim_start_date <= win_model.train_end_date:
        raise ValueError("'sim_start_date' must be after model training end date")

    data_all = pd.read_csv(Path(root_dir, "data", "raceform.csv"))
    data_all.sort_values(["horse", "date"], inplace=True)

    logging.info("Loaded all data")

    # Separate 'informational' data points into a new df,
    # as these will be dropped in preprocessing
    data_sim = data_all[
        ["date", "race_id", "race_name", "ran", "off", "horse", "sp"]
    ].copy()

    if args.data:
        data_preprocessed = joblib.load(args.data)
        logging.info("Preprocessed data has been loaded")
    else:
        # Need to preprocess on all data to use entire history of each horse
        data_preprocessed = win_model.preprocess(data_all)

        logging.info("Data has been preprocessed")

    # Filter both preprocessed data and sim data to relevant dates
    data_preprocessed = data_preprocessed[
        (data_preprocessed["date"] >= sim_start_date)
        & (data_preprocessed["date"] <= sim_end_date)
    ]

    data_sim = data_sim[
        (data_sim["date"] >= sim_start_date) & (data_sim["date"] <= sim_end_date)
    ]

    data_sim = pd.concat([data_sim, data_preprocessed[["win", "place"]]], axis=1)

    X_win, y_win = win_model.split_data(
        data=data_preprocessed, apply_train_test_split=False
    )

    win_preds = win_model.predict(X_win)

    logging.info("Predictions made for 'win' outcome")

    data_sim = pd.concat([data_sim, win_preds], axis=1)

    if allow_ew_bets:
        place_model = RacingPredictor.load(
            Path(root_dir, "artifacts", "models", f"model_place_{race_type}.joblib")
        )

        logging.info("'Place' model loaded")

        if (win_model.train_start_date != place_model.train_start_date) or (
            win_model.train_end_date != place_model.train_end_date
        ):
            raise ModelMismatchException(
                "The win and place models must be trained on the same data set"
            )

        X_place, y_place = place_model.split_data(
            data=data_preprocessed, apply_train_test_split=False
        )

        place_preds = place_model.predict(X_place)

        logging.info("Predictions made for 'place' outcome")

        data_sim = pd.concat([data_sim, place_preds], axis=1)

    data_sim["odds_dec"] = data_sim["sp"].apply(lambda x: get_decimal_odds(x))
    data_sim["implied_proba"] = 1 / data_sim["odds_dec"]
    data_sim["place_terms"] = 0.2

    logging.info(f"Now running simulation from {sim_start_date} to {sim_end_date}")

    sim = RacingSimulator(
        data=data_sim,
        start_date=sim_start_date,
        end_date=sim_end_date,
        starting_balance=args.starting_balance,
        margin=margin,
        allow_ew_bets=allow_ew_bets,
        betting_strategy=args.betting_strategy,
        kelly_frac=kelly_frac,
    )

    sim.simulate(
        runners_col="ran",
        win_res_col="win",
        place_res_col="place",
        win_odds_col="odds_dec",
        implied_prob_col="implied_proba",
        win_pred_prob_col="predicted_proba_win",
        place_pred_prob_col="predicted_proba_place",
        place_terms_col="place_terms",
    )

    if allow_ew_bets:
        sim.save(
            Path(
                root_dir,
                "artifacts",
                "simulations",
                f"sim_{sim_start_date}_{sim_end_date}_{margin}_{betting_strategy}_win and ew.joblib",
            )
        )
    else:
        sim.save(
            Path(
                root_dir,
                "artifacts",
                "simulations",
                f"sim_{sim_start_date}_{sim_end_date}_{margin}_{betting_strategy}_win only.joblib",
            )
        )

    logging.info(f"Simulation complete. Final balance is: {sim.balance}.")
