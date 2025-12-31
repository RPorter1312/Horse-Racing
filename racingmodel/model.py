from .utils import processing as proc

import os
import joblib
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgbm
from functools import partial
from sklearn.model_selection import (
    cross_val_score,
    KFold,
)
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import make_scorer, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay


class RacingPredictor:
    def __init__(
        self,
        target: str,
        race_type: str,
        train_start_date: str,
        train_end_date: str,
        test_end_date: str,
        excluded_features: list,
        horse_windows: list[int],
        jockey_windows: list[int],
        trainer_windows: list[int],
        date_windows: list[int],
        dist_bins: list[int],
        include_lifetime: bool = True,
    ):
        if target != "win" and target != "place":
            raise ValueError("target must be either 'win' or 'place'")
        if race_type != "flat" and race_type != "jumps" and race_type != "all":
            raise ValueError("race_type must be either 'flat', 'jumps' or 'all'")

        self.target = target
        self.race_type = race_type
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_end_date = test_end_date
        self.excluded_features = excluded_features
        self.horse_windows = horse_windows
        self.jockey_windows = jockey_windows
        self.trainer_windows = trainer_windows
        self.date_windows = date_windows
        self.dist_bins = dist_bins
        self.include_lifetime = include_lifetime
        self.model = None

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        data = self._apply_base_transforms(df=data, dist_bins=self.dist_bins)

        # Fixing some incorrectly classified race types
        data.loc[(data["dist_f"] > 22) & (data["type"] == "Flat"), "type"] = (
            "Chase"
        )

        if self.race_type == "all":
            pass
        elif self.race_type == "flat":
            data = data[data["type"] == "Flat"]
        elif self.race_type == "jumps":
            data = data[data["type"].isin(["NH Flat", "Hurdle", "Chase"])]

        data.sort_values(["horse", "date"], inplace=True)

        data = proc.get_run_num(data)
        data = proc.get_is_first_race_flag(data)

        data["ovr_btn"] = pd.to_numeric(data["ovr_btn"].replace("-", np.nan))
        data["btn"] = pd.to_numeric(data["ovr_btn"].replace("-", np.nan))

        data = proc.get_momentum_feats(
            df=data,
            var="speed",
            windows=self.horse_windows,
            known_pre_race=False,
            group_cols=["dist", "going"],
            compute_trend=True,
        )

        data = proc.get_speed_vs_avg(df=data, windows=self.horse_windows)

        data = proc.get_field_strength(
            df=data,
            rating_col="ofr",
            windows=self.horse_windows,
            include_lifetime=self.include_lifetime,
        )

        data = proc.get_days_since_last_race(data)
        data = proc.get_races_in_last_n_days(df=data, windows=self.date_windows)

        data["hg_grp"] = proc.group_by_threshold(df=data, var="hg", threshold=0.1)

        for var in ["trainer", "jockey"]:
            data[f"{var}_grp"] = proc.group_top_n(df=data, var=var, num=50)

        data["win"] = data["pos"].apply(lambda x: 1 if x == "1" else 0)
        data["place"] = data.apply(lambda x: proc.is_placed(x["pos"], x["ran"]), axis=1)

        for category, windows in zip(
            ["horse", "jockey", "trainer"],
            [self.horse_windows, self.jockey_windows, self.trainer_windows],
        ):
            data = proc.get_win_and_place_rates(
                df=data,
                category=category,
                win_col="win",
                place_col="place",
                windows=windows,
                include_lifetime=self.include_lifetime,
            )
            data = proc.get_win_or_place_rate_diff(
                df=data,
                category_var=category,
                result_vars=["win", "place"],
                comparison_vars=["going", "dist_bins"],
            )

        data = proc.get_non_finishes(
            df=data, var="pos", non_fins=["F", "PU", "UR"], windows=self.horse_windows
        )

        data.drop(columns=self.excluded_features, inplace=True, errors="ignore")

        # Need to drop calculated columns that were used for historical trends
        data.drop(
            columns=[
                "speed",
                "time_seconds",
                "speed_z",
                "avg_race_speed",
                "speed_vs_avg",
            ],
            inplace=True,
        )

        data = data.astype(
            {col: "category" for col in data.select_dtypes(include="O").columns}
        )
        return data

    def split_data(
        self, data: pd.DataFrame, apply_train_test_split: bool = True
    ) -> tuple[pd.DataFrame | pd.Series]:
        data = data.copy()

        if self.target == "win":
            data = data.drop(columns=["place"])
        elif self.target == "place":
            data = data.drop(columns=["win"])

        if apply_train_test_split:
            train_data = data[
                (data["date"] >= self.train_start_date)
                & (data["date"] <= self.train_end_date)
            ]

            test_data = data[
                (data["date"] > self.train_end_date)
                & (data["date"] <= self.test_end_date)
            ]

            X_train = train_data.drop(columns=["date", self.target])
            y_train = train_data[self.target]
            X_test = test_data.drop(columns=["date", self.target])
            y_test = test_data[self.target]

            return X_train, X_test, y_train, y_test
        else:
            return data.drop(columns=["date", self.target]), data[self.target]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train: bool = True,
        calibrate: bool = True,
        tune: bool = True,
        n_trials: int = 10,
        n_jobs: int = 1,
    ):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "is_unbalance": "true",
            "verbosity": -1,
        }

        if train:
            if tune:
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    partial(self._objective, X=X, y=y), n_trials=n_trials, n_jobs=n_jobs
                )
                best_params = study.best_params

                params = {**params, **best_params}

            self.model = lgbm.LGBMClassifier(**params)

        self.model.fit(X, y)

        if calibrate:
            self.model = CalibratedClassifierCV(
                FrozenEstimator(self.model), cv=5, method="isotonic"
            )

            self.model.fit(X, y)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "model") or self.model is None:
            raise AttributeError("Model has not been trained yet")

        y_pred = self.model.predict_proba(data)[:, 1]

        return pd.Series(
            y_pred, index=data.index, name=f"predicted_proba_{self.target}"
        )

    def performance(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        plot_calibration_curve: bool = True,
        n_bins: int = 10,
    ) -> dict:
        metrics = {}
        y_pred = self.model.predict_proba(X)[:, 1]

        metrics["log loss"] = log_loss(y_true, y_pred)
        metrics["brier score"] = brier_score_loss(y_true, y_pred)

        if plot_calibration_curve:
            CalibrationDisplay.from_predictions(
                y_true=y_true, y_prob=y_pred, n_bins=n_bins
            )

        return metrics

    def save(self, path: str | os.PathLike):
        joblib.dump(self, filename=path)

    @staticmethod
    def load(path: str | os.PathLike):
        return joblib.load(path)

    def _apply_base_transforms(
        self, df: pd.DataFrame, dist_bins: list[int]
    ) -> pd.DataFrame:
        df = df.copy()
        df["time_seconds"] = df["time"].map(proc.time_to_seconds)
        df["dist_f"] = df["dist_f"].str.replace("f", "").astype(float)
        df["dist_bins"] = pd.cut(
            df["dist_f"], bins=dist_bins, include_lowest=True
        )
        # Need to make new bin categories JSON-safe for LGBM
        # (will raise 'Circular reference' ValueError in fit method else)
        df["dist_bins"] = df["dist_bins"].astype(str).astype("category")

        df["flat_or_jumps"] = df["type"].apply(
            lambda x: "Flat" if x == "Flat" else "Jumps"
        )
        df["speed"] = df["dist_f"] / df["time_seconds"]

        return df

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series):
        scorer = make_scorer(log_loss, greater_is_better=False)

        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        }

        scores = cross_val_score(
            estimator=lgbm.LGBMClassifier(**trial_params),
            X=X,
            y=y,
            scoring=scorer,
            cv=KFold(5, shuffle=True),
        )

        return np.mean(scores) * scorer._sign
