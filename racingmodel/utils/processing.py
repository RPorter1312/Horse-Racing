import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import re


def get_run_num(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["run_num"] = df.groupby("horse").cumcount() + 1

    return df


def get_is_first_race_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_first_race"] = 0
    first_idx = df.groupby("horse").head(1).index
    df.loc[first_idx, "is_first_race"] = 1

    return df


def time_to_seconds(time: str) -> float:
    if time == "-" or pd.isna(time):
        return None
    try:
        mins, secs = time.split(":")
        return int(round(int(mins) * 60 + float(secs)))
    except:
        raise Exception("'time' variable is not formatted correctly in some cases.")


def st_to_lbs(wgt_in_st: str) -> int:
    try:
        st, lbs = wgt_in_st.split("-")
        wgt_in_lbs = int(st) * 14 + int(lbs)
    except:
        raise Exception("'wgt' variable is not formatted correctly in some cases.")
    return wgt_in_lbs


def dist_to_furlongs(dist: str) -> int:
    miles = 0
    furlongs = 0

    frac_map = {"¼": 0.25, "½": 0.5, "¾": 0.75}

    def parse_fractional_part(match):
        number = match.group(1)
        fraction = match.group(2)
        return str(float(number) + frac_map[fraction])

    dist = re.sub(r"(\d)([¼½¾])", parse_fractional_part, dist)

    try:
        mile_match = re.search(r"(\d+(?:\.\d+)?)m", dist)
        if mile_match:
            miles = float(mile_match.group(1))

        furlong_match = re.search(r"(\d+(?:\.\d+)?)f", dist)
        if furlong_match:
            furlongs = float(furlong_match.group(1))

    except Exception as e:
        raise ValueError(f"Could not parse 'dist' string '{dist}': {e}")

    return miles * 8 + furlongs


def get_momentum_feats(
    df: pd.DataFrame,
    var: str,
    windows: list[int],
    known_pre_race: bool = False,
    group_cols: list[str] | None = None,
    compute_trend: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    if not type(windows) == list:
        raise Exception("'windows' must be a list of integers.")

    for window in windows:
        # Get z-score across grouping if specified, otherwise compare to entire dataset
        z_col = f"{var}_z"

        if group_cols is not None:
            df[z_col] = df.groupby(group_cols)[var].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0)
            )
        else:
            df[z_col] = (df[var] - df[var].mean()) / df[var].std(ddof=0)

        # Get the rolling z-score for each horse across specified window. If it's a post-race metric, use the previous value
        roll_col = f"{z_col}_rolling_{window}_{'curr' if known_pre_race else 'prev'}"
        grouped = df.groupby("horse", sort=False)[z_col]

        if known_pre_race:
            df[roll_col] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        else:
            df[roll_col] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Get momentum feature based on z-score
        momentum_col = f"{var}_momentum_{window}"
        if known_pre_race:
            df[momentum_col] = df[z_col] - df[roll_col]
        else:
            df[momentum_col] = df.groupby("horse")[z_col].shift(1) - df[roll_col]

        # Apply a linear regression to get the trend of recent performances
        if compute_trend:

            def get_slope(ser: pd.Series) -> float:
                lr = LinearRegression()
                if ser.isna().sum() > 0 or len(ser) < 2:
                    return np.nan
                X = np.arange(len(ser)).reshape(-1, 1)
                lr.fit(X, ser.values.reshape(-1, 1))
                return lr.coef_[0]

            trend_col = f"{var}_trend_{window}"
            if known_pre_race:
                df[trend_col] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=2).apply(get_slope)
                )
            else:
                df[trend_col] = grouped.transform(
                    lambda x: x.shift(1).rolling(window, min_periods=2).apply(get_slope)
                )

    return df


def get_speed_vs_avg(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()

    df["avg_race_speed"] = df.groupby("race_id")["speed"].mean()
    df["speed_vs_avg"] = df["speed"] - df["avg_race_speed"]

    for window in windows:
        df[f"speed_vs_avg_prev_{window}"] = df.groupby("horse")[
            "speed_vs_avg"
        ].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())

    return df


def get_field_strength(
    df: pd.DataFrame,
    rating_col: str,
    windows: list[int],
    include_lifetime: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    # Make sure the rating column is numeric
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

    # Exclude races where the entire field has no rating
    df["valid_race"] = df.groupby("race_id")[rating_col].transform(
        lambda x: x.notna().any()
    )
    df = df[df["valid_race"]].drop(columns=["valid_race"])

    # Calculate the average OR in a race, and for each horse, the difference between it's rating and the avg
    df[f"{rating_col}_field_strength"] = df.groupby("race_id")[rating_col].mean()
    df[f"{rating_col}_diff_vs_field"] = (
        df[rating_col] - df[f"{rating_col}_field_strength"]
    )
    df[f"{rating_col}_rank_in_race"] = df.groupby("race_id")[rating_col].rank(
        ascending=False, method="min"
    )

    for window in windows:
        # Calculate avg field strength of a horses last n runs
        fs_col = f"field_strength_prev_{window}"

        df[fs_col] = df.groupby("horse")[f"{rating_col}_field_strength"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        # Calculate avg difference between horses rating and field avg over last n runs

        fs_diff_col = f"{rating_col}_diff_vs_field_prev_{window}"

        df[fs_diff_col] = df.groupby("horse")[f"{rating_col}_diff_vs_field"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        # Calculate the avg rank of a horse by rating over last n runs
        fs_rank_col = f"{rating_col}_rank_prev_{window}"

        df[fs_rank_col] = df.groupby("horse")[f"{rating_col}_rank_in_race"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        # Set window calcs to 0 for each horse's first run
        first_idx = df.groupby("horse").head(1).index
        df.loc[first_idx, [fs_col, fs_diff_col, fs_rank_col]] = 0

    # Perform the same calculations as above over all of a horse's runs
    if include_lifetime:
        fs_col_lt = f"{rating_col}_field_strength_lifetime"
        df[fs_col_lt] = df.groupby("horse")[f"{rating_col}_field_strength"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        fs_diff_col_lt = f"{rating_col}_diff_vs_field_lifetime"
        df[fs_diff_col_lt] = df.groupby("horse")[
            f"{rating_col}_diff_vs_field"
        ].transform(lambda x: x.shift(1).expanding().mean())

        fs_rank_col_lt = f"{rating_col}_rank_lifetime"
        df[fs_rank_col_lt] = df.groupby("horse")[
            f"{rating_col}_rank_in_race"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Set window calcs to 0 for each horse's first run
        first_idx = df.groupby("horse").head(1).index
        df.loc[first_idx, [fs_col_lt, fs_diff_col_lt, fs_rank_col_lt]] = 0

    return df


def get_days_since_last_race(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure correct ordering
    df.sort_values(["horse", "date"], inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    df["days_since_last_race"] = df.groupby("horse")["date"].transform(
        lambda x: x.diff().dt.days
    )

    # If its the first race, set to -1
    first_idx = df.groupby("horse").head(1).index
    df.loc[first_idx, ["days_since_last_race"]] = -1

    return df


def get_races_in_last_n_days(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["horse", "date"]).copy()

    for window in windows:
        col_name = f"runs_last_{window}_days"
        df[col_name] = 0

        # Process per horse, but efficiently
        for horse, group in df.groupby("horse"):
            dates = group["date"].values.astype("datetime64[D]")
            idxs = group.index
            counts = np.zeros(len(dates), dtype=int)

            for i in range(len(dates)):
                cutoff = dates[i] - np.timedelta64(window, "D")
                start = np.searchsorted(dates, cutoff, side="left")
                counts[i] = (
                    i - start
                )  # how many races happened before the current date within the window

            df.loc[idxs, col_name] = counts

    return df


def group_by_threshold(df: pd.DataFrame, var: str, threshold: float) -> pd.Series:
    df = df.copy()
    proportions = df[var].value_counts(normalize=True, dropna=False)
    to_keep = proportions[proportions >= threshold].index

    return df[var].where(df[var].isin(to_keep), other="Other")


# Function to rename categories in a series which are below the top n categories by frequency.
def group_top_n(df: pd.DataFrame, var: str, num: int) -> pd.Series:
    df = df.copy()
    to_keep = df[var].value_counts().nlargest(num).index

    return df[var].where(df[var].isin(to_keep), other="Other")


def get_runners(df: pd.DataFrame, id_var: str, new_col_name: str) -> pd.DataFrame:
    df = df.copy()
    ser_runners = df.groupby(id_var)["horse"].count()
    ser_runners.name = new_col_name
    df = df.merge(ser_runners, how="inner", on=id_var)
    return df


def is_placed(pos: str | int, runners: int) -> int:
    try:
        pos = int(pos)
    except:
        return 0

    if runners < 5:
        return 1 if pos == 1 else 0
    elif runners <= 10:
        return 1 if pos <= 2 else 0
    elif runners <= 13:
        return 1 if pos <= 3 else 0
    elif runners <= 16:
        return 1 if pos <= 4 else 0
    else:
        return 1 if pos <= 5 else 0


def get_win_and_place_rates(
    df: pd.DataFrame,
    category: str,
    win_col: str,
    place_col: str,
    windows: list[int],
    include_lifetime: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    for window in windows:
        new_win_col = f"{category}_win_rate_{window}"
        new_place_col = f"{category}_place_rate_{window}"

        df[new_win_col] = df.groupby(category, sort=False)[win_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        df[new_place_col] = df.groupby(category, sort=False)[place_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        first_idx = df.groupby(category).head(1).index
        df.loc[first_idx, [new_win_col, new_place_col]] = 0

    if include_lifetime:
        new_win_col = f"{category}_win_rate_lifetime"
        new_place_col = f"{category}_place_rate_lifetime"

        df[new_win_col] = df.groupby(category, sort=False)[win_col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        df[new_place_col] = df.groupby(category, sort=False)[place_col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        first_idx = df.groupby(category).head(1).index
        df.loc[first_idx, [new_win_col, new_place_col]] = 0

    return df


def get_win_or_place_rate_diff(
    df: pd.DataFrame,
    category_var: str,
    result_vars: list[str],
    comparison_vars: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for res_var in result_vars:
        for comp_var in comparison_vars:
            prefix = f"{comp_var}_{res_var}"

            df[f"cumul_{prefix}"] = df.groupby(
                [category_var, comp_var], observed=False
            )[res_var].transform(lambda x: x.shift(1).cumsum())

            df[f"prev_runs_{prefix}"] = df.groupby(
                [category_var, comp_var], observed=False
            )[res_var].transform(lambda x: x.shift(1).expanding().count())

            df[f"cumul_overall_{res_var}"] = df.groupby(category_var)[
                res_var
            ].transform(lambda x: x.shift(1).cumsum())

            df[f"prev_runs_overall_{res_var}"] = df.groupby(category_var)[
                res_var
            ].transform(lambda x: x.shift(1).expanding().count())

            # Calculate win/place rate over the specified category, if first race, replace with 0
            df[f"{prefix}_rate"] = (
                df[f"cumul_{prefix}"] / df[f"prev_runs_{prefix}"]
            ).where(df[f"prev_runs_{prefix}"] > 0, 0)

            # Calculate overall win/place rate, if first race, replace with 0
            df[f"overall_{res_var}_rate"] = (
                df[f"cumul_overall_{res_var}"] / df[f"prev_runs_overall_{res_var}"]
            ).where(df[f"prev_runs_overall_{res_var}"] > 0, 0)

            df[f"{prefix}_rate_diff_to_overall"] = (
                df[f"{prefix}_rate"] - df[f"overall_{res_var}_rate"]
            )

    return df


def get_non_finishes(
    df: pd.DataFrame, var: str, non_fins: list[str], windows: list[int]
) -> pd.DataFrame:
    df = df.copy()

    for window in windows:
        for reason in non_fins:
            non_fin_col = f"{reason}_count_{window}"
            df[non_fin_col] = (df[var] == reason).astype(int)
            df[non_fin_col] = df.groupby("horse")[non_fin_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
            )

            first_idx = df.groupby("horse").head(1).index
            df.loc[first_idx, non_fin_col] = 0

    return df
