import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def get_decimal_odds(odds: str | None) -> float:
    if not pd.notnull(odds):
        return 2.0
    if "evens" in odds.lower() or "evs" in odds.lower():
        return 2.0
    odds = re.sub("[^0-9/]", "", odds)
    if odds == "":
        return 2.0
    num, dec = odds.split("/")
    try:
        num, dec = float(num), float(dec)
    except:
        return "error" + str(odds)
        # raise TypeError("Either numerator or denominator could not be converted to int")

    return (num / dec) + 1.0


def calculate_ev(
    win_odds: float,
    win_prob: float,
    is_ew: bool = False,
    place_prob: float | None = None,
    place_terms: int | None = None,
) -> float:
    ev_win = (win_prob * win_odds) - 1

    if is_ew:
        if place_prob is None or place_terms is None:
            raise Exception(
                "If 'is_ew' is True,both place_prob and place_terms must be supplied"
            )

        ev_win_part = ev_win
        ev_place_part = (place_prob * (win_odds * place_terms)) - 1
        ev_ew = (ev_win_part + ev_place_part) / 2

        return ev_ew

    return ev_win


def evaluate_bet_type(win_ev: float, ew_ev: float, tiebreak: str = "win") -> str:
    if tiebreak == "win":
        if win_ev >= ew_ev:
            return "win"
        else:
            return "ew"
    elif tiebreak == "ew":
        if ew_ev >= win_ev:
            return "ew"
        else:
            return "win"
    else:
        raise ValueError("'tiebreak' must be either 'win' or 'ew'")


def kelly_bet(
    balance: float,
    win_prob: float,
    win_odds: float,
    win_or_ew: str = "win",
    place_prob: float | None = None,
    place_terms: int | None = None,
    kelly_frac: float = 1.0,
    default_bet_amount: float = 10.0,
) -> float:
    if win_or_ew == "win":
        proportion = win_prob - ((1 - win_prob) / (win_odds - 1))
        return proportion * balance * kelly_frac
    elif win_or_ew == "ew":
        if place_prob is None or place_terms is None:
            raise ValueError(
                "If making an 'ew' bet,place_prob and place_terms must be provided"
            )

        win_ret = (win_odds - 1) + ((win_odds * place_terms) - 1)
        place_ret = (win_odds * place_terms) - 2
        lose_ret = -2

        def expected_log_utility(f):
            if f <= 0 or f >= 0.5:
                return -np.inf
            win_term = win_prob * np.log(1 + f * win_ret)
            place_term = (place_prob - win_prob) * np.log(1 + f * place_ret)
            lose_term = (1 - place_prob) * np.log(1 + f * lose_ret)
            return -(win_term + place_term + lose_term)

        opt = minimize_scalar(expected_log_utility, bounds=(0, 0.5), method="bounded")

        return (
            2 * round(opt.x, 2) * balance * kelly_frac
            if opt.success
            else default_bet_amount
        )
