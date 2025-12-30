import pandas as pd
import json
import os

from .processing import dist_to_furlongs

def prep_racecard_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Drop individual columns we don't need
    data_dropped = data.drop(
        columns=[
            "dob",
            "sex",
            "colour",
            "breeder",
            "dam_region",
            "sire_region",
            "grandsire",
            "damsire",
            "damsire_region",
            "trainer_id",
            "trainer_location",
            "prev_trainers",
            "prev_owners",
            "comment",
            "spotlight",
            "quotes",
            "stable_tour",
            "headgear_first",
            "jockey_id",
            "last_run",
            "form",
            "trainer_rtf",
            "trainer_14_days.runs",
            "trainer_14_days.wins",
            "trainer_14_days.percent",
            "medical",
        ]
    )

    # Drop all stats columns in one line
    data_dropped = data_dropped.drop(data_dropped.filter(regex="stats").columns, axis=1)

    data_renamed = data_dropped.rename(
        columns={
            "sex_code": "sex",
            "distance": "dist",
            "headgear": "hg",
            "race_class": "class",
            "name": "horse",
            "off_time": "off",
            "number": "num",
        }
    )

    data_renamed["dist_f"] = data_renamed["dist"].apply(lambda x: dist_to_furlongs(x))

    return data_renamed

def parse_racecards(json_file: str | os.PathLike, regions: list[str]) -> pd.DataFrame:
    with open(json_file) as f:
        racecards = json.load(f)

    races = []

    for region in regions:
        for course in racecards[region].keys():
            for off_time in racecards[region][course].keys():
                race = pd.json_normalize(
                    data=racecards[region][course][off_time],
                    record_path="runners",
                    meta=[
                        "date",
                        "course",
                        "race_id",
                        "off_time",
                        "race_name",
                        "distance",
                        "going",
                        "pattern",
                        "type",
                        "race_class",
                        "age_band",
                        "rating_band",
                        "prize",
                    ],
                )
                races.append(race)

    racecard_data = pd.concat(races)

    return prep_racecard_data(racecard_data)
