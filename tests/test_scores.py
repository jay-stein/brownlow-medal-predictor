import pandas as pd

from brownlow import scores
from brownlow.cli import parse_seasons


def test_labelled_rows_filters_unresolved():
    frame = pd.DataFrame({"LABEL_STATUS": ["voted", "zero", "unresolved"]})
    assert list(scores.labelled_rows(frame)["LABEL_STATUS"]) == ["voted", "zero"]


def test_parse_seasons_range_and_list():
    assert parse_seasons("2020-2022") == [2020, 2021, 2022]
    assert parse_seasons("2021,2023") == [2021, 2023]
