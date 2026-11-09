import numpy as np
import pandas as pd
import pytest

from cobot.config import load_config
from cobot.data import (
    SENSORS,
    SPEEDS,
    TARGETS,
    add_session_ids,
    clean,
    read_processed,
    sha256,
    write_processed,
)

RAW_SENSORS = [
    *(f"{kind}_J{i}" for i in range(6) for kind in ("Current", "Temperature")),
    *(f"Speed_J{i}" for i in range(6)),
    "Tool_current",
]


@pytest.fixture
def raw() -> pd.DataFrame:
    """Five rows that reproduce every quirk found in the real spreadsheet."""
    n = 5
    df = pd.DataFrame(np.ones((n, len(RAW_SENSORS))), columns=RAW_SENSORS)
    df.insert(0, "Num", range(1, n + 1))
    df.insert(
        1,
        "Timestamp",
        [
            "2022-10-26T08:00:00.000Z",
            '"2022-10-26T08:00:01.000Z"',
            "2022-10-26T08:00:02.000Z",
            "2022-10-26T08:00:03.000Z",
            "2022-10-26T09:00:00.000Z",
        ],
    )
    df["cycle "] = 1
    df["Robot_ProtectiveStop"] = [0.0, 1.0, np.nan, np.nan, 0.0]
    df["grip_lost"] = [False, False, False, True, False]
    df.loc[2, RAW_SENSORS] = np.nan  # fully empty row
    df.loc[3, RAW_SENSORS[1:]] = np.nan  # only Current_J0 survives
    df.loc[0, "Speed_J5"] = 1e-44
    return df.rename(columns={"Temperature_J0": "Temperature_T0"})


def test_columns_are_standardised(raw):
    df = clean(raw, session_gap_s=60)
    assert list(df.columns) == ["timestamp", "session_id", "cycle", *SENSORS, *TARGETS]


def test_timestamps_parse_including_quoted(raw):
    df = clean(raw, session_gap_s=60)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].notna().all()


def test_empty_rows_dropped_and_partial_rows_forward_filled(raw):
    df = clean(raw, session_gap_s=60)
    assert len(df) == 4
    assert df.notna().all().all()
    partial = df[df["timestamp"] == pd.Timestamp("2022-10-26T08:00:03Z")].iloc[0]
    assert partial["protective_stop"] == 1  # carried from the previous sample


def test_targets_are_integers_and_denormals_zeroed(raw):
    df = clean(raw, session_gap_s=60)
    assert (df[TARGETS].dtypes == "int64").all()
    assert df.loc[0, "speed_j5"] == 0.0
    assert (df[SPEEDS].abs().lt(1e-12) <= df[SPEEDS].eq(0)).all().all()


def test_sessions_split_on_long_gaps():
    ts = pd.Series(
        pd.to_datetime(["08:00:00", "08:00:01", "08:00:40", "09:00:00"], format="%H:%M:%S")
    )
    assert add_session_ids(ts, gap_s=60).tolist() == [0, 0, 0, 1]


def test_processed_csv_round_trips(raw, tmp_path):
    df = clean(raw, session_gap_s=60)
    path = tmp_path / "clean.csv"
    write_processed(df, path)
    pd.testing.assert_frame_equal(read_processed(path), df, check_dtype=False)
    assert isinstance(read_processed(path)["timestamp"].dtype, pd.DatetimeTZDtype)


def test_committed_raw_file_matches_checksum():
    cfg = load_config().data
    assert sha256(cfg.raw_path) == cfg.sha256
