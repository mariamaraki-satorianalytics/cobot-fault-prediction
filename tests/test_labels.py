import pandas as pd

from cobot.labels import event_onsets, make_onset_target


def test_onsets_mark_only_the_first_sample_of_each_run():
    flag = pd.Series([0, 1, 1, 0, 1, 0])
    session = pd.Series([0] * 6)
    assert event_onsets(flag, session).tolist() == [False, True, False, False, True, False]


def test_onsets_restart_at_session_boundaries():
    flag = pd.Series([1, 1, 1, 1])
    session = pd.Series([0, 0, 1, 1])
    assert event_onsets(flag, session).tolist() == [True, False, True, False]


def frame(flags: list[int], sessions: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"stop": flags, "session_id": sessions or [0] * len(flags)})


def test_onset_target_looks_ahead_by_the_horizon():
    labels = make_onset_target(frame([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]), "stop", horizon=2)
    # The onset is at position 4, so positions 2 and 3 are "stop ahead".
    assert labels["y"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert labels.loc[labels["y"].eq(1), "event"].tolist() == [4.0, 4.0]


def test_rows_in_a_fault_and_at_session_end_are_not_usable():
    labels = make_onset_target(frame([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]), "stop", horizon=2)
    in_fault, at_session_end = [4, 5], [8, 9]
    assert labels.index[~labels["usable"]].tolist() == in_fault + at_session_end


def test_onset_target_does_not_look_across_sessions():
    labels = make_onset_target(frame([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1]), "stop", horizon=2)
    assert labels["y"].tolist() == [0, 0, 0, 0, 0, 0]
