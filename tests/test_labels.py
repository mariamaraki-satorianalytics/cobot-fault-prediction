import pandas as pd

from cobot.labels import event_onsets


def test_onsets_mark_only_the_first_sample_of_each_run():
    flag = pd.Series([0, 1, 1, 0, 1, 0])
    session = pd.Series([0] * 6)
    assert event_onsets(flag, session).tolist() == [False, True, False, False, True, False]


def test_onsets_restart_at_session_boundaries():
    flag = pd.Series([1, 1, 1, 1])
    session = pd.Series([0, 0, 1, 1])
    assert event_onsets(flag, session).tolist() == [True, False, True, False]
