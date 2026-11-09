import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from cobot.evaluate import best_threshold, event_recall, false_alarm_episodes, out_of_fold_proba
from cobot.models import MODELS, build_model


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_registered_model_fits_and_predicts_probabilities(name):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)))
    y = pd.Series((X[0] > 0.8).astype(int))
    proba = build_model(name).fit(X, y).predict_proba(X)
    assert proba.shape == (120, 2)


def test_unknown_model_names_are_rejected():
    with pytest.raises(KeyError, match="choose from"):
        build_model("not_a_model")


def test_out_of_fold_predictions_never_cover_the_first_training_chunk():
    X = pd.DataFrame({"x": np.arange(60.0)})
    y = pd.Series([0, 1] * 30)
    proba = out_of_fold_proba(build_model("dummy"), X, y, TimeSeriesSplit(n_splits=3))
    assert proba.iloc[:15].isna().all()
    assert proba.iloc[15:].notna().all()


def test_event_recall_counts_events_with_at_least_one_alert():
    events = pd.Series([np.nan, 4, 4, np.nan, 9, 9])
    alerts = pd.Series([True, False, True, False, False, False])
    assert event_recall(alerts, events) == 0.5


def test_false_alarms_count_episodes_not_seconds():
    alerts = pd.Series([True, True, True, False, True, True, False, True])
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 0])
    # Episodes: rows 0-2 (false), 4-5 (true, overlaps a positive), 7 (false).
    assert false_alarm_episodes(alerts, y) == 2


def test_best_threshold_separates_perfectly_ranked_scores():
    y = pd.Series([0, 0, 0, 1, 1])
    proba = pd.Series([0.1, 0.2, 0.3, 0.8, 0.9])
    assert best_threshold(y, proba) == 0.8
