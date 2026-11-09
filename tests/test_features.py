import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from cobot.config import load_config
from cobot.data import SENSORS, TEMPERATURES
from cobot.features import make_feature_pipeline
from cobot.models import build_model


@pytest.fixture
def readings() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(80, len(SENSORS))), columns=SENSORS)
    df[TEMPERATURES] += 35
    df.insert(0, "session_id", [0] * 50 + [1] * 30)
    return df


@pytest.fixture
def pipeline():
    return make_feature_pipeline(**load_config().features.model_dump())


def test_features_never_use_the_future(readings, pipeline):
    """Perturbing every sample after t must leave the features at t unchanged."""
    t = 40
    perturbed = readings.copy()
    perturbed.loc[t + 1 :, SENSORS] *= 10
    before = pipeline.fit_transform(readings).loc[:t]
    after = pipeline.fit_transform(perturbed).loc[:t]
    pd.testing.assert_frame_equal(before, after)


def test_rolling_windows_restart_at_session_boundaries(readings, pipeline):
    features = pipeline.fit_transform(readings)
    first_of_session = features.loc[50]
    assert first_of_session["total_abs_current_std_15s"] == 0.0
    assert first_of_session["total_abs_current_mean_15s"] == pytest.approx(
        first_of_session["total_abs_current"]
    )


def test_output_is_numeric_without_missing_values_or_identifiers(readings, pipeline):
    features = pipeline.fit_transform(readings)
    assert features.notna().all().all()
    assert (features.dtypes == "float64").all()
    assert not {"session_id", *TEMPERATURES} & set(features.columns)


def test_features_and_a_fitted_model_serve_as_one_pipeline(readings, pipeline):
    """The logged model is features + classifier; it must predict straight from raw readings."""
    X = pipeline.fit_transform(readings)
    y = (X["total_abs_current"] > X["total_abs_current"].median()).astype(int)
    served = Pipeline(
        [("features", pipeline), ("model", build_model("logistic_regression").fit(X, y))]
    )
    assert served.predict_proba(readings).shape == (len(readings), 2)


def test_physics_features_follow_their_definitions(readings, pipeline):
    features = pipeline.fit_transform(readings)
    expected_power = readings["current_j1"].abs() * readings["speed_j1"].abs()
    np.testing.assert_allclose(features["power_j1"], expected_power)
    expected_total = readings[[f"current_j{i}" for i in range(6)]].abs().sum(axis=1)
    np.testing.assert_allclose(features["total_abs_current"], expected_total)
