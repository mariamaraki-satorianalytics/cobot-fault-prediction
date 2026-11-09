"""Physics-informed and temporal features, as stateless scikit-learn transformers.

Both transformers only ever look at the current and past samples of the same session, so the
features at time t cannot depend on anything that happens after t.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from cobot.data import CURRENTS, SPEEDS, TEMPERATURES

# Joints count as moving above this speed (rad/s); smaller values are sensor noise.
_MOVING_RAD_S = 1e-3


class _Stateless(BaseEstimator, TransformerMixin):
    """Transformers whose output depends only on the rows they are given, not on a fit."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True


class PhysicsFeatures(_Stateless):
    """Instantaneous features grounded in motor physics.

    Joint torque is roughly proportional to motor current (tau = k_t * I), so |I| is a load proxy
    and |I| * |omega| a mechanical-power proxy. Temperatures are collapsed to their mean because
    the six joint temperatures move together (r >= 0.99).
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        abs_current = X[CURRENTS].abs().to_numpy()
        abs_speed = X[SPEEDS].abs().to_numpy()
        features = {
            **{f"abs_current_j{i}": abs_current[:, i] for i in range(6)},
            **{f"power_j{i}": abs_current[:, i] * abs_speed[:, i] for i in range(6)},
            "total_abs_current": abs_current.sum(axis=1),
            "mean_abs_speed": abs_speed.mean(axis=1),
            "joints_moving": (abs_speed > _MOVING_RAD_S).sum(axis=1),
            "mean_temperature": X[TEMPERATURES].mean(axis=1).to_numpy(),
        }
        return X.assign(**features)


class RollingFeatures(_Stateless):
    """Rolling statistics over the recent past, computed per recording session."""

    def __init__(self, signals: Sequence[str], windows: Sequence[int], temperature_window: int):
        self.signals = signals
        self.windows = windows
        self.temperature_window = temperature_window

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        by_session = X.groupby("session_id", sort=False)
        features = {}
        for w in self.windows:
            rolling = by_session[list(self.signals)].rolling(w, min_periods=1)
            for stat in ("mean", "std", "max"):
                values = getattr(rolling, stat)().reset_index(level=0, drop=True)
                features |= {f"{s}_{stat}_{w}s": values[s] for s in self.signals}
        features |= {f"{s}_diff": by_session[s].diff() for s in self.signals}
        # Rate of heating (degC/s) rather than the raw level, which is confounded with time of day.
        w = self.temperature_window
        features["temperature_rate"] = by_session["mean_temperature"].diff(w) / w
        return pd.concat([X, pd.DataFrame(features).fillna(0.0)], axis=1)


class SelectModelInputs(_Stateless):
    """Drop identifiers, targets and absolute temperatures before the model sees the data."""

    excluded = ["timestamp", "session_id", "cycle", "protective_stop", "grip_lost"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dropped = [c for c in [*self.excluded, *TEMPERATURES, "mean_temperature"] if c in X]
        return X.drop(columns=dropped).astype(np.float64)


def make_feature_pipeline(
    signals: Sequence[str], windows: Sequence[int], temperature_window: int
) -> Pipeline:
    return Pipeline(
        [
            ("physics", PhysicsFeatures()),
            ("rolling", RollingFeatures(signals, windows, temperature_window)),
            ("select", SelectModelInputs()),
        ]
    )
