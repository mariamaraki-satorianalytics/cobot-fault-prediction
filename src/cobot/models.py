"""Candidate classifiers, registered by name so new ones can be added without touching callers."""

from collections.abc import Callable
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ModelFactory = Callable[..., ClassifierMixin | Pipeline]

MODELS: dict[str, ModelFactory] = {
    # Predicts the class prior: the floor every real model must beat.
    "dummy": lambda **kw: DummyClassifier(strategy="prior", **kw),
    # Linear baseline; needs scaled inputs.
    "logistic_regression": lambda **kw: make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, **kw)
    ),
    "random_forest": lambda **kw: RandomForestClassifier(
        class_weight="balanced_subsample", n_jobs=-1, **kw
    ),
    "hist_gradient_boosting": lambda **kw: HistGradientBoostingClassifier(
        class_weight="balanced", **kw
    ),
}


def build_model(name: str, params: dict[str, Any] | None = None) -> ClassifierMixin | Pipeline:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**(params or {}))
