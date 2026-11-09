"""Metrics and figures for imbalanced, time-ordered fault forecasting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

from cobot.plotting import INK_SECONDARY, MUTED, SERIES, STOP

SECONDS_PER_HOUR = 3600


def out_of_fold_proba(estimator, X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> pd.Series:
    """Positive-class probabilities from forward-chaining CV; NaN for rows never validated."""
    proba = pd.Series(np.nan, index=X.index)
    for train_idx, val_idx in cv.split(X):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba.iloc[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return proba


def best_threshold(y: pd.Series, proba: pd.Series) -> float:
    """The probability cut-off that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def event_recall(alerts: pd.Series, events: pd.Series) -> float:
    """Share of fault onsets preceded by at least one alert inside the forecast horizon."""
    warned = alerts[events.notna()].groupby(events.dropna()).any()
    return float(warned.mean()) if len(warned) else float("nan")


def false_alarm_episodes(alerts: pd.Series, y: pd.Series) -> int:
    """Runs of consecutive alerts that never overlap a "stop ahead" sample."""
    episode = alerts.ne(alerts.shift()).cumsum()[alerts]
    return int(y[alerts].groupby(episode).max().eq(0).sum())


def classification_metrics(
    y: pd.Series, proba: pd.Series, threshold: float, events: pd.Series
) -> dict[str, float]:
    alerts = proba.ge(threshold)
    hours = len(y) / SECONDS_PER_HOUR
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "precision": precision_score(y, alerts, zero_division=0),
        "recall": recall_score(y, alerts),
        "f1": f1_score(y, alerts),
        "event_recall": event_recall(alerts, events),
        "false_alarms_per_hour": false_alarm_episodes(alerts, y) / hours,
        "threshold": threshold,
    }


def _model_colours(names: list[str]) -> dict[str, str]:
    palette = iter(SERIES)
    return {n: MUTED if n == "dummy" else next(palette) for n in names}


def plot_pr_curves(y: pd.Series, probas: dict[str, pd.Series]) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, colour in _model_colours(list(probas)).items():
        ap = average_precision_score(y, probas[name])
        if probas[name].nunique() == 1:  # a constant score has no curve, only the class prior
            ax.axhline(y.mean(), color=colour, label=f"{name}, no skill (PR-AUC {ap:.2f})")
            continue
        precision, recall, _ = precision_recall_curve(y, probas[name])
        ax.plot(recall, precision, color=colour, label=f"{name} (PR-AUC {ap:.2f})")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Precision-recall on the held-out test period")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_roc_curves(y: pd.Series, probas: dict[str, pd.Series]) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, colour in _model_colours(list(probas)).items():
        fpr, tpr, _ = roc_curve(y, probas[name])
        auc = roc_auc_score(y, probas[name])
        ax.plot(fpr, tpr, color=colour, label=f"{name} (ROC-AUC {auc:.2f})")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("ROC on the held-out test period")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def plot_confusion(y: pd.Series, alerts: pd.Series, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y,
        alerts,
        display_labels=["no stop ahead", "stop ahead"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.grid(False)
    ax.set_title(title)
    return fig


def plot_calibration(y: pd.Series, proba: pd.Series, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    CalibrationDisplay.from_predictions(
        y, proba, n_bins=10, strategy="quantile", ax=ax, color=SERIES[0]
    )
    ax.set_title(title)
    return fig


def plot_permutation_importance(
    model, X: pd.DataFrame, y: pd.Series, top: int = 15, seed: int = 0
) -> Figure:
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=3, random_state=seed
    )
    importance = pd.Series(result.importances_mean, index=X.columns).nlargest(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * top + 1))
    ax.barh(importance.index, importance.to_numpy(), color=SERIES[0], height=0.6)
    ax.set_xlabel("drop in PR-AUC when the feature is shuffled")
    ax.set_title(f"Top {top} features by permutation importance")
    return fig


def plot_alert_timeline(
    timestamps: pd.Series, proba: pd.Series, threshold: float, onsets: pd.Series
) -> Figure:
    """Predicted risk over the test period, with the actual fault onsets marked underneath.

    ``timestamps`` and ``onsets`` cover every sample in the period (including those inside a
    fault, which are not scored); ``proba`` covers the scored samples only.
    """
    minutes = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(minutes[proba.index], proba, color=SERIES[0], lw=0.9, label="predicted risk")
    ax.axhline(threshold, color=INK_SECONDARY, lw=1, label=f"alert threshold ({threshold:.2f})")
    ax.vlines(minutes[onsets], -0.12, -0.03, color=STOP, lw=1.2, label="actual fault onset")
    ax.set(xlabel="minutes into the test period", ylabel="P(fault within horizon)")
    ax.set_ylim(-0.14, 1.02)
    ax.set_title("Forecast risk vs. actual faults (held-out test period)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    return fig
