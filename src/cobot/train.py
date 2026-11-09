"""Train and compare fault-forecasting models, tracking every run in MLflow.

Usage: uv run cobot-train [--target protective_stop|grip_lost] [--save-figures]
"""

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import skops.io
from mlflow.models import infer_signature
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from cobot.config import PROJECT_ROOT, Config, load_config
from cobot.data import SENSORS, TARGETS, read_processed
from cobot.evaluate import (
    best_threshold,
    classification_metrics,
    out_of_fold_proba,
    plot_alert_timeline,
    plot_calibration,
    plot_confusion,
    plot_permutation_importance,
    plot_pr_curves,
    plot_roc_curves,
)
from cobot.features import make_feature_pipeline
from cobot.labels import event_onsets, make_onset_target
from cobot.models import build_model
from cobot.plotting import apply_style, save

FIGURES_DIR = PROJECT_ROOT / "docs" / "img"


@dataclass
class Split:
    X: pd.DataFrame
    y: pd.Series
    events: pd.Series


@dataclass
class Dataset:
    raw: pd.DataFrame
    features: Pipeline
    train: Split
    test: Split


def prepare(cfg: Config, target: str) -> Dataset:
    """Build features on the full time series, label it, then hold out the latest period."""
    df = read_processed(cfg.data.processed_path)
    features = make_feature_pipeline(**cfg.features.model_dump())
    X_all = features.fit_transform(df)
    labels = make_onset_target(df, target, cfg.model.horizon_s)

    usable = labels["usable"]
    X, y, events = X_all[usable], labels.loc[usable, "y"], labels.loc[usable, "event"]
    cut = int(len(X) * (1 - cfg.model.test_fraction))
    return Dataset(
        raw=df,
        features=features,
        train=Split(X.iloc[:cut], y.iloc[:cut], events.iloc[:cut]),
        test=Split(X.iloc[cut:], y.iloc[cut:], events.iloc[cut:]),
    )


def tune(estimator, cfg: Config, train: Split, cv: TimeSeriesSplit) -> dict:
    search = RandomizedSearchCV(
        estimator,
        cfg.model.search.space,
        n_iter=cfg.model.search.n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=0,
    ).fit(train.X, train.y)
    return search.best_params_


def run_candidate(name: str, cfg: Config, data: Dataset, cv: TimeSeriesSplit) -> dict:
    """Cross-validate, pick a threshold, refit on the full training period and score the test."""
    params = cfg.model.params.get(name, {})
    if cfg.model.search and cfg.model.search.model == name:
        params |= tune(build_model(name, params), cfg, data.train, cv)
    estimator = build_model(name, params)

    oof = out_of_fold_proba(estimator, data.train.X, data.train.y, cv)
    validated = oof.notna()
    threshold = best_threshold(data.train.y[validated], oof[validated])
    cv_metrics = classification_metrics(
        data.train.y[validated], oof[validated], threshold, data.train.events[validated]
    )

    model = estimator.fit(data.train.X, data.train.y)
    proba = pd.Series(model.predict_proba(data.test.X)[:, 1], index=data.test.X.index)
    test_metrics = classification_metrics(data.test.y, proba, threshold, data.test.events)

    mlflow.log_params({k: str(v) for k, v in params.items()})
    mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    return {
        "model": model,
        "proba": proba,
        "threshold": threshold,
        "cv": cv_metrics,
        "test": test_metrics,
    }


def report_figures(
    target: str, best: str, results: dict, data: Dataset, save_to_docs: bool
) -> None:
    test, chosen = data.test, results[best]
    # Every sample of the test period, including those inside a fault that are never scored.
    period = data.raw.loc[test.X.index[0] : test.X.index[-1]]
    onsets = event_onsets(data.raw[target], data.raw["session_id"]).loc[period.index]
    alerts = chosen["proba"].ge(chosen["threshold"])
    figures = {
        "pr_curves": plot_pr_curves(test.y, {n: r["proba"] for n, r in results.items()}),
        "roc_curves": plot_roc_curves(test.y, {n: r["proba"] for n, r in results.items()}),
        "confusion_matrix": plot_confusion(
            test.y, alerts, f"{best} at threshold {chosen['threshold']:.2f}"
        ),
        "calibration": plot_calibration(test.y, chosen["proba"], f"Calibration: {best}"),
        "permutation_importance": plot_permutation_importance(chosen["model"], test.X, test.y),
        "alert_timeline": plot_alert_timeline(
            period["timestamp"], chosen["proba"], chosen["threshold"], onsets
        ),
    }
    for name, fig in figures.items():
        mlflow.log_figure(fig, f"figures/{name}.png")
        if save_to_docs:
            save(fig, FIGURES_DIR / f"results_{target}_{name}.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--target", choices=TARGETS, default="protective_stop")
    parser.add_argument(
        "--save-figures", action="store_true", help=f"also write figures to {FIGURES_DIR}"
    )
    args = parser.parse_args()

    cfg = load_config()
    apply_style()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.resolved_uri))
    mlflow.set_experiment(cfg.mlflow.experiment)

    data = prepare(cfg, args.target)
    cv = TimeSeriesSplit(n_splits=cfg.model.cv_splits, gap=cfg.model.horizon_s)

    with mlflow.start_run(run_name=args.target):
        mlflow.log_params(
            {
                "target": args.target,
                "horizon_s": cfg.model.horizon_s,
                "n_train": len(data.train.y),
                "n_test": len(data.test.y),
                "n_features": data.train.X.shape[1],
            }
        )
        results = {}
        for name in cfg.model.candidates:
            with mlflow.start_run(run_name=name, nested=True):
                results[name] = run_candidate(name, cfg, data, cv)

        # Model selection uses cross-validation only; the test period is just reported.
        best = max(results, key=lambda n: results[n]["cv"]["pr_auc"])
        mlflow.set_tag("best_model", best)
        mlflow.log_metrics({f"best_test_{k}": v for k, v in results[best]["test"].items()})
        report_figures(args.target, best, results, data, args.save_figures)

        summary = pd.DataFrame(
            {n: {**r["test"], "cv_pr_auc": r["cv"]["pr_auc"]} for n, r in results.items()}
        ).T
        print(summary.round(3).to_string())
        print(f"\nBest model by CV PR-AUC: {best}")

        served = Pipeline([("features", data.features), ("model", results[best]["model"])])
        log_served_model(served, data.raw[["session_id", *SENSORS]].head(20))


def log_served_model(served: Pipeline, example: pd.DataFrame) -> None:
    """Log features + model as one object: raw sensor readings in, probability out.

    MLflow stores scikit-learn models with skops, which only reloads types it is told to trust.
    We trust our own feature classes and the scikit-learn internals of the model we just trained.
    """
    untrusted = skops.io.get_untrusted_types(data=skops.io.dumps(served))
    mlflow.sklearn.log_model(
        served,
        name="model",
        input_example=example,
        signature=infer_signature(example, served.predict_proba(example)),
        pyfunc_predict_fn="predict_proba",
        skops_trusted_types=[t for t in untrusted if t.startswith(("cobot.", "sklearn."))],
    )


if __name__ == "__main__":
    main()
