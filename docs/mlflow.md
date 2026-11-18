# Experiment tracking and model registry

Every training run is tracked in [MLflow](https://mlflow.org/). The tracking server runs in Docker, so runs, models and figures live in one place outside the project folder and survive restarts.

## The server

[`docker-compose.yml`](../docker-compose.yml) defines a single `mlflow` service:

| Setting | Value | Why |
|---|---|---|
| Image | `ghcr.io/mlflow/mlflow:v3.18.0` | Pinned to the same version as the Python client in `uv.lock` |
| Backend store | SQLite at `/mlflow/mlflow.db` | Experiments, runs, metrics and the model registry |
| Artifacts | `/mlflow/artifacts`, served through the server | Figures and model files; clients upload over HTTP and need no shared disk |
| Volume | `mlflow-data` mounted at `/mlflow` | Everything survives `docker compose down` and container restarts |
| Port | `5000` | The UI and API at http://localhost:5000 |
| Allowed hosts | `localhost`, `127.0.0.1`, `mlflow` | MLflow rejects unknown `Host` headers; `mlflow` is the service name other containers will use |
| Health check | `GET /health` every 10 s | `docker compose ps` shows when the server is ready |

```bash
docker compose up -d mlflow                                   # start the server in the background
uv run cobot-train --tracking-uri http://localhost:5000        # train against it
docker compose down                                           # stop it (data is kept in the volume)
```

Without `--tracking-uri` (or the `MLFLOW_TRACKING_URI` environment variable), training logs to a local `mlflow.db` file instead, which is handy for quick experiments without Docker.

## What a training run records

`cobot-train` creates one parent run per target, with a nested run for each candidate model:

- **Parameters:** target, horizon, train/test sizes, number of features, and each model's hyperparameters (including the random forest's tuned values).
- **Metrics:** `cv_*` (out-of-fold, used for model selection) and `test_*` (held-out period) for PR-AUC, ROC-AUC, precision, recall, F1, event recall, false alarms per hour and the alert threshold.
- **Figures:** precision-recall and ROC curves, confusion matrix, calibration, permutation importance and the risk timeline.
- **Model:** the winning pipeline (feature transformers + classifier) with an input example and signature. It takes raw sensor readings and returns class probabilities.

## Model registry

Each target has its own registered model: `cobot-protective-stop` and `cobot-grip-lost`. Every training run adds a new **version**. The **`champion`** alias marks the version that should be served:

- A new version takes `@champion` only if its cross-validation PR-AUC is **at least** the current champion's. The score is stored as a `cv_pr_auc` tag on each version.
- Cross-validation, not the test period, decides. The held-out data therefore never influences which model is served, the same rule used to pick the model within a run.
- Older versions stay in the registry, so rolling back is a matter of moving the alias in the UI.

The logic lives in [`cobot.registry`](../src/cobot/registry.py) and is unit-tested.

Loading the current champion anywhere, for example in the API coming in milestone 4:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.pyfunc.load_model("models:/cobot-protective-stop@champion")
probabilities = model.predict(readings)  # readings: session_id + the 19 sensor columns
```
