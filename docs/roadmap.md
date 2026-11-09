# Roadmap

The project is built in four milestones. Each one is merged from `dev` into `main` through a pull request.

| Milestone | Scope | Status |
|---|---|---|
| **M1: data foundation** | uv project, raw and cleaned data with tests, EDA notebook, README and UR3 primer | ✅ done |
| **M2: modelling** | Onset labels, physics and rolling-window features (with a leakage test), time-aware CV, baselines vs. Random Forest, evaluation plots, MLflow tracking | ✅ done |
| **M3: MLflow in Docker** | `docker compose` MLflow server, model registry with a `champion` alias | ⏳ next |
| **M4: serving and CI** | FastAPI `/predict` and `/health`, Dockerfile, GitHub Actions (lint, test, build) | |
