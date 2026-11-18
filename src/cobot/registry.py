"""Model registry: version every trained model and keep a ``champion`` alias on the best one."""

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION = "champion"
SCORE_TAG = "cv_pr_auc"


def registered_name(prefix: str, target: str) -> str:
    """One registered model per target, e.g. ``cobot-protective-stop``."""
    return f"{prefix}-{target.replace('_', '-')}"


def promote_if_better(client: MlflowClient, name: str, version: str, score: float) -> bool:
    """Move the champion alias to ``version`` unless the current champion scored higher.

    Versions are compared on cross-validation PR-AUC, the same metric used to pick the model
    within a run, so the held-out test period never influences which model gets served.
    """
    client.set_model_version_tag(name, version, SCORE_TAG, str(score))
    try:
        champion = client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:  # no champion yet
        champion = None
    if champion is not None and float(champion.tags.get(SCORE_TAG, "-inf")) > score:
        return False
    client.set_registered_model_alias(name, CHAMPION, version)
    return True
