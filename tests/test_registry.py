import pytest
from mlflow import MlflowClient

from cobot.registry import CHAMPION, promote_if_better, registered_name


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> MlflowClient:
    # Creating an MLflow database is slow, so the tests share one and use separate model names.
    uri = f"sqlite:///{(tmp_path_factory.mktemp('mlflow') / 'mlflow.db').as_posix()}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


@pytest.fixture
def model(client, request) -> str:
    name = request.node.name.replace("_", "-")
    client.create_registered_model(name)
    return name


def new_version(client: MlflowClient, name: str, tmp_path) -> str:
    return client.create_model_version(name, source=tmp_path.as_uri()).version


def champion(client: MlflowClient, name: str) -> str:
    return client.get_model_version_by_alias(name, CHAMPION).version


def test_registered_names_are_per_target():
    assert registered_name("cobot", "protective_stop") == "cobot-protective-stop"


def test_first_version_becomes_champion(client, model, tmp_path):
    v1 = new_version(client, model, tmp_path)
    assert promote_if_better(client, model, v1, 0.20)
    assert champion(client, model) == v1


def test_champion_moves_only_to_an_equal_or_better_version(client, model, tmp_path):
    v1 = new_version(client, model, tmp_path)
    promote_if_better(client, model, v1, 0.20)

    worse = new_version(client, model, tmp_path)
    assert not promote_if_better(client, model, worse, 0.15)
    assert champion(client, model) == v1

    better = new_version(client, model, tmp_path)
    assert promote_if_better(client, model, better, 0.25)
    assert champion(client, model) == better
