"""Typed project configuration loaded from YAML."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, HttpUrl, PositiveFloat, PositiveInt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class DataConfig(BaseModel):
    url: HttpUrl
    file_name: str
    sha256: str
    raw_dir: Path
    processed_dir: Path
    processed_file_name: str
    session_gap_s: PositiveFloat

    @property
    def raw_path(self) -> Path:
        return PROJECT_ROOT / self.raw_dir / self.file_name

    @property
    def processed_path(self) -> Path:
        return PROJECT_ROOT / self.processed_dir / self.processed_file_name


class FeatureConfig(BaseModel):
    signals: list[str]
    windows: list[PositiveInt]
    temperature_window: PositiveInt


class SearchConfig(BaseModel):
    model: str
    n_iter: PositiveInt
    space: dict[str, list[Any]]


class ModelConfig(BaseModel):
    horizon_s: PositiveInt
    test_fraction: float = Field(gt=0, lt=1)
    cv_splits: PositiveInt
    candidates: list[str]
    params: dict[str, dict[str, Any]] = {}
    search: SearchConfig | None = None


class MlflowConfig(BaseModel):
    tracking_uri: str
    experiment: str

    @property
    def resolved_uri(self) -> str:
        """Anchor a relative sqlite path at the project root, so runs land in one place."""
        prefix = "sqlite:///"
        path = Path(self.tracking_uri.removeprefix(prefix))
        if self.tracking_uri.startswith(prefix) and not path.is_absolute():
            return f"{prefix}{(PROJECT_ROOT / path).as_posix()}"
        return self.tracking_uri


class Config(BaseModel):
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    mlflow: MlflowConfig


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
