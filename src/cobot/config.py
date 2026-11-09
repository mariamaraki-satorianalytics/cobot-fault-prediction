"""Typed project configuration loaded from YAML."""

from pathlib import Path

import yaml
from pydantic import BaseModel, HttpUrl, PositiveFloat

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


class Config(BaseModel):
    data: DataConfig


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
