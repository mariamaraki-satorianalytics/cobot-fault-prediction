"""Download, load and clean the UR3 CobotOps dataset."""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from cobot.config import DataConfig

JOINTS = range(6)
CURRENTS = [f"current_j{i}" for i in JOINTS]
TEMPERATURES = [f"temperature_j{i}" for i in JOINTS]
SPEEDS = [f"speed_j{i}" for i in JOINTS]
SENSORS = [*CURRENTS, *TEMPERATURES, *SPEEDS, "tool_current"]
TARGETS = ["protective_stop", "grip_lost"]

# The raw file names joint 0's temperature "T0" and has inconsistent casing and spacing.
_RENAMES = {"temperature_t0": "temperature_j0", "robot_protectivestop": "protective_stop"}
# Speeds contain float noise such as 1e-44 where the joint is actually at rest.
_DENORMAL_EPS = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(cfg: DataConfig, force: bool = False) -> Path:
    """Fetch the UCI archive, extract the spreadsheet and verify its checksum."""
    path = cfg.raw_path
    if force or not path.exists():
        with urllib.request.urlopen(str(cfg.url), timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(cfg.file_name))

    if (actual := sha256(path)) != cfg.sha256:
        raise ValueError(f"Checksum mismatch for {path.name}: expected {cfg.sha256}, got {actual}")
    return path


def add_session_ids(timestamps: pd.Series, gap_s: float) -> pd.Series:
    """Number contiguous recording sessions, starting a new one after each long gap."""
    return timestamps.diff().dt.total_seconds().gt(gap_s).cumsum()


def clean(raw: pd.DataFrame, session_gap_s: float) -> pd.DataFrame:
    """Turn the raw spreadsheet into a tidy, typed, gap-aware time series."""
    df = raw.rename(columns=lambda c: c.strip().lower()).rename(columns=_RENAMES)
    df = df.drop(columns="num")
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip('"'), utc=True)
    df = df.sort_values("timestamp").dropna(subset=SENSORS, how="all")
    df["session_id"] = add_session_ids(df["timestamp"], session_gap_s)

    # A few rows keep a single reading; at 1 Hz the previous sample is the best estimate.
    filled = [*SENSORS, "protective_stop"]
    df[filled] = df.groupby("session_id")[filled].ffill()
    df = df.dropna(subset=filled)

    df[SPEEDS] = df[SPEEDS].mask(df[SPEEDS].abs() < _DENORMAL_EPS, 0.0)
    df[TARGETS] = df[TARGETS].astype(int)
    return df[["timestamp", "session_id", "cycle", *SENSORS, *TARGETS]].reset_index(drop=True)


def load_dataset(cfg: DataConfig) -> pd.DataFrame:
    return clean(pd.read_excel(cfg.raw_path), cfg.session_gap_s)
