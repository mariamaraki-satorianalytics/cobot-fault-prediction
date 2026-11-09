"""Verify the raw UR3 CobotOps spreadsheet (fetching it from UCI if missing) and rebuild the
cleaned dataset in data/processed/."""

import argparse

from cobot.config import load_config
from cobot.data import download, load_dataset, write_processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="re-download the raw file from UCI")
    cfg = load_config().data

    raw_path = download(cfg, force=parser.parse_args().download)
    df = load_dataset(cfg)
    write_processed(df, cfg.processed_path)
    print(f"Raw data verified at {raw_path}")
    print(f"Wrote {len(df)} clean rows to {cfg.processed_path}")
