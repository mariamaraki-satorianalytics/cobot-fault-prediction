"""Download the UR3 CobotOps dataset from UCI into data/raw/ and verify its checksum."""

import argparse

from cobot.config import load_config
from cobot.data import download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    path = download(load_config().data, force=parser.parse_args().force)
    print(f"Dataset ready at {path}")
