"""Turn fault flags into events and prediction targets."""

import numpy as np
import pandas as pd


def event_onsets(flag: pd.Series, session_id: pd.Series) -> pd.Series:
    """Mark the first sample of every run of 1s, without bridging recording sessions."""
    previous = flag.groupby(session_id).shift(fill_value=0)
    return flag.eq(1) & previous.eq(0)


def make_onset_target(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Label each sample by whether a fault starts within the next ``horizon`` samples.

    Returns three columns aligned with ``df``:
    - ``y``: 1 if an onset falls in (t, t + horizon], else 0
    - ``event``: position of that upcoming onset (NaN when ``y`` is 0), used for event-level recall
    - ``usable``: False for samples already inside a fault (nothing left to predict) and for the
      last ``horizon`` samples of each session (their future is cut off by the session end)
    """
    session = df["session_id"]
    position = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    onset_position = position.where(event_onsets(df[target], session))
    # The first onset strictly after each sample, within the same session.
    next_onset = onset_position.groupby(session).transform(lambda s: s.shift(-1).bfill())

    y = (next_onset - position).le(horizon).astype(int)
    samples_left = position.groupby(session).transform("max") - position
    usable = df[target].eq(0) & samples_left.ge(horizon)
    return pd.DataFrame({"y": y, "event": next_onset.where(y.eq(1)), "usable": usable})
