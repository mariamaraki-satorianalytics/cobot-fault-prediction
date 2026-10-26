"""Turn fault flags into events and prediction targets."""

import pandas as pd


def event_onsets(flag: pd.Series, session_id: pd.Series) -> pd.Series:
    """Mark the first sample of every run of 1s, without bridging recording sessions."""
    previous = flag.groupby(session_id).shift(fill_value=0)
    return flag.eq(1) & previous.eq(0)
