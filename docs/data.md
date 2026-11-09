# Data cleaning

The raw UCI spreadsheet ([`data/raw/dataset_02052023.xlsx`](../data/raw/)) needed real work before it could be modelled. Every fix lives in [`cobot.data.clean`](../src/cobot/data.py) and is unit-tested in [`tests/test_data.py`](../tests/test_data.py). The notebook [`01_eda.ipynb`](../notebooks/01_eda.ipynb) shows how each issue was found.

| Issue in the raw file | Fix |
|---|---|
| A column named `cycle ` with a trailing space; mixed casing | Strip names and convert to snake_case |
| Joint 0's temperature is named `Temperature_T0` | Rename to `temperature_j0` |
| 863 timestamps wrapped in literal `"` quotes | Strip quotes, parse as UTC datetimes |
| 46 rows with every sensor missing | Drop them |
| 8 rows that kept only `Current_J0` | Forward-fill within the session (the previous 1 Hz sample is the best estimate) |
| 1,802 speed values like `1e-44` | Set to exactly 0: float noise where the joint is at rest |
| One target is a float with NaNs, the other a bool | Cast both to integers |

## Result

| | Raw | Cleaned |
|---|---|---|
| File | `data/raw/dataset_02052023.xlsx` | `data/processed/ur3_cobotops_clean.csv` |
| Rows | 7,409 | 7,363 |
| Columns | 24 | 24 |
| Missing values | 54 rows affected | none |

The cleaned data also gains a `session_id` column. The recording is two sessions on 26 October 2022, 49 and 76 minutes long, about five hours apart. Any gap longer than 60 s starts a new session; rolling features are computed per session so a window never spans the gap.

To rebuild the cleaned CSV after changing the cleaning code:

```bash
uv run python scripts/prepare_data.py
```
