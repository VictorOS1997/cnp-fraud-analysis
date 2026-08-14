"""Download the transaction sample and print the day-one sanity checks.

Run from the project root:

    python scripts/00_download_data.py

Writes: data/raw/transactional-sample.csv
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://gist.githubusercontent.com/cloudwalk-tests/"
    "76993838e65d7e0f988f40f1b1909c97/raw/transactional-sample.csv"
)

ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "transactional-sample.csv"


def download(force: bool = False) -> Path:
    """Fetch the CSV once. Existing file is kept unless force is set."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists() and not force:
        print(f"[skip] file already present: {CSV_PATH}")
        return CSV_PATH

    print(f"[get ] {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, CSV_PATH)
    print(f"[ok  ] saved to {CSV_PATH} ({CSV_PATH.stat().st_size / 1024:.1f} KB)")
    return CSV_PATH


def load(path: Path) -> pd.DataFrame:
    """Load and type the raw file. has_cbk arrives as the strings TRUE/FALSE."""
    df = pd.read_csv(path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["has_cbk"] = df["has_cbk"].astype(str).str.strip().str.upper().eq("TRUE")
    return df.sort_values("transaction_date").reset_index(drop=True)


def sanity_checks(df: pd.DataFrame) -> None:
    """Print the numbers every later conclusion is compared against."""
    line = "-" * 62

    print(f"\n{line}\nSHAPE\n{line}")
    print(f"rows: {len(df):,}   columns: {df.shape[1]}")
    print(df.dtypes.to_string())

    print(f"\n{line}\nMISSING VALUES\n{line}")
    missing = df.isna().sum()
    for col, n in missing.items():
        pct = n / len(df) * 100
        print(f"{col:22} {n:>7,}  ({pct:5.2f}%)")

    print(f"\n{line}\nDUPLICATES\n{line}")
    print(f"duplicated transaction_id: {df['transaction_id'].duplicated().sum():,}")
    print(f"fully duplicated rows:     {df.duplicated().sum():,}")

    print(f"\n{line}\nPERIOD\n{line}")
    start, end = df["transaction_date"].min(), df["transaction_date"].max()
    print(f"from {start} to {end}")
    print(f"span: {(end - start).days} days ({(end - start).total_seconds() / 3600:.1f} hours)")

    print(f"\n{line}\nBASE RATE  <-- write this one down\n{line}")
    n_cbk = int(df["has_cbk"].sum())
    print(f"chargebacks: {n_cbk:,} of {len(df):,}  =  {df['has_cbk'].mean() * 100:.2f}%")

    total_amount = df["transaction_amount"].sum()
    cbk_amount = df.loc[df["has_cbk"], "transaction_amount"].sum()
    print(f"volume:      {total_amount:,.2f}")
    print(f"disputed:    {cbk_amount:,.2f}  ({cbk_amount / total_amount * 100:.2f}% of volume)")

    print(f"\n{line}\nAMOUNTS\n{line}")
    print(df["transaction_amount"].describe().to_string())
    print(f"negative or zero amounts: {(df['transaction_amount'] <= 0).sum():,}")

    print(f"\n{line}\nCARDINALITY\n{line}")
    for col in ["user_id", "card_number", "device_id", "merchant_id"]:
        n = df[col].nunique()
        print(f"{col:22} {n:>7,} distinct   ({len(df) / n:.1f} transactions each on average)")

    print(f"\n{line}\nMISSING DEVICE vs FRAUD  (do not drop these rows)\n{line}")
    by_device = df.assign(has_device=df["device_id"].notna()).groupby("has_device")["has_cbk"]
    print(by_device.agg(chargeback_rate="mean", transactions="size").to_string())


def main() -> int:
    force = "--force" in sys.argv
    path = download(force=force)
    df = load(path)
    sanity_checks(df)
    print("\nNext: python scripts/01_eda.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
