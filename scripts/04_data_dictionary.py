"""Build the data dictionaries for the raw and processed datasets.

Run from the project root, after 01_eda.py:

    python scripts/04_data_dictionary.py

Writes: outputs/data_dictionary_raw.csv
        outputs/data_dictionary_processed.csv

Each row documents one column: what it means, how it is computed, the profile measured on the
current file, whether it is safe to use in a real-time rule, and the pitfall to avoid.

The `time_safety` field is the one to read before building any rule:
  raw            — present in the source file, known at authorization time
  past-only      — derived, uses only information available before the transaction
  whole-period   — derived over the entire dataset; DESCRIPTIVE ONLY, it sees the future
  lagging-label  — derived from chargebacks, which in production arrive days to weeks later
  target         — the label itself, never an input
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
RAW_PATH = ROOT / "data" / "raw" / "transactional-sample.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "transactions_enriched.csv"
OUT_DIR = ROOT / "outputs"

# column -> (role, unit, time_safety, description, formula, pitfall)
META: dict[str, tuple[str, str, str, str, str, str]] = {
    # ------------------------------------------------------------------ raw columns
    "transaction_id": (
        "identifier", "-", "raw",
        "Unique identifier of the transaction.",
        "source file",
        "Looks sequential and correlates with time; it is an identifier, never a feature.",
    ),
    "merchant_id": (
        "identifier", "-", "raw",
        "Merchant that received the payment.",
        "source file",
        "70.7% of merchants appear once. Any per-merchant rate needs a minimum volume filter.",
    ),
    "user_id": (
        "identifier", "-", "raw",
        "Cardholder placing the transaction.",
        "source file",
        "Behaves like a merchant-side customer id (only 0.9% of users appear at more than one "
        "merchant) and is created by the customer at sign-up, so it is attacker-controlled.",
    ),
    "card_number": (
        "identifier", "-", "raw",
        "Masked card number: first 6 digits (BIN) plus last 4.",
        "source file",
        "Pseudo-identifier only — two different cards from the same issuer can collide on the "
        "last 4 digits.",
    ),
    "transaction_date": (
        "attribute", "timestamp", "raw",
        "Date and time of the transaction, microsecond precision.",
        "source file",
        "No timezone declared; a single timezone is assumed. Sort by this before any "
        "time-based feature.",
    ),
    "transaction_amount": (
        "attribute", "currency units", "raw",
        "Transaction amount.",
        "source file",
        "No currency declared (BRL assumed). Strongly right-skewed — use the median, not the "
        "mean, to describe it.",
    ),
    "device_id": (
        "identifier", "-", "raw",
        "Identifier of the device used in the transaction.",
        "source file",
        "26% missing. Absence is a category, not dirt — dropping those rows removes a quarter "
        "of the base. In this sample each device belongs to a single user.",
    ),
    "has_cbk": (
        "target", "boolean", "target",
        "Whether the transaction received a fraud-related chargeback. The fraud label.",
        "source file (TRUE/FALSE strings)",
        "Never use as an input feature. It is a LAGGING label: the dispute arrives days to "
        "weeks after the transaction, so recent fraud is still marked FALSE.",
    ),
    # ------------------------------------------------------------ derived, safe
    "hour": (
        "derived feature", "hour of day (0-23)", "past-only",
        "Hour of day of the transaction.",
        "transaction_date.dt.hour",
        "Describes when the attacker operates, not why. Low-volume hours produce meaningless "
        "rates — filter by minimum volume before ranking.",
    ),
    "has_device": (
        "derived feature", "boolean", "past-only",
        "Whether a device identifier was captured.",
        "device_id.notna()",
        "In this base, missing device has a LOWER chargeback rate (8.1% vs 13.7%) — the "
        "intuitive rule would have been wrong.",
    ),
    "card_bin": (
        "derived feature", "-", "past-only",
        "Bank Identification Number: first 6 digits of the card, identifying the issuer.",
        "card_number.str[:6]",
        "Keep it as text — parsing to integer destroys leading zeros.",
    ),
    "card_last4": (
        "derived feature", "-", "past-only",
        "Last four digits of the card.",
        "card_number.str[-4:]",
        "Only meaningful combined with the BIN.",
    ),
    "seconds_since_prev_user_tx": (
        "derived feature", "seconds", "past-only",
        "Seconds elapsed since the previous transaction of the same user.",
        "groupby(user_id).transaction_date.diff()",
        "Null on a user's first transaction — treat null as 'no history', not as zero.",
    ),
    "seconds_since_prev_card_tx": (
        "derived feature", "seconds", "past-only",
        "Seconds elapsed since the previous transaction on the same card.",
        "groupby(card_number).transaction_date.diff()",
        "Same null caveat as the user version.",
    ),
    "prev_card_amount": (
        "derived feature", "currency units", "past-only",
        "Amount of the previous transaction on the same card.",
        "groupby(card_number).transaction_amount.shift(1)",
        "Used for the card-testing pattern (small probe followed by a large charge); without "
        "declined authorizations in the data this pattern is nearly invisible.",
    ),
    # -------------------------------------------------- derived, whole-period (unsafe)
    "user_tx_count": (
        "derived feature", "count", "whole-period",
        "Number of transactions by this user across the entire dataset.",
        "groupby(user_id).transaction_id.transform('size')",
        "Counts transactions that happen AFTER the current one. Descriptive only — the "
        "real-time equivalent is a rolling window of prior transactions.",
    ),
    "user_card_count": (
        "derived feature", "count", "whole-period",
        "Distinct cards used by this user across the entire dataset.",
        "groupby(user_id).card_number.transform('nunique')",
        "Same future leakage; the operational version counts only cards seen before now.",
    ),
    "card_user_count": (
        "derived feature", "count", "whole-period",
        "Distinct users that used this card across the entire dataset.",
        "groupby(card_number).user_id.transform('nunique')",
        "Same future leakage. Also inherits the masked-card collision caveat.",
    ),
    "device_user_count": (
        "derived feature", "count", "whole-period",
        "Distinct users that used this device across the entire dataset.",
        "groupby(device_id).user_id.transform('nunique')",
        "Always 1 in this sample — the device-sharing hypothesis could not be tested here.",
    ),
    "merchant_tx_count": (
        "derived feature", "count", "whole-period",
        "Number of transactions of this merchant across the entire dataset.",
        "groupby(merchant_id).transaction_id.transform('size')",
        "Future leakage; use as a volume filter for descriptive rankings only.",
    ),
    "amount_vs_user_median": (
        "derived feature", "ratio", "whole-period",
        "Transaction amount divided by the median amount of the same user.",
        "transaction_amount / groupby(user_id).transaction_amount.transform('median')",
        "The median includes future transactions, and with 1.18 transactions per user most "
        "users have a median equal to the transaction itself, giving a ratio of 1.0.",
    ),
    # ------------------------------------------------------- derived from the label
    "user_cbk_before": (
        "derived feature", "count", "lagging-label",
        "Number of chargebacks this user had strictly before the current transaction.",
        "groupby(user_id).has_cbk.transform(shift(1).cumsum())",
        "Correct in ordering (no future rows), but NOT knowable in real time: 82.6% of these "
        "transactions occur within 24h of the disputed one, weeks before the dispute exists. "
        "Belongs in a block list, never in an authorization rule.",
    ),
    "is_repeat_offender": (
        "derived feature", "boolean", "lagging-label",
        "Whether the user already had at least one chargeback before this transaction.",
        "user_cbk_before > 0",
        "89.8% of these transactions are fraud, which makes it the most tempting and least "
        "usable signal in the dataset. See the dispute-delay table in the report.",
    ),
}

DEFAULT_META = (
    "derived feature", "-", "review",
    "Not documented yet — add it to META in scripts/04_data_dictionary.py.",
    "unknown", "Undocumented column.",
)


def profile(series: pd.Series) -> dict:
    """Measured profile of one column on the current file."""
    non_null = int(series.notna().sum())
    out = {
        "dtype": str(series.dtype),
        "non_null": non_null,
        "null_pct": round((1 - non_null / len(series)) * 100, 2),
        "distinct": int(series.nunique(dropna=True)),
        "min": "",
        "max": "",
        "example": "",
    }

    clean = series.dropna()
    if not clean.empty:
        out["example"] = str(clean.iloc[0])
        if pd.api.types.is_numeric_dtype(clean) and not pd.api.types.is_bool_dtype(clean):
            out["min"] = round(float(clean.min()), 2)
            out["max"] = round(float(clean.max()), 2)
        elif pd.api.types.is_datetime64_any_dtype(clean):
            out["min"] = str(clean.min())
            out["max"] = str(clean.max())
        elif pd.api.types.is_bool_dtype(clean):
            out["min"] = f"{clean.mean() * 100:.2f}% true"
    return out


def build(df: pd.DataFrame, dataset: str, source: str) -> pd.DataFrame:
    rows = []
    for position, column in enumerate(df.columns, start=1):
        role, unit, safety, description, formula, pitfall = META.get(column, DEFAULT_META)
        rows.append(
            {
                "dataset": dataset,
                "source_file": source,
                "position": position,
                "column": column,
                "role": role,
                "unit": unit,
                "time_safety": safety,
                "description": description,
                "formula": formula,
                **profile(df[column]),
                "pitfall": pitfall,
            }
        )
    return pd.DataFrame(rows)


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise SystemExit(f"missing {RAW_PATH} — run: python scripts/00_download_data.py")
    df = pd.read_csv(RAW_PATH, dtype={"card_number": str})
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["has_cbk"] = df["has_cbk"].astype(str).str.strip().str.upper().eq("TRUE")
    return df


def load_processed() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise SystemExit(f"missing {PROCESSED_PATH} — run: python scripts/01_eda.py")
    return pd.read_csv(
        PROCESSED_PATH,
        parse_dates=["transaction_date"],
        dtype={"card_number": str, "card_bin": str, "card_last4": str},
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_dict = build(load_raw(), "raw", "data/raw/transactional-sample.csv")
    proc_dict = build(load_processed(), "processed", "data/processed/transactions_enriched.csv")

    raw_out = OUT_DIR / "data_dictionary_raw.csv"
    proc_out = OUT_DIR / "data_dictionary_processed.csv"
    raw_dict.to_csv(raw_out, index=False, encoding="utf-8-sig")
    proc_dict.to_csv(proc_out, index=False, encoding="utf-8-sig")

    for name, table, path in (
        ("RAW", raw_dict, raw_out),
        ("PROCESSED", proc_dict, proc_out),
    ):
        print(f"\n{'=' * 78}\n{name} — {len(table)} columns\n{'=' * 78}")
        print(table[["column", "role", "time_safety", "dtype", "null_pct", "distinct"]]
              .to_string(index=False))
        print(f"wrote {path}")

    undocumented = [
        c for t in (raw_dict, proc_dict) for c in t.loc[t["formula"] == "unknown", "column"]
    ]
    if undocumented:
        print(f"\nWARNING — columns without documentation: {', '.join(undocumented)}")

    print("\nUsable in a real-time rule: rows marked 'raw' or 'past-only'.")
    print("Everything marked 'whole-period' or 'lagging-label' is descriptive only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
