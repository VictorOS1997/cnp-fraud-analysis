"""Clean, enrich and chart the transaction sample.

Run from the project root, after 00_download_data.py:

    python scripts/01_eda.py

Writes: outputs/figures/01..05_*.png and data/processed/transactions_enriched.csv
Each chart prints the sentence it supports — that sentence is what goes in the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
CSV_PATH = ROOT / "data" / "raw" / "transactional-sample.csv"
FIG_DIR = ROOT / "outputs" / "figures"
PROC_DIR = ROOT / "data" / "processed"

MIN_TX_PER_MERCHANT = 10  # below this, a chargeback rate is noise
MIN_TX_PER_HOUR = 30      # same idea for the hourly breakdown
BAR = "#4c72b0"
HIGHLIGHT = "#c44e52"


# --------------------------------------------------------------------------- load


def load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise SystemExit(f"missing {CSV_PATH} — run: python scripts/00_download_data.py")
    df = pd.read_csv(CSV_PATH)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["has_cbk"] = df["has_cbk"].astype(str).str.strip().str.upper().eq("TRUE")
    return df.sort_values("transaction_date").reset_index(drop=True)


# ----------------------------------------------------------------------- enrich


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns. Everything here must be knowable at decision time."""
    df = df.copy()

    df["hour"] = df["transaction_date"].dt.hour
    df["has_device"] = df["device_id"].notna()
    df["card_bin"] = df["card_number"].str[:6]
    df["card_last4"] = df["card_number"].str[-4:]

    # entity counts over the whole window (descriptive, not for real-time rules)
    df["user_tx_count"] = df.groupby("user_id")["transaction_id"].transform("size")
    df["user_card_count"] = df.groupby("user_id")["card_number"].transform("nunique")
    df["card_user_count"] = df.groupby("card_number")["user_id"].transform("nunique")
    df["device_user_count"] = df.groupby("device_id")["user_id"].transform("nunique")
    df["merchant_tx_count"] = df.groupby("merchant_id")["transaction_id"].transform("size")

    # time since the previous transaction of the same user / same card
    df["seconds_since_prev_user_tx"] = (
        df.groupby("user_id")["transaction_date"].diff().dt.total_seconds()
    )
    df["seconds_since_prev_card_tx"] = (
        df.groupby("card_number")["transaction_date"].diff().dt.total_seconds()
    )
    df["prev_card_amount"] = df.groupby("card_number")["transaction_amount"].shift(1)

    # amount relative to the user's own history — only meaningful with enough history
    user_median = df.groupby("user_id")["transaction_amount"].transform("median")
    df["amount_vs_user_median"] = df["transaction_amount"] / user_median

    # PAST ONLY: chargebacks of this user strictly before the current transaction.
    # Using the user's total would leak the future into the present.
    df["user_cbk_before"] = df.groupby("user_id")["has_cbk"].transform(
        lambda s: s.shift(1, fill_value=False).cumsum()
    )
    df["is_repeat_offender"] = df["user_cbk_before"] > 0

    return df


# ----------------------------------------------------------------------- charts


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def chart_01_by_hour(df: pd.DataFrame, base_rate: float) -> None:
    g = df.groupby("hour")["has_cbk"].agg(transactions="size", rate="mean")

    # Ignore hours with too little volume — a 50% rate over 2 transactions is noise.
    eligible = g[g["transactions"] >= MIN_TX_PER_HOUR]
    worst = eligible["rate"].idxmax() if not eligible.empty else g["rate"].idxmax()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(g.index, g["transactions"], color=BAR, label="transactions")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("transactions")
    ax.set_xticks(range(0, 24))

    ax2 = ax.twinx()
    # Only plot the rate where there is enough volume to mean anything.
    rate = g["rate"].where(g["transactions"] >= MIN_TX_PER_HOUR) * 100
    ax2.plot(g.index, rate, color=HIGHLIGHT, marker="o", label="chargeback rate")
    ax2.axhline(base_rate * 100, color="grey", linestyle="--", linewidth=1)
    ax2.set_ylabel("chargeback rate (%)")

    ax.set_title(
        f"Chargeback rate peaks at {worst}h ({g.loc[worst, 'rate'] * 100:.1f}%) "
        f"vs {base_rate * 100:.1f}% overall\n"
        f"(rate shown only for hours with at least {MIN_TX_PER_HOUR} transactions)",
        fontsize=11,
    )
    _save(fig, "01_chargeback_by_hour.png")
    print(f"  -> worst hour: {worst}h, rate {g.loc[worst, 'rate'] * 100:.1f}%, "
          f"{int(g.loc[worst, 'transactions'])} transactions")


def chart_02_amounts(df: pd.DataFrame) -> None:
    legit = df.loc[~df["has_cbk"], "transaction_amount"]
    fraud = df.loc[df["has_cbk"], "transaction_amount"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([legit, fraud], tick_labels=["no chargeback", "chargeback"], showfliers=False)
    ax.set_ylabel("transaction amount")
    ratio = fraud.median() / legit.median() if legit.median() else float("nan")
    ax.set_title(
        f"Median ticket: {legit.median():,.0f} vs {fraud.median():,.0f} "
        f"({ratio:.1f}x higher on fraud)"
    )
    _save(fig, "02_amount_distribution.png")
    print(f"  -> median legit {legit.median():,.2f} | median fraud {fraud.median():,.2f} "
          f"({len(fraud)} fraudulent transactions)")


def chart_03_merchants(df: pd.DataFrame, base_rate: float) -> None:
    m = (
        df.groupby("merchant_id")
        .agg(transactions=("transaction_id", "size"), rate=("has_cbk", "mean"))
        .query("transactions >= @MIN_TX_PER_MERCHANT")
        .sort_values("rate", ascending=False)
        .head(12)
    )
    if m.empty:
        print("  !! no merchant reaches the minimum volume — skipping chart 03")
        return

    labels = [f"{mid} (n={int(n)})" for mid, n in zip(m.index, m["transactions"])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, m["rate"] * 100, color=BAR)
    ax.axvline(base_rate * 100, color=HIGHLIGHT, linestyle="--", linewidth=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("chargeback rate (%)")
    ax.set_title(
        f"Riskiest merchants (min. {MIN_TX_PER_MERCHANT} transactions) — "
        f"dashed line is the {base_rate * 100:.1f}% base rate"
    )
    _save(fig, "03_chargeback_by_merchant.png")
    print(f"  -> worst merchant {m.index[0]}: {m['rate'].iloc[0] * 100:.1f}% "
          f"over {int(m['transactions'].iloc[0])} transactions")


def chart_04_velocity(df: pd.DataFrame, base_rate: float) -> None:
    buckets = pd.cut(
        df["user_tx_count"],
        bins=[0, 1, 2, 3, 5, 10, 10_000],
        labels=["1", "2", "3", "4-5", "6-10", "11+"],
    )
    g = df.groupby(buckets, observed=True)["has_cbk"].agg(transactions="size", rate="mean")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(g.index.astype(str), g["rate"] * 100, color=BAR)
    ax.axhline(base_rate * 100, color=HIGHLIGHT, linestyle="--", linewidth=1.2)
    for i, (rate, n) in enumerate(zip(g["rate"], g["transactions"])):
        ax.text(i, rate * 100, f"n={int(n)}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("transactions by the same user in the period")
    ax.set_ylabel("chargeback rate (%)")
    ax.set_title("Chargeback rate rises with the number of transactions per user")
    _save(fig, "04_chargeback_by_user_velocity.png")
    print("  -> " + " | ".join(
        f"{idx}: {row['rate'] * 100:.1f}% (n={int(row['transactions'])})"
        for idx, row in g.iterrows()
    ))


def chart_05_repeat_offenders(df: pd.DataFrame, base_rate: float) -> None:
    g = df.groupby("is_repeat_offender")["has_cbk"].agg(transactions="size", rate="mean")
    if True not in g.index:
        print("  !! no user has a second transaction after a chargeback — skipping chart 05")
        return

    labels = ["no prior chargeback", "after a chargeback"]
    rates = [g.loc[False, "rate"] * 100, g.loc[True, "rate"] * 100]
    counts = [int(g.loc[False, "transactions"]), int(g.loc[True, "transactions"])]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, rates, color=[BAR, HIGHLIGHT])
    ax.axhline(base_rate * 100, color="grey", linestyle="--", linewidth=1)
    for i, (r, n) in enumerate(zip(rates, counts)):
        ax.text(i, r, f"{r:.1f}%  (n={n})", ha="center", va="bottom")
    ax.set_ylabel("chargeback rate (%)")
    ax.set_title(
        "After a chargeback the same user's next transactions are almost all fraud —\n"
        "descriptive only: those transactions arrive hours later, the dispute weeks later "
        "(see figure 07)",
        fontsize=10,
    )
    _save(fig, "05_repeat_offenders.png")
    lift = rates[1] / rates[0] if rates[0] else float("inf")
    print(f"  -> {rates[1]:.1f}% after a chargeback vs {rates[0]:.1f}% before — "
          f"{lift:.1f}x, over {counts[1]} transactions")


# ------------------------------------------------------------------------- main


def main() -> int:
    df = enrich(load())
    base_rate = df["has_cbk"].mean()

    print(f"\nrows: {len(df):,} | base chargeback rate: {base_rate * 100:.2f}%\n")
    print("charts")
    chart_01_by_hour(df, base_rate)
    chart_02_amounts(df)
    chart_03_merchants(df, base_rate)
    chart_04_velocity(df, base_rate)
    chart_05_repeat_offenders(df, base_rate)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / "transactions_enriched.csv"
    df.to_csv(out, index=False)
    print(f"\nenriched dataset: {out}")
    print("Next: python scripts/02_rules.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
