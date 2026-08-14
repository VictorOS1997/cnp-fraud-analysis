"""Run the fraud rules, measure each one and write the alert list.

Run from the project root, after 01_eda.py:

    python scripts/02_rules.py

Writes: outputs/alerts.csv, outputs/rule_performance.csv,
        outputs/figures/06_rule_performance.png

Thresholds live in THRESHOLDS below. They are a starting point, not an answer:
tune them against the printed table and justify the values you keep.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
ENRICHED = ROOT / "data" / "processed" / "transactions_enriched.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

THRESHOLDS = {
    "velocity_seconds": 300,        # R01 two transactions by the same user within 5 min
    "cards_per_user": 3,            # R04
    "amount_ratio": 3.0,            # R07 amount over 3x the user's own median
    "min_user_history": 3,          # R07 only for users with enough history
    "testing_small": 20.0,          # R08 probe amount
    "testing_large": 200.0,         # R08 follow-up amount
    "testing_window_s": 3600,       # R08
    "night_hours": (0, 1, 2, 3, 4, 5),  # R09
    "merchant_min_tx": 10,          # R10 minimum volume to rank a merchant
    "merchant_rate_multiple": 2.0,  # R10 merchant rate above 2x the base rate
    "hop_seconds": 600,             # R12 different merchant within 10 min
}


def load() -> pd.DataFrame:
    if not ENRICHED.exists():
        raise SystemExit(f"missing {ENRICHED} — run: python scripts/01_eda.py")
    df = pd.read_csv(
        ENRICHED,
        parse_dates=["transaction_date"],
        dtype={"card_number": str, "card_bin": str, "card_last4": str},
    )
    df["has_cbk"] = df["has_cbk"].astype(bool)
    return df.sort_values("transaction_date").reset_index(drop=True)


def build_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Each rule returns a boolean Series aligned with df."""
    t = THRESHOLDS
    base_rate = df["has_cbk"].mean()

    # R10 uses merchant history. Computed here on the same data for illustration;
    # in production it must come from a previous period, never from the current one.
    merchant_rate = df.groupby("merchant_id")["has_cbk"].transform("mean")
    merchant_volume = df.groupby("merchant_id")["transaction_id"].transform("size")

    # R12 needs the previous merchant of the same user
    prev_merchant = df.groupby("user_id")["merchant_id"].shift(1)

    # R11 counts identical card/amount pairs
    repeated_pair = df.groupby(["card_number", "transaction_amount"])["transaction_id"].transform(
        "size"
    )

    return {
        "R01_velocity_same_user": df["seconds_since_prev_user_tx"] < t["velocity_seconds"],
        "R02_repeat_offender": df["is_repeat_offender"].astype(bool),
        "R03_card_shared_by_users": df["card_user_count"] > 1,
        "R04_many_cards_per_user": df["user_card_count"] >= t["cards_per_user"],
        "R05_device_shared_by_users": df["device_user_count"] > 1,
        "R06_missing_device": ~df["has_device"].astype(bool),
        "R07_amount_above_user_profile": (
            (df["amount_vs_user_median"] > t["amount_ratio"])
            & (df["user_tx_count"] >= t["min_user_history"])
        ),
        "R08_card_testing_pattern": (
            (df["prev_card_amount"] < t["testing_small"])
            & (df["transaction_amount"] > t["testing_large"])
            & (df["seconds_since_prev_card_tx"] < t["testing_window_s"])
        ),
        "R09_night_hours": df["hour"].isin(t["night_hours"]),
        "R10_high_risk_merchant": (
            (merchant_volume >= t["merchant_min_tx"])
            & (merchant_rate > base_rate * t["merchant_rate_multiple"])
        ),
        "R11_repeated_card_amount": repeated_pair > 1,
        "R12_merchant_hopping": (
            (df["merchant_id"] != prev_merchant)
            & (df["seconds_since_prev_user_tx"] < t["hop_seconds"])
        ),
    }


def evaluate(df: pd.DataFrame, rules: dict[str, pd.Series]) -> pd.DataFrame:
    """Precision, recall and operational cost of each rule."""
    total_fraud = int(df["has_cbk"].sum())
    rows = []
    for name, mask in rules.items():
        mask = mask.fillna(False).astype(bool)
        alerts = int(mask.sum())
        caught = int(df.loc[mask, "has_cbk"].sum())
        rows.append(
            {
                "rule": name,
                "alerts": alerts,
                "alert_share": alerts / len(df),
                "fraud_caught": caught,
                "precision": caught / alerts if alerts else 0.0,
                "recall": caught / total_fraud if total_fraud else 0.0,
                "alerts_per_fraud": alerts / caught if caught else float("inf"),
            }
        )
    return pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)


def combined_coverage(df: pd.DataFrame, rules: dict[str, pd.Series]) -> pd.Series:
    """Union of all rules — rules overlap, so recalls cannot be added up."""
    any_rule = pd.Series(False, index=df.index)
    for mask in rules.values():
        any_rule |= mask.fillna(False).astype(bool)
    return any_rule


def chart_06(perf: pd.DataFrame, base_rate: float) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = perf.sort_values("recall")

    y = range(len(d))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([i + 0.2 for i in y], d["recall"] * 100, height=0.38,
            color="#4c72b0", label="fraud caught (recall %)")
    ax.barh([i - 0.2 for i in y], d["precision"] * 100, height=0.38,
            color="#c44e52", label="precision %")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["rule"])
    ax.axvline(base_rate * 100, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("%")
    ax.set_title("Rule performance — coverage vs precision (dashed line: base rate)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = FIG_DIR / "06_rule_performance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def main() -> int:
    df = load()
    base_rate = df["has_cbk"].mean()
    rules = build_rules(df)
    perf = evaluate(df, rules)

    print(f"\nrows: {len(df):,} | base chargeback rate: {base_rate * 100:.2f}% "
          f"| fraud: {int(df['has_cbk'].sum())}\n")

    shown = perf.assign(
        alerts_pct=lambda d: (d["alert_share"] * 100).round(1),
        precision_pct=lambda d: (d["precision"] * 100).round(1),
        recall_pct=lambda d: (d["recall"] * 100).round(1),
        per_fraud=lambda d: d["alerts_per_fraud"].round(1),
    )[["rule", "alerts", "alerts_pct", "fraud_caught", "precision_pct", "recall_pct", "per_fraud"]]
    print(shown.to_string(index=False))

    any_rule = combined_coverage(df, rules)
    caught = int(df.loc[any_rule, "has_cbk"].sum())
    total_fraud = int(df["has_cbk"].sum())
    print(
        f"\nall rules combined: {int(any_rule.sum()):,} alerts "
        f"({any_rule.mean() * 100:.1f}% of the base), "
        f"catching {caught}/{total_fraud} frauds ({caught / total_fraud * 100:.1f}%), "
        f"precision {caught / any_rule.sum() * 100:.1f}%"
    )
    print("Rules overlap — this is the real coverage, not the sum of the individual recalls.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perf.to_csv(OUT_DIR / "rule_performance.csv", index=False)

    rule_frame = pd.DataFrame(
        {name: mask.fillna(False).astype(bool) for name, mask in rules.items()}
    )
    fired = rule_frame.loc[any_rule]
    alerts = df.loc[any_rule].copy()
    alerts["triggered_rules"] = fired.apply(
        lambda row: ",".join(fired.columns[row.to_numpy()]), axis=1
    )
    alerts["rule_count"] = fired.sum(axis=1)
    cols = [
        "transaction_id", "transaction_date", "user_id", "merchant_id", "card_number",
        "transaction_amount", "device_id", "has_cbk", "rule_count", "triggered_rules",
    ]
    alerts.sort_values("rule_count", ascending=False)[cols].to_csv(
        OUT_DIR / "alerts.csv", index=False
    )

    print(f"\nwrote {OUT_DIR / 'alerts.csv'} and {OUT_DIR / 'rule_performance.csv'}")
    print("charts")
    chart_06(perf, base_rate)

    print("\nHow to read the table:")
    print("  high precision, low recall  -> candidate for automatic decline")
    print("  high recall, low precision  -> review queue only, never a hard block")
    print("  alerts_per_fraud            -> legitimate customers bothered per fraud caught")

    print("\nCAVEAT — this script is DESCRIPTIVE, not operational:")
    print("  * R02 and R10 are built from chargeback outcomes, which in production arrive weeks")
    print("    after the transaction; their precision here assumes instant knowledge.")
    print("  * R10 also includes the current transaction in the merchant rate (transform mean).")
    print("  Run scripts/03_impact.py for the past-only replay and the operational numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
