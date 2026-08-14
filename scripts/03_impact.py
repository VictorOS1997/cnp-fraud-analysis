"""Replay the base as a streaming decision engine and size the business impact.

Run from the project root, after 00_download_data.py:

    python scripts/03_impact.py

Why this script exists
----------------------
01_eda.py and 02_rules.py describe the data with columns computed over the whole period. That
is enough to *describe* fraud, but it is not how a rule runs in production: at decision time
only the past exists, and a chargeback is only known weeks after the transaction it disputes.

This script replays every transaction in chronological order, builds each feature from prior
transactions only, applies the dispute delay to every chargeback-derived feature (user *and*
merchant), and then measures a concrete decision policy: fraud stopped, money protected,
legitimate customers affected, and how the same policy behaves outside the demand peak and at a
realistic fraud prevalence.

Writes: outputs/policy_simulation.csv, outputs/figures/07_label_timing.png,
        outputs/figures/08_policy_tiers.png
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
CSV_PATH = ROOT / "data" / "raw" / "transactional-sample.csv"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# --- policy configuration (the recommended policy is built from these constants) ---
VELOCITY_WINDOW_H = 24
VELOCITY_REVIEW_PRIOR_TX = 1     # review from the user's 2nd transaction in the window
VELOCITY_DECLINE_PRIOR_TX = 2    # decline from the 3rd
CARDS_REVIEW = 2                 # review once a 2nd distinct card is seen
CARDS_DECLINE = 3                # decline at the 3rd
# NOTE: 10 is the *diagnostic* threshold kept for comparison with 02_rules.py. The recommended
# batch trigger is lower — 3 transactions and 2 known disputes — see the merchant grid below.
MERCHANT_MIN_TX = 10             # minimum merchant history before judging it
MERCHANT_RATE_MULTIPLE = 2.0     # merchant rate above 2x the running base rate
AMOUNT_STEPUP_Q = 0.90           # step-up above this quantile of past amounts
AMOUNT_MIN_HISTORY = 200         # transactions needed before the quantile is trusted

# --- label timing ---
DISPUTE_LAG_DAYS = 15            # assumed delay before a chargeback is known to the engine
LAG_GRID_DAYS = (0, 1, 3, 7, 15, 21)

# --- reporting ---
REGIME_CUT = pd.Timestamp("2019-11-21")   # demand peak starts here (see the regime section)
TARGET_PREVALENCE = 0.01                  # realistic production chargeback rate


def load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise SystemExit(f"missing {CSV_PATH} — run: python scripts/00_download_data.py")
    df = pd.read_csv(CSV_PATH)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["has_cbk"] = df["has_cbk"].astype(str).str.strip().str.upper().eq("TRUE")
    return df.sort_values("transaction_date").reset_index(drop=True)


# --------------------------------------------------------------- streaming replay


def replay(df: pd.DataFrame) -> pd.DataFrame:
    """One chronological pass. Every feature uses prior transactions only.

    Chargeback-derived features come in two flavours:
      *_cbk      : chargeback known the instant the disputed transaction happens (unrealistic,
                   kept only to show what a naive analysis would have measured)
      *_cbk_known: only chargebacks whose transaction is at least DISPUTE_LAG_DAYS old, which is
                   what the engine could actually know at decision time
    """
    user_times: dict[int, list[pd.Timestamp]] = defaultdict(list)
    user_cards: dict[int, set[str]] = defaultdict(set)
    user_cbk_times: dict[int, list[pd.Timestamp]] = defaultdict(list)
    merchant_times: dict[int, list[pd.Timestamp]] = defaultdict(list)
    merchant_cbk_times: dict[int, list[pd.Timestamp]] = defaultdict(list)
    past_amounts: list[float] = []

    rows = []
    running_tx = 0
    running_cbk = 0

    for r in df.itertuples(index=False):
        uid, mid, card, now = r.user_id, r.merchant_id, r.card_number, r.transaction_date
        known_cutoff = now - pd.Timedelta(days=DISPUTE_LAG_DAYS)

        # velocity: prior transactions of this user inside the window
        times = user_times[uid]
        prior_tx_window = len(times) - bisect_left(times, now - pd.Timedelta(hours=VELOCITY_WINDOW_H))

        # chargebacks of this user: ages in days of EVERY prior dispute, so any lag can be
        # simulated later (using only the most recent one would understate a block list)
        cbk_times = user_cbk_times[uid]
        user_cbk_ages = [(now - t).total_seconds() / 86400 for t in cbk_times]
        days_since_last_cbk = user_cbk_ages[-1] if user_cbk_ages else None

        # merchant history before this transaction, with and without the dispute delay
        m_times = merchant_times[mid]
        m_cbk_times = merchant_cbk_times[mid]
        m_tx = len(m_times)
        m_tx_known = bisect_left(m_times, known_cutoff)
        m_cbk_ages = [(now - t).total_seconds() / 86400 for t in m_cbk_times]
        m_age_h = (now - m_times[0]).total_seconds() / 3600 if m_times else 0.0
        m_rate = len(m_cbk_times) / m_tx if m_tx else 0.0
        m_cbk_known = bisect_left(m_cbk_times, known_cutoff)
        m_rate_known = m_cbk_known / m_tx_known if m_tx_known else 0.0

        # expanding amount quantile — past amounts only
        if len(past_amounts) >= AMOUNT_MIN_HISTORY:
            amount_threshold = past_amounts[int(AMOUNT_STEPUP_Q * len(past_amounts))]
        else:
            amount_threshold = float("inf")

        rows.append(
            {
                "transaction_id": r.transaction_id,
                "transaction_date": now,
                "user_id": uid,
                "merchant_id": mid,
                "card_number": card,
                "transaction_amount": r.transaction_amount,
                "has_device": pd.notna(r.device_id),
                "has_cbk": r.has_cbk,
                "user_prior_tx_24h": prior_tx_window,
                "user_prior_cards": len(user_cards[uid]),
                "user_prior_cbk": len(cbk_times),
                "user_prior_cbk_known": bisect_left(cbk_times, known_cutoff),
                "user_cbk_ages": user_cbk_ages,
                "days_since_last_cbk": days_since_last_cbk,
                "merchant_prior_tx": m_tx,
                "merchant_age_h": m_age_h,
                "merchant_prior_cbk": len(m_cbk_times),
                "merchant_cbk_ages": m_cbk_ages,
                "merchant_prior_cbk_rate": m_rate,
                "merchant_prior_tx_known": m_tx_known,
                "merchant_prior_cbk_rate_known": m_rate_known,
                "amount_threshold": amount_threshold,
                "running_base_rate": running_cbk / running_tx if running_tx else 0.0,
            }
        )

        # update state only after scoring
        times.append(now)
        user_cards[uid].add(card)
        m_times.append(now)
        insort(past_amounts, r.transaction_amount)
        running_tx += 1
        if r.has_cbk:
            cbk_times.append(now)
            m_cbk_times.append(now)
            running_cbk += 1

    return pd.DataFrame(rows)


def flag_prior_cbk(df: pd.DataFrame, lag_days: int) -> pd.Series:
    """Would a block list fire, if disputes took `lag_days` to reach the engine?

    Every prior dispute of the user is considered, not only the most recent one: an account
    with one mature chargeback and one from an hour ago is still on the list.
    """
    return df["user_cbk_ages"].apply(
        lambda ages: any(age >= lag_days for age in ages) if ages else False
    )


def merchant_action_impact(
    df: pd.DataFrame, min_tx: int, min_cbk: int, lag_days: int, action_delay_h: int = 0
) -> dict:
    """Prospective value of a merchant-monitoring batch.

    Two independent clocks, both parameterised:
      lag_days       — how long a dispute takes to reach the engine
      action_delay_h — how long the batch takes to act once the trigger is true
                       (a daily batch acts on the next cycle, not instantly)

    The value credited is only the loss occurring AFTER the merchant is actioned — never the
    whole month, which is knowable only in hindsight.
    """
    fires = (
        (df["merchant_prior_tx"] >= min_tx)
        & df["merchant_cbk_ages"].apply(
            lambda ages: sum(1 for age in ages if age >= lag_days) >= min_cbk
        )
    )
    total_fraud_amount = df.loc[df["has_cbk"], "transaction_amount"].sum()

    after = pd.Series(False, index=df.index)
    triggered_merchants = 0
    for mid, idx in df.loc[fires].groupby("merchant_id").groups.items():
        acts_at = df.loc[min(idx), "transaction_date"] + pd.Timedelta(hours=action_delay_h)
        triggered_merchants += 1
        after |= (df["merchant_id"] == mid) & (df["transaction_date"] >= acts_at)

    blocked = df.loc[after]
    fraud = blocked.loc[blocked["has_cbk"]]
    legit = blocked.loc[~blocked["has_cbk"]]
    return {
        "trigger": f"{min_tx}+ tx, {min_cbk}+ disputes, {lag_days}d label lag, "
                   f"{action_delay_h}h to act",
        "merchants_actioned": triggered_merchants,
        "transactions_stopped": len(blocked),
        "frauds_stopped": len(fraud),
        "fraud_amount_stopped": round(fraud["transaction_amount"].sum(), 2),
        "pct_of_disputed_volume": round(
            fraud["transaction_amount"].sum() / total_fraud_amount * 100, 1
        ),
        "legit_transactions": len(legit),
        "legit_amount": round(legit["transaction_amount"].sum(), 2),
    }


def volume_ramp_impact(df: pd.DataFrame, max_tx: int, window_h: int) -> dict:
    """A new-merchant volume ramp limit: no chargeback needed to trigger it.

    Blocks a transaction when the merchant is still inside its first `window_h` hours of
    observed activity and has already processed `max_tx` transactions.
    """
    mask = (df["merchant_age_h"] <= window_h) & (df["merchant_prior_tx"] >= max_tx)
    total_fraud_amount = df.loc[df["has_cbk"], "transaction_amount"].sum()
    blocked = df.loc[mask]
    fraud = blocked.loc[blocked["has_cbk"]]
    legit = blocked.loc[~blocked["has_cbk"]]
    return {
        "rule": f"max {max_tx} transactions in the merchant's first {window_h}h",
        "merchants_hit": blocked["merchant_id"].nunique(),
        "transactions_blocked": len(blocked),
        "frauds_blocked": len(fraud),
        "precision_pct": round(len(fraud) / len(blocked) * 100, 1) if len(blocked) else 0.0,
        "fraud_amount_blocked": round(fraud["transaction_amount"].sum(), 2),
        "pct_of_disputed_volume": round(
            fraud["transaction_amount"].sum() / total_fraud_amount * 100, 1
        ),
        "legit_transactions": len(legit),
        "legit_amount": round(legit["transaction_amount"].sum(), 2),
    }


# ------------------------------------------------------------------------ metrics


def measure(df: pd.DataFrame, mask: pd.Series, name: str, action: str) -> dict:
    total_fraud = int(df["has_cbk"].sum())
    total_fraud_amount = df.loc[df["has_cbk"], "transaction_amount"].sum()

    hit = df.loc[mask.fillna(False)]
    fraud = hit.loc[hit["has_cbk"]]
    legit = hit.loc[~hit["has_cbk"]]

    # precision if the same lift applied at a realistic prevalence: keep the frauds, scale the
    # legitimate transactions up to what they would be at TARGET_PREVALENCE
    p0 = df["has_cbk"].mean()
    weight = (p0 / (1 - p0)) * ((1 - TARGET_PREVALENCE) / TARGET_PREVALENCE)
    adj_precision = len(fraud) / (len(fraud) + len(legit) * weight) if len(hit) else 0.0

    return {
        "rule": name,
        "action": action,
        "transactions": len(hit),
        "share_pct": round(len(hit) / len(df) * 100, 2),
        "frauds": len(fraud),
        "recall_pct": round(len(fraud) / total_fraud * 100, 1) if total_fraud else 0.0,
        "precision_pct": round(len(fraud) / len(hit) * 100, 1) if len(hit) else 0.0,
        "precision_at_1pct_base": round(adj_precision * 100, 1),
        "fraud_amount": round(fraud["transaction_amount"].sum(), 2),
        "fraud_amount_pct": round(
            fraud["transaction_amount"].sum() / total_fraud_amount * 100, 1
        ) if total_fraud_amount else 0.0,
        "legit_transactions": len(legit),
        "legit_users": legit["user_id"].nunique(),
        "legit_amount": round(legit["transaction_amount"].sum(), 2),
        "legit_per_fraud": round(len(legit) / len(fraud), 2) if len(fraud) else float("inf"),
        "legit_per_fraud_at_1pct": round(
            len(legit) * weight / len(fraud), 1
        ) if len(fraud) else float("inf"),
    }


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------------- policy


def build_policy(df: pd.DataFrame) -> dict[str, pd.Series]:
    """The recommended four-outcome policy, built only from past-only features.

    Merchant risk is deliberately NOT here: with a realistic dispute delay it fires on a single
    transaction in this window, and merchant action belongs to a daily batch anyway.
    """
    risky_merchant = (
        (df["merchant_prior_tx_known"] >= MERCHANT_MIN_TX)
        & (df["merchant_prior_cbk_rate_known"] > df["running_base_rate"] * MERCHANT_RATE_MULTIPLE)
    )
    decline = (
        (df["user_prior_tx_24h"] >= VELOCITY_DECLINE_PRIOR_TX)
        | (df["user_prior_cards"] >= CARDS_DECLINE)
    )
    review = ~decline & (
        (df["user_prior_tx_24h"] >= VELOCITY_REVIEW_PRIOR_TX)
        | (df["user_prior_cards"] >= CARDS_REVIEW)
    )
    stepup = ~decline & ~review & (df["transaction_amount"] > df["amount_threshold"])
    approve = ~decline & ~review & ~stepup
    return {
        "decline": decline,
        "review": review,
        "stepup": stepup,
        "approve": approve,
        "risky_merchant": risky_merchant,
    }


# --------------------------------------------------------------------------- charts


def chart_07_label_timing(gaps: pd.Series, lag_table: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    within_1d = (gaps < 1).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(gaps.clip(upper=15), bins=30, color="#4c72b0")
    axes[0].set_xlabel("days between the charged-back transaction and the user's next one")
    axes[0].set_ylabel("transactions")
    axes[0].set_title(f"{within_1d:.0f}% of follow-ups happen within 24h of the fraud")

    axes[1].plot(lag_table["lag_days"], lag_table["recall_pct"], marker="o", color="#c44e52")
    axes[1].set_xlabel("assumed days until the dispute is known to the engine")
    axes[1].set_ylabel("% of all fraud the rule would catch")
    axes[1].set_title("The prior-chargeback rule decays with realistic dispute timing")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "'Prior chargeback' is a reporting signal, not a real-time one — the burst is over "
        "before the dispute exists",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_label_timing.png", dpi=140)
    plt.close(fig)
    print(f"  saved {FIG_DIR / '07_label_timing.png'}")


def chart_08_policy(tiers: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = tiers["rule"].str.replace("tier_", "", regex=False).tolist()
    colors = ["#c44e52", "#dd8452", "#937860", "#4c72b0"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, col, title in (
        (axes[0], "transactions", "Transactions by decision"),
        (axes[1], "frauds", "Fraud by decision"),
    ):
        values = tiers[col].tolist()
        ax.bar(labels, values, color=colors[: len(labels)])
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:,}\n({v / sum(values) * 100:.1f}%)", ha="center", va="bottom",
                    fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(col)

    fig.suptitle("Recommended policy, simulated with past-only features", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_policy_tiers.png", dpi=140)
    plt.close(fig)
    print(f"  saved {FIG_DIR / '08_policy_tiers.png'}")


# ----------------------------------------------------------------------------- main


def main() -> int:
    raw = load()
    df = replay(raw)

    base_rate = df["has_cbk"].mean()
    total_fraud = int(df["has_cbk"].sum())
    total_amount = df["transaction_amount"].sum()
    fraud_amount = df.loc[df["has_cbk"], "transaction_amount"].sum()
    days = (df["transaction_date"].max() - df["transaction_date"].min()).days or 1

    section("BASE")
    print(f"transactions    : {len(df):,}   over {days} days")
    print(f"period          : {df['transaction_date'].min()} .. {df['transaction_date'].max()}")
    print(f"chargebacks     : {total_fraud:,} ({base_rate * 100:.2f}%)")
    print(f"volume          : {total_amount:,.2f}")
    print(f"disputed volume : {fraud_amount:,.2f} ({fraud_amount / total_amount * 100:.2f}%)")

    # ------------------------------------------------------------------- regimes
    section("TWO REGIMES — the month is not homogeneous")
    early = df["transaction_date"] < REGIME_CUT
    print(f"01-20 Nov      : {int(early.sum()):,} transactions, "
          f"{df.loc[early, 'has_cbk'].mean() * 100:.1f}% chargeback")
    print(f"21 Nov - 01 Dec: {int((~early).sum()):,} transactions, "
          f"{df.loc[~early, 'has_cbk'].mean() * 100:.1f}% chargeback")
    print(f"the last 11 days carry {(~early).mean() * 100:.0f}% of the volume and "
          f"{df.loc[~early, 'has_cbk'].sum() / total_fraud * 100:.0f}% of the fraud "
          f"(Black Friday week)")
    daily = df.groupby(df["transaction_date"].dt.date)["has_cbk"].agg(tx="size", rate="mean")
    print("\nlast 8 days:")
    print(daily.tail(8).assign(rate=lambda d: (d["rate"] * 100).round(1)).to_string())

    # ------------------------------------------------------------- concentration
    section("CONCENTRATION — where the loss actually sits")
    users = df.groupby("user_id").agg(
        tx=("transaction_id", "size"),
        frauds=("has_cbk", "sum"),
        fraud_amount=("transaction_amount", lambda s: s[df.loc[s.index, "has_cbk"]].sum()),
    )
    fraud_users = users.loc[users["frauds"] > 0]
    top_users = users.sort_values("fraud_amount", ascending=False).head(10)
    print(f"users with a chargeback : {len(fraud_users):,} of {len(users):,} "
          f"({len(fraud_users) / len(users) * 100:.1f}% of users, "
          f"{fraud_users['tx'].sum() / len(df) * 100:.1f}% of transactions)")
    print(f"top 10 users            : {top_users['fraud_amount'].sum():,.2f} disputed "
          f"({top_users['fraud_amount'].sum() / fraud_amount * 100:.1f}% of all disputed volume)")

    merchants = df.groupby("merchant_id").agg(
        tx=("transaction_id", "size"),
        frauds=("has_cbk", "sum"),
        fraud_amount=("transaction_amount", lambda s: s[df.loc[s.index, "has_cbk"]].sum()),
        legit_amount=("transaction_amount", lambda s: s[~df.loc[s.index, "has_cbk"]].sum()),
    )
    top_m = merchants.sort_values("fraud_amount", ascending=False).head(10)
    print(f"\nmerchants               : {len(merchants):,} "
          f"({int((merchants['tx'] == 1).sum()):,} with a single transaction, "
          f"{int((merchants['tx'] >= MERCHANT_MIN_TX).sum())} with {MERCHANT_MIN_TX}+)")
    print(f"top 10 merchants        : {int(top_m['tx'].sum())} transactions, "
          f"{int(top_m['frauds'].sum())} frauds "
          f"({top_m['frauds'].sum() / top_m['tx'].sum() * 100:.1f}% chargeback rate)")
    print(f"  disputed volume       : {top_m['fraud_amount'].sum():,.2f} "
          f"({top_m['fraud_amount'].sum() / fraud_amount * 100:.1f}% of the total)")
    print(f"  legitimate at risk    : {top_m['legit_amount'].sum():,.2f} "
          f"— the collateral damage of acting on these merchants")

    # ---------------------------------------------------------------- label lag
    section("LABEL TIMING — can a chargeback-based rule run in real time?")
    after = df.loc[df["user_prior_cbk"] > 0]
    gaps = after["days_since_last_cbk"].dropna()
    print(f"transactions after a prior chargeback : {len(after):,}, "
          f"{after['has_cbk'].mean() * 100:.1f}% fraudulent (base {base_rate * 100:.2f}%)")
    print(f"  median gap to that chargeback       : {gaps.median():.2f} days")
    print(f"  within 24h                          : {(gaps < 1).mean() * 100:.1f}%")

    lag_rows = []
    for lag in LAG_GRID_DAYS:
        m = flag_prior_cbk(df, lag)
        res = measure(df, m, f"prior_chargeback_lag_{lag}d", "diagnostic")
        res["lag_days"] = lag
        lag_rows.append(res)
    lag_table = pd.DataFrame(lag_rows)
    print("\nsame rule, varying how long the dispute takes to reach the engine:")
    print(lag_table[["lag_days", "transactions", "frauds", "recall_pct", "precision_pct"]]
          .to_string(index=False))
    print("\nThe conclusion is robust to the assumption: at any realistic dispute delay the rule")
    print("misses the burst it is supposed to stop. It remains valid as a *block list* built")
    print("from confirmed fraud — it is simply not the rule that stops this attack pattern.")

    # the merchant rule gets the same test — it is also chargeback-derived
    merch_now = (
        (df["merchant_prior_tx"] >= MERCHANT_MIN_TX)
        & (df["merchant_prior_cbk_rate"] > df["running_base_rate"] * MERCHANT_RATE_MULTIPLE)
    )
    merch_known = (
        (df["merchant_prior_tx_known"] >= MERCHANT_MIN_TX)
        & (df["merchant_prior_cbk_rate_known"] > df["running_base_rate"] * MERCHANT_RATE_MULTIPLE)
    )
    print("\nmerchant-history rule under the same scrutiny:")
    print(pd.DataFrame([
        measure(df, merch_now, "merchant_instant_knowledge", "diagnostic"),
        measure(df, merch_known, f"merchant_{DISPUTE_LAG_DAYS}d_lag", "review"),
    ])[["rule", "transactions", "frauds", "recall_pct", "precision_pct"]].to_string(index=False))
    print(f"Only {int((merchants['tx'] >= MERCHANT_MIN_TX).sum())} of {len(merchants):,} merchants "
          f"ever reach {MERCHANT_MIN_TX} transactions in this window, so within 30 days the rule")
    print("is structurally weak. In production a merchant carries months of history and the same")
    print("rule is strong — that is a production premise, not something this sample proves.")

    # ------------------------------------------------- merchant action, prospectively
    section("MERCHANT MONITORING — value measured AFTER the trigger, not in hindsight")
    print("The top-10 concentration above is an autopsy. What a daily batch would actually save")
    print("is the loss that happens after it fires, so that is what is measured here.\n")
    combos = [
        (MERCHANT_MIN_TX, 1, 0, 0), (MERCHANT_MIN_TX, 1, 3, 24),
        (3, 2, 0, 0), (3, 2, 0, 24), (3, 2, 3, 24), (3, 2, DISPUTE_LAG_DAYS, 24),
        (2, 1, 0, 0), (2, 1, 0, 24), (2, 1, 3, 24),
    ]
    merchant_grid = pd.DataFrame(
        [merchant_action_impact(df, mt, mc, lag, delay) for mt, mc, lag, delay in combos]
    )
    print(merchant_grid.to_string(index=False))
    print("\nTwo clocks matter: how long the dispute takes to arrive, and how long the batch takes")
    print("to act. A daily batch acts on the next cycle — the 24h rows are the honest ones.")

    # --------------------------------------------- does a short lifespan discriminate?
    section("SHORT-LIVED MERCHANTS — testing the intuition against a base rate")
    spans = raw.groupby("merchant_id")["transaction_date"].agg(["min", "max", "size"])
    spans["life_days"] = (spans["max"] - spans["min"]).dt.total_seconds() / 86400
    spans["frauds"] = raw.groupby("merchant_id")["has_cbk"].sum()
    spans["fraud_amount"] = merchants["fraud_amount"]

    dirty = spans.loc[spans["frauds"] > 0]
    clean = spans.loc[spans["frauds"] == 0]
    short_dirty = dirty.loc[dirty["life_days"] <= 2]
    short_clean = clean.loc[clean["life_days"] <= 2]
    print(f"merchants with a chargeback living <=2 days   : {len(short_dirty)} of {len(dirty)} "
          f"({len(short_dirty) / len(dirty) * 100:.0f}%), {short_dirty['fraud_amount'].sum():,.2f} "
          f"disputed ({short_dirty['fraud_amount'].sum() / fraud_amount * 100:.1f}%)")
    print(f"merchants with NO chargeback living <=2 days  : {len(short_clean)} of {len(clean)} "
          f"({len(short_clean) / len(clean) * 100:.0f}%)")
    print(f"of the short-lived merchants with fraud, "
          f"{int((short_dirty['size'] == 1).sum())} have a single transaction and "
          f"{int((dirty.loc[dirty['life_days'] <= 2, 'max'] >= df['transaction_date'].max() - pd.Timedelta(days=2)).sum())} "
          f"are still active in the last two days of the window (right-censored).")
    print("\nSo lifespan does NOT discriminate: most legitimate merchants here are also short-lived,")
    print("because 70.7% of the portfolio has a single transaction and the window is 30 days long.")
    print("Without an onboarding date this variable cannot be measured. What DOES discriminate is")
    print("volume compressed into the merchant's first hours — measured next.")

    # ---------------------------------------------------- new-merchant volume ramp
    section("NEW-MERCHANT VOLUME RAMP — a control that needs no chargeback at all")
    ramp_grid = pd.DataFrame([
        volume_ramp_impact(df, 3, 48),
        volume_ramp_impact(df, 5, 48),
        volume_ramp_impact(df, 5, 24),
        volume_ramp_impact(df, 10, 48),
    ])
    print(ramp_grid.to_string(index=False))
    print("\nThis is the only merchant control that fires before any dispute exists, which is why")
    print("it reaches the burst-and-vanish profile that monitoring cannot.")
    print("Caveat: without an onboarding date, 'first hours' means first *observed* transaction —")
    print("an old, low-volume merchant can enter the cut. This is a ceiling, not a measurement,")
    print("and one more reason to bring merchant onboarding data into the base.")

    # ------------------------------------------------------------ real-time rules
    section("REAL-TIME RULES — no chargeback knowledge required")
    variants = []
    for k in (1, 2, 3):
        variants.append((f"velocity_{k}+_prior_tx_24h", "decline", df["user_prior_tx_24h"] >= k))
    for k in (2, 3):
        variants.append((f"cards_{k}+_distinct", "decline", df["user_prior_cards"] >= k))
    variants.append(
        ("amount_above_expanding_p90", "step-up",
         df["transaction_amount"] > df["amount_threshold"])
    )
    variants.append(("no_device", "score only", ~df["has_device"]))
    grid = pd.DataFrame([measure(df, m, n, a) for n, a, m in variants])
    print(grid[["rule", "action", "transactions", "frauds", "recall_pct", "precision_pct",
                "precision_at_1pct_base", "legit_transactions", "legit_users"]]
          .to_string(index=False))

    # overlap between the two headline rules
    vel = df["user_prior_tx_24h"] >= VELOCITY_DECLINE_PRIOR_TX
    cards = df["user_prior_cards"] >= CARDS_DECLINE
    print(f"\noverlap: velocity fires on {int(vel.sum())}, cards on {int(cards.sum())}, "
          f"both on {int((vel & cards).sum())} "
          f"({int((vel & cards & df['has_cbk']).sum())} of them fraud) — "
          f"union is {int((vel | cards).sum())}, not the sum.")

    # ------------------------------------------------------------------- policy
    section("RECOMMENDED POLICY — decline / review / step-up / approve")
    pol = build_policy(df)
    tiers = pd.DataFrame([
        measure(df, pol["decline"], "tier_decline", "decline"),
        measure(df, pol["review"], "tier_review", "manual review"),
        measure(df, pol["stepup"], "tier_stepup", "3DS step-up"),
        measure(df, pol["approve"], "tier_approve", "approve"),
    ])
    print(tiers[["rule", "transactions", "share_pct", "frauds", "recall_pct", "precision_pct",
                 "precision_at_1pct_base", "fraud_amount", "legit_transactions", "legit_users",
                 "legit_amount"]].to_string(index=False))

    d = pol["decline"]
    d_amount = df.loc[d & df["has_cbk"], "transaction_amount"].sum()
    d_legit_amount = df.loc[d & ~df["has_cbk"], "transaction_amount"].sum()
    print(f"\ndecline  : {int(d.sum()):,} transactions ({d.mean() * 100:.1f}%), "
          f"{int(df.loc[d, 'has_cbk'].sum())} frauds stopped, {d_amount:,.2f} protected")
    print(f"           collateral: {int((d & ~df['has_cbk']).sum())} legitimate transactions from "
          f"{df.loc[d & ~df['has_cbk'], 'user_id'].nunique()} distinct customers "
          f"({d_legit_amount:,.2f}), {d_legit_amount / d_amount:.2f} lost per 1.00 protected")
    print(f"review   : {int(pol['review'].sum()):,} transactions "
          f"(~{pol['review'].sum() / days:.0f}/day, {pol['review'].mean() * 100:.1f}% of the flow)")
    print(f"step-up  : {int(pol['stepup'].sum()):,} transactions — automatic 3DS, no analyst cost")

    # breakeven of the human queue: no analyst-accuracy data exists, so state the threshold
    q_fraud_amt = df.loc[pol["review"] & df["has_cbk"], "transaction_amount"].sum()
    q_legit_amt = df.loc[pol["review"] & ~df["has_cbk"], "transaction_amount"].sum()
    breakeven = q_legit_amt / (q_fraud_amt + q_legit_amt)
    print(f"\nreview queue breakeven: it holds {q_fraud_amt:,.2f} of fraud and {q_legit_amt:,.2f} "
          f"of legitimate value.")
    print(f"  Assuming an analyst is equally accurate on both sides, the queue is value-positive "
          f"above {breakeven * 100:.0f}% accuracy.")
    print("  (No analyst-outcome data exists in this sample — this is arithmetic on a stated "
          "premise, not a measurement.)")
    stopped = df.loc[d, "has_cbk"].sum()
    routed = df.loc[pol["review"] | pol["stepup"], "has_cbk"].sum()
    print(f"\nfraud STOPPED outright        : {int(stopped)}/{total_fraud} "
          f"({stopped / total_fraud * 100:.1f}%)")
    print(f"fraud ROUTED for a decision   : {int(routed)}/{total_fraud} "
          f"({routed / total_fraud * 100:.1f}%) — depends on analyst accuracy / 3DS outcome")
    print(f"fraud APPROVED (residual)     : {int(df.loc[pol['approve'], 'has_cbk'].sum())}/"
          f"{total_fraud} ({df.loc[pol['approve'], 'has_cbk'].mean() * 100:.1f}% of that tier)")

    # ------------------------------------------------------- policy by regime
    section("THE SAME POLICY, MEASURED IN EACH REGIME")
    regime_rows = []
    for label, mask_regime in (("01-20 Nov (normal)", early), ("21 Nov-01 Dec (peak)", ~early)):
        sub = df.loc[mask_regime].copy()
        sub_pol = build_policy(sub)
        for tier in ("decline", "review"):
            row = measure(sub, sub_pol[tier], f"{label} — {tier}", tier)
            regime_rows.append(row)
    regimes = pd.DataFrame(regime_rows)
    print(regimes[["rule", "transactions", "frauds", "recall_pct", "precision_pct",
                   "legit_transactions"]].to_string(index=False))
    print("\nOutside the peak the decline tier is less precise and covers less fraud. The headline")
    print("numbers above are the blended month — quote both when presenting.")

    # ------------------------------------------------------- prevalence sensitivity
    section(f"SENSITIVITY — what happens at a realistic {TARGET_PREVALENCE * 100:.0f}% base rate")
    p0 = base_rate
    weight = (p0 / (1 - p0)) * ((1 - TARGET_PREVALENCE) / TARGET_PREVALENCE)
    print(f"this sample is fraud-enriched: {p0 * 100:.2f}% vs ~{TARGET_PREVALENCE * 100:.0f}% in "
          f"production, so legitimate volume is under-represented by ~{weight:.1f}x")
    print(f"decline tier precision      : {tiers.loc[0, 'precision_pct']:.1f}% here  ->  "
          f"{tiers.loc[0, 'precision_at_1pct_base']:.1f}% at {TARGET_PREVALENCE * 100:.0f}%")
    print(f"legitimate blocked per fraud: {tiers.loc[0, 'legit_per_fraud']:.2f} here  ->  "
          f"{tiers.loc[0, 'legit_per_fraud_at_1pct']:.1f} at {TARGET_PREVALENCE * 100:.0f}%")
    money_ratio = d_legit_amount / d_amount
    print(f"legitimate volume lost per 1.00 protected: {money_ratio:.2f} here  ->  "
          f"~{money_ratio * weight:.1f} at {TARGET_PREVALENCE * 100:.0f}%")
    print("The rule still beats the base rate by a wide margin, but the operating point must be")
    print("re-tuned on production data before any auto-decline is switched on.")
    print("Premise: the missing legitimate volume behaves like the observed one. How this sample")
    print("was enriched is unknown, so treat this as a scenario, not a measurement.")

    # ------------------------------------------------------------- adversary note
    section("ADVERSARY — how easily this policy is evaded")
    per_user = raw.groupby("user_id")["merchant_id"].nunique()
    print(f"users transacting at more than one merchant: {int((per_user > 1).sum())} "
          f"({(per_user > 1).mean() * 100:.1f}%)")
    print(f"average transactions per user             : {raw.groupby('user_id').size().mean():.2f}")
    print("`user_id` behaves like a merchant-side customer id, and the attacker controls it at")
    print("sign-up. Every rule keyed on it is one new account away from being bypassed — which is")
    print("exactly the profile of the residual fraud in the approve tier. The policy must also be")
    print("keyed on card, device and BIN, and the additional data in the report is what makes")
    print("that possible.")

    # -------------------------------------------------------------------- amounts
    section("AMOUNT BANDS")
    bands = pd.qcut(df["transaction_amount"], 10, duplicates="drop")
    amt = df.groupby(bands, observed=True)["has_cbk"].agg(transactions="size", rate="mean")
    print(amt.assign(rate=lambda d: (d["rate"] * 100).round(1)).to_string())

    # --------------------------------------------------------------------- output
    section("CHARTS")
    chart_07_label_timing(gaps, lag_table)
    chart_08_policy(tiers)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([grid, tiers, regimes, lag_table.drop(columns=["lag_days"])]).to_csv(
        OUT_DIR / "policy_simulation.csv", index=False
    )
    print(f"\nwrote {OUT_DIR / 'policy_simulation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
