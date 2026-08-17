# Data dictionary

Every column of both datasets: what it means, how it is computed, its measured profile, and —
most importantly — **whether it can be used in a real-time rule**.

## Where to read it

| Format | Link |
|---|---|
| **Spreadsheet (easiest to browse, filter and sort)** | **[financial_data on Google Sheets](https://docs.google.com/spreadsheets/d/1RAcqoFj2LyJE-MhhAPq-hg_ulXXaRvvGzY4wbElGMnE/edit?gid=1689652905#gid=1689652905)** |
| CSV — raw dataset (8 columns) | [`outputs/data_dictionary_raw.csv`](../outputs/data_dictionary_raw.csv) |
| CSV — processed dataset (23 columns) | [`outputs/data_dictionary_processed.csv`](../outputs/data_dictionary_processed.csv) |
| Generator | [`scripts/04_data_dictionary.py`](../scripts/04_data_dictionary.py) |

The CSVs are generated, not maintained by hand: descriptions, formulas and pitfalls live in the
script, while the profile columns (type, nulls, distinct values, range, example) are measured
from the current files every time it runs.

## Fields

`dataset · source_file · position · column · role · unit · time_safety · description · formula ·
dtype · non_null · null_pct · distinct · min · max · example · pitfall`

## `time_safety` — the field that decides what you may build

A feature is only usable in an authorization rule if it is knowable **at the moment of the
transaction**. This classification is the reason the report separates descriptive findings from
operational ones.

| Value | Meaning | Usable in a real-time rule? |
|---|---|---|
| `raw` | Present in the source file, known at authorization time | **Yes** |
| `past-only` | Derived, uses only information available before the transaction | **Yes** |
| `whole-period` | Aggregated over the entire dataset — it sees the future | No — descriptive only |
| `lagging-label` | Derived from chargebacks, which arrive days to weeks later | No — belongs in block lists and batch jobs |
| `target` | The label itself | Never an input |

## The two datasets

**Raw** — `data/raw/transactional-sample.csv`, 3,199 rows, 8 columns. Downloaded by
`scripts/00_download_data.py`; not versioned in this repository.

**Processed** — `data/processed/transactions_enriched.csv`, 23 columns: the original 8 plus 15
derived features, produced by `scripts/01_eda.py`.

Of the 15 derived columns, 7 are `past-only`, 6 are `whole-period` and 2 are `lagging-label`.
That split is worth reading before reusing any of them: the two most predictive-looking
columns in the file (`user_cbk_before` and `is_repeat_offender`) are precisely the ones that
cannot run in production.

## Note on the streaming features

`scripts/03_impact.py` builds a further set of features inside its chronological replay
(`user_prior_tx_24h`, `user_prior_cards`, `merchant_prior_tx`, `merchant_age_h`,
`user_cbk_ages`, …). Those are all past-only by construction and are documented in the script
itself; they are not persisted to disk, so they do not appear in these dictionaries.
