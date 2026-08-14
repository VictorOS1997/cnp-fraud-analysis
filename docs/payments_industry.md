# Payments industry context

Four questions on how the card payments industry works, and why each answer matters for the
fraud analysis in the main report.

---

## 1. Money flow, information flow, and the main players

**The players.**

| Player | Role | Revenue |
|---|---|---|
| Cardholder | pays with the card | — |
| Issuer | the cardholder's bank; approves or declines, bills the cardholder | interchange, annual fees, interest |
| Card scheme (Visa, Mastercard, Elo) | the network connecting issuer and acquirer; writes the rules and arbitrates disputes | scheme fees |
| Acquirer | underwrites the merchant, processes and settles the transaction, carries the risk | MDR |
| Merchant | sells | the sale |
| Gateway | technical layer between checkout and acquirer | per-transaction or subscription fee |
| Sub-acquirer | aggregates small merchants under its own merchant account | spread on the MDR |

**Two flows, opposite directions, very different speeds.**

```
INFORMATION (sub-second, authorization)
  Merchant -> Gateway -> Acquirer -> Scheme -> Issuer
                                                 | (limit, balance, risk)
  Merchant <- Gateway <- Acquirer <- Scheme <----+
                     approved / declined

MONEY (days, clearing and settlement)
  Cardholder -> Issuer -> Scheme -> Acquirer -> Merchant
                  |          |          |
             interchange  scheme fee   MDR margin
```

The transaction has three distinct stages that are easy to confuse: **authorization** (the
issuer reserves the limit — no money moves), **clearing** (the merchant confirms the sale and
the batch is sent for settlement), and **settlement** (funds actually change hands, net of
fees; in Brazil typically D+1 for debit and D+30 for credit unless the merchant pays for early
settlement).

**Why it matters here.** Between authorization and settlement there is a window in which the
acquirer still controls the money. Detecting fraud inside that window is far cheaper than
detecting it afterwards — the transaction can be left uncaptured or refunded before it ever
becomes a chargeback.

---

## 2. Acquirer vs sub-acquirer vs gateway

The difference comes down to three questions: **does it hold a scheme licence, does it move
the money, and does it carry the risk?**

| | Acquirer | Sub-acquirer | Gateway |
|---|---|---|---|
| Scheme licence | yes | no (operates under an acquirer's) | no |
| Moves funds | yes | yes (receives and pays out) | no |
| Carries chargeback risk | yes | yes, towards the acquirer | no |
| Underwrites merchants | yes | yes, its own sellers | no |
| Sells | processing, settlement, risk | fast onboarding for small sellers | connectivity, routing, tokenisation |

**How the flow changes.** A **gateway** adds one hop to the information leg and nothing to the
money leg — funds never touch it. A **sub-acquirer** changes the money leg: the end merchant is
invisible to the scheme and to the acquirer, appears inside the sub-acquirer's own merchant
account, and is paid by the sub-acquirer rather than by the acquirer.

```
direct        Merchant ---------------> Acquirer -> Scheme -> Issuer
with gateway  Merchant -> Gateway -----> Acquirer -> Scheme -> Issuer      (money: unchanged)
sub-acquirer  Seller ---> Sub-acquirer -> Acquirer -> Scheme -> Issuer
              Seller <--- Sub-acquirer <- Acquirer                          (money: extra hop)
```

**Why it matters here.** Aggregation hides the end merchant. A single `merchant_id`
concentrating chargebacks may not be one shop at all — it may be an aggregator whose sellers
were never properly underwritten. This is the practical reason merchant-level monitoring is one
of the recommendations in the report.

---

## 3. Chargebacks, cancellations, and their link to fraud

A **chargeback** is a forced reversal. The cardholder disputes a transaction with the issuer,
which pulls the funds back from the acquirer under scheme rules. Every dispute carries a
**reason code**, and reason codes fall into four families: fraud, service (goods not received,
not as described), processing (duplicate or wrong amount), and authorization. There is a defined
dispute cycle — the merchant can accept the loss or contest it (representment), and unresolved
cases escalate to pre-arbitration and arbitration, decided by the scheme.

A **cancellation or refund** is the opposite in nature: the merchant voluntarily returns the
money. No dispute, no scheme fee, and no impact on the monitored chargeback ratio.

| | Chargeback | Refund |
|---|---|---|
| Initiated by | cardholder, through the issuer | merchant |
| Voluntary | no | yes |
| Dispute process | yes | no |
| Extra cost | dispute fee plus handling | none beyond the amount |
| Counts towards scheme monitoring | yes | no |

**The link with fraud, from the acquirer's seat.** Card-not-present fraud almost always
resurfaces as a fraud-coded chargeback weeks later, which is exactly why `has_cbk` is used as
the fraud label in this analysis — an imperfect and **lagging** one. Two consequences drive the
whole report: measured fraud is a floor rather than the truth, and any rule that depends on
knowing a chargeback cannot run at authorization time. Financially, the merchant pays first, but
when the merchant is insolvent or fraudulent the **acquirer absorbs the loss** — and schemes
penalise acquirers and merchants whose chargeback ratios breach programme thresholds. Strong
authentication (3DS) can shift liability for fraud disputes to the issuer, which is why
risk-based step-up is a cheaper answer than a hard decline.

---

## 4. What an anti-fraud system is, and how an acquirer uses it

An **anti-fraud system** is the combination of data, lists, rules and models that decides, in
real time, whether a transaction is approved, sent to review, or declined.

```
DATA      transaction + entity history + device + network + KYC
LISTS     block / allow (card, device, user, merchant)
RULES     explainable thresholds, changeable in minutes
MODEL     score for the combinations rules cannot express
DECISION  approve · step-up / review · decline
              ^                                  |
              +---- feedback: chargebacks and analyst outcomes
```

Rules and models are complementary, not alternatives: rules are auditable and can be explained
to a merchant who complains, models capture combinations of weak signals. Rules first — that is
the order used in this analysis.

**An acquirer uses it at three moments.**

1. **Merchant underwriting.** The most expensive fraud for an acquirer is not a stolen card: it
   is a fake merchant that processes volume, gets settled, and disappears, leaving the
   chargebacks behind.
2. **Real-time decisioning.** Each transaction is scored within a tight latency budget. The
   policy can also step up to 3DS rather than declining outright — shifting liability instead of
   losing the sale.
3. **Post-transaction monitoring.** Chargeback ratios per merchant, behavioural drift, incoming
   disputes fed back into rules and models, and preventive refunds for transactions identified
   as fraud after approval.

**How it is judged.** Not by how much it blocks. A system that declines everything has zero
fraud and no business. The pair of metrics that matters is **approval rate and fraud loss,
reported together** — which is why every recommendation in the main report is quoted with both
the fraud it prevents and the legitimate volume it costs.
