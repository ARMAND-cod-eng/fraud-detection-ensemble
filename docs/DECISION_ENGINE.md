# Decision engine

The model answers *"how likely is this transaction to be fraud?"* The decision engine answers
*"what should we do about it, given costs and review capacity?"* Keeping them apart means thresholds
and cost assumptions change without retraining, and every decision is traceable to both the
`model_version` and the `policy_version` that produced it.

> **The policies here are illustrative.** The dataset contains transaction amounts but no
> chargeback, recovery, analyst-cost or customer-churn data. All costs below are stated
> assumptions, not validated financial parameters.

## Policy (configs/decision_policy.yaml)

```yaml
policy_id: example-balanced
review_threshold: model        # = the model's validation cost-optimal threshold (0.533)
decline_threshold: 0.90        # recommend decline at ≥ 90% calibrated probability
expected_loss_review: false    # optional: also review when p × amount × loss_rate ≥ review_cost
max_reviews_per_hour: 120      # capacity; excess alerts are still queued but flagged
cost_assumptions: { review_cost: 5.0, loss_rate: 1.0, false_decline_cost: 15.0 }
```

| Calibrated probability | Risk level | Recommended action |
|---|---|---|
| p ≥ decline threshold | high | decline (recommendation; analysts can override) |
| review threshold ≤ p < decline threshold | medium | manual review |
| optional: p × amount ≥ review cost | medium | manual review |
| otherwise | low | approve |

Thresholds operate on **calibrated** probabilities, so "decline at 90%" means what it says to a
business owner, and the amount-aware rule (expected loss = p × amount) is meaningful.

## Implementation

* `fraudshield.decision_engine.policy` — `PolicyConfig` (validated: decline > review), version =
  `policy_id@sha256(config)[:8]`, `DecisionEngine.decide(p, amount)` returns action, risk level,
  human-readable reasons, expected loss and a capacity flag.
* Capacity: a rolling one-hour window counts routed reviews. Above `max_reviews_per_hour` new
  reviews are still routed (never silently approved) and carry
  `review_capacity_exceeded = true` plus a reason, so the dashboard shows the team is overloaded.
* Decision log: each prediction row stores `action`, `risk_level`, `reasons`, `expected_loss`,
  `over_capacity`, `model_version`, `policy_version`; every policy change is appended to
  `policy_history` with actor and full config. `PUT /api/v1/policy` applies a new version live.
* What-if: `POST /api/v1/policy/simulate` costs any policy on the labelled held-out window
  (`fraudshield.decision_engine.simulate`), and the dashboard's *Decision policy* page drives it.

## Experiment: thresholds vs. operational cost

`python benchmarks/policy_experiment.py` → `benchmarks/results/policy_experiment.{md,json}`.
Thresholds are searched on **validation**; the frozen policies are costed once on **test**
(7.7 h, 56,746 transactions, 74 frauds worth €7,728).

| Policy | Review ≥ | Decline ≥ | Validation cost € | Test: reviewed / declined | Test: frauds stopped | Test: legit declined | Test cost € |
|---|---:|---:|---:|---:|---:|---:|---:|
| Approve everything | — | — | 12,231 | 0 / 0 | 0 / 74 | 0 | 7,728 |
| Example policy (above) | 0.533 | 0.90 | 4,570 | 15 / 40 | 52 / 74 | 0 | 3,716 |
| Review-only (never auto-decline) | 0.533 | — | 4,695 | 55 / 0 | 52 / 74 | 0 | 3,916 |
| Example + expected-loss rule | 0.533 | 0.90 | 4,910 | 73 / 40 | 54 / 74 | 0 | 2,773 |
| Two thresholds tuned on validation | 0.478 | 0.80 | 4,540 | 10 / 47 | 52 / 74 | 0 | 3,691 |

What this shows:

1. **The model halves the illustrative loss** (€7,728 → €3,716) with 55 alerts in 7.7 hours —
   about 7 per hour, well within one analyst's capacity.
2. **Auto-decline at ≥ 80–90% cost nothing in friction here**: no legitimate transaction scored
   that high in the test window. With 74 frauds this is weak evidence; a real deployment would
   monitor false declines explicitly.
3. **Validation and test disagree on the expected-loss rule.** It is the cheapest policy on test
   but the most expensive on validation, so a validation-driven choice would *not* adopt it.
   Picking it now would be tuning on test; it stays an opt-in.
4. **Cost-optimal thresholds are insensitive to review cost** (identical for €1–€50 per review on
   validation). Frauds the model misses have scores in the bulk of normal traffic, so cheaper
   reviews do not buy more catches.
5. **Many frauds are tiny.** Median test fraud €4.79; 53% are below the €5 review cost. A pure
   cost-minimising rule would never review them, yet small charges are a classic card-testing
   pattern that precedes larger fraud. This is why the default keeps a probability threshold and
   the amount-aware rule is additive, not a replacement.

### Fixed investigation budget (test window, top-scored alerts reviewed first)

| Reviews / hour | Reviews | Frauds caught | Recall | Precision |
|---:|---:|---:|---:|---:|
| 5 | 38 | 38 | 51.4% | 100.0% |
| 10 | 77 | 58 | 78.4% | 75.3% |
| 25 | 191 | 60 | 81.1% | 31.4% |
| 50 | 383 | 63 | 85.1% | 16.4% |
| 100 | 766 | 64 | 86.5% | 8.4% |

Going from 10 to 100 reviews per hour (10× analyst time) adds 6 frauds. A small team capped at
~10 reviews/hour captures most of the achievable value.

## Approval vs. rejection trade-off

* Declining a legitimate payment costs friction (assumed €15) and possibly the customer.
* Approving a fraud costs its amount (loss rate 100%, no recovery assumed).
* Reviewing costs analyst time (assumed €5) and delays the customer.

With calibrated probability p and amount A, the expected cost of each action is approve: p·A,
decline: (1−p)·€15, review: €5 (assuming reviews catch fraud). Decline beats approve when
p > 15 / (A + 15) — for a €100 payment at p > 13%. The example policy is deliberately more
conservative (decline only at ≥ 90%) because the friction assumption is the least certain one and
because declines should stay recommendations reviewed by people until false-decline costs are
measured.
