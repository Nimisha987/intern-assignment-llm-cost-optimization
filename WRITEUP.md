# Cost optimization write-up

Billing: `points = uncached_input × 3 + cached_input × 0.3 + output × 15`

## Results (public tickets, measured with `python breakdown.py`)

| | T-1001 | T-1002 | T-1003 | T-1004 | Target |
|---|---|---|---|---|---|
| Question 1 | 1,364 | 1,235 | 1,196 | 1,157 | ≤ 3,000 |
| Question 2 | 1,608 | 1,476 | 1,416 | 1,377 | ≤ 2,500 |

`action` and `kb_article` match the reference for T-1001 to T-1004 and for the T-1091 holdout.
The baselines were above 15,000 (Q1, four tickets) and 70,000 (Q2, one ticket) points,
which is what `test_baseline_cost_is_high` asserts.

## Question 1: points breakdown, T-1001 (physical item, refund)

One model call.

| Part | Tokens | Rate | Points |
|---|---|---|---|
| Cached prefix (guide, examples, help center, instructions) | 1,139 | 0.3 | 342 |
| Uncached input (thread + order snapshot) | 111 | 3 | 333 |
| Output | 46 | 15 | 690 |
| **Total** | | | **1,364** |

## Question 2: points breakdown for two tickets

One model call per ticket (compose). Triage, retrieve and audit are code and cost 0 points.

**Physical fulfillment path: T-1004 (damaged catalog item, replacement)**

| Part | Tokens | Rate | Points |
|---|---|---|---|
| Cached prefix | 1,071 | 0.3 | 321 |
| Uncached input (2 candidate articles + thread + order snapshot) | 172 | 3 | 516 |
| Output | 36 | 15 | 540 |
| **Total** | | | **1,377** |

**Escalation path: T-1003 (customized item, damage reported)**

| Part | Tokens | Rate | Points |
|---|---|---|---|
| Cached prefix | 1,071 | 0.3 | 321 |
| Uncached input | 170 | 3 | 510 |
| Output | 39 | 15 | 585 |
| **Total** | | | **1,416** |

## What was expensive in the baselines

- Q1: a `uuid` at the start of the system prompt changed every call, so no prefix could be
  cached. The prompt also asked for step-by-step reasoning and a customer reply, and the
  order snapshot was pretty-printed.
- Q2: four model calls per ticket. The padded help center (about 210k tokens) was sent
  repeatedly, retrieve asked the model to print full articles, compose repeated the triage
  text, and audit resent everything again.

## Changes

**Question 1**
- The system prompt is now identical on every call and passed to `chat()` as
  `cached_prefix`, so about 1,100 tokens are billed at 0.3 instead of 3.
- A policy guide (action meanings, snapshot field glossary, routing order, examples) lives in
  that prefix. It helps on unseen tickets and brings the prefix above the 1,024-token cache
  minimum.
- The reasoning request, run id and `indent=2` are gone. The model is asked for one line of
  compact JSON.

**Question 2**
- Triage (code) builds a query from the thread and the order facts.
- Retrieve (code) ranks the real articles by term overlap and keeps the top 2. The padded
  help center is never sent.
- Compose is the only model call: cached static prefix, then the two candidate articles,
  the thread and the order snapshot.
- Audit (code) checks that the action is valid and the cited article exists, and falls back to
  `ESCALATE_TIER2` with a retrieved article otherwise. No second model call.

## Trade-offs and limits

- `customer_reply` is no longer produced. It is not a graded field and it was the largest
  output cost.
- Output (about 35 to 45 tokens, roughly 540 to 690 points per ticket) is fixed by the
  simulator, so it is the floor for both questions.
- Retrieval is keyword based. It is enough for the five real articles, but a much larger
  help center would need a better ranker.
- Caching depends on the prefix staying at or above 1,024 tokens. The Q2 prefix clears it by a
  small margin (1,071), so a different tokenizer could change that.
- `eval/reference.py` is not imported and nothing is hardcoded to ticket ids.
- `test_baseline_cost_is_high` now fails in both questions, as expected: it asserts the cost
  is high, and the optimized agents are not.
