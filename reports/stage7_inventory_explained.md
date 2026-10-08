# Stage 7: From forecast to reorder decision (plain-English version)

Run: `python -m src.inventory` (raw output: `inventory_summary.txt`).

## The idea in one paragraph
A forecast is a guess. Stock exactly the guess and you run out about half the time.
**Safety stock** is the extra cushion that covers forecast error. **Reorder point**
= expected demand while you wait for the delivery (**lead time**) + safety stock.
Better forecast = smaller errors = smaller cushion for the same protection. That is
the business value of Stages 5-6.

## Assumptions (my defaults, all changeable at the top of `src/inventory.py`)
- Lead time = 7 days; orders placed weekly, so each week's stock must cover that
  week's demand. Reorder point for the week = 7-day forecast + safety stock
  (also the "order-up-to" level: you order enough to reach it).
- Target **service level** 95% = "no stockout in 95% of weeks". Service level maps to
  a multiplier z (z = 1.645 for 95%, from the normal distribution).
- **Safety stock = z x typical weekly forecast error.** "Typical error" is measured on
  Jul-Sep 2017 (validation: forecasts from a model that had never seen those
  days), so the test answers never influence the stock levels.
- Simulated 13 full test weeks (Oct 1 - Dec 30, 2017) x 500 store-items = 6,500 weekly order
  decisions per model. Simplified: each week starts fully stocked; no
  carry-over, no ordering costs, no minimum order sizes, no supplier delays.
- Compared: "same as last week" baseline vs XGBoost (Prophet skipped, as it is not a
  like-for-like horizon).

## Results (real)
**A. Each model sizes its own safety stock for 95%**
| | Baseline | XGBoost |
|---|---|---|
| Weeks with stockout | 3.02% | 4.68% |
| Avg safety stock per store-item-week | 56.20 | 38.03 (-32%) |
| Avg stock left at week end | 66.05 | 38.14 (-42%) |
| Fill rate (share of demand served) | 99.91% | 99.89% |
Both beat the 95% target. XGBoost needs about a third less cushion and carries about 42% less
leftover stock for nearly the same service. The baseline overshoots (97%) because its
cushion is oversized.

**C. Same safety stock for both (the baseline's amount): pure forecast-quality effect**
| | Baseline | XGBoost |
|---|---|---|
| Weeks with stockout | 3.02% | 0.95% (-68%) |
| Units short per store-item-week | 0.36 | 0.06 |
| Avg leftover stock | 66.05 | 55.95 |
This is the "stockout risk reduction vs baseline": same cushion, about two-thirds fewer stockout weeks.

**B. No safety stock (order exactly the forecast)**
Baseline stocks out in 42.5% of weeks, XGBoost in 49.0%, i.e. XGBoost has MORE
stockout weeks here. Reason: the baseline over-forecasts by ~1.3 units a day on
average (Stage 6 bias), which acts as accidental cushion; XGBoost is unbiased, so
it falls short about half the time. XGBoost still misses fewer units in total
(8.63 vs 10.17 per week). This is exactly why safety stock is needed.

## Limits
- Simplified simulation, one quarter, public data, no real costs; the "95%" is only as
  good as the assumption that the next quarter's errors look like the validation quarter's
  (XGBoost hit 95.3% vs the 95% target, so here it held).
- Savings in units, not money: no unit costs in the dataset, so I won't claim a rupee/dollar saving.
- A different lead time or service level changes all numbers.
