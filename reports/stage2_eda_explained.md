# Stage 2: What the data is telling us (plain-English version)

All numbers below come from running `python -m src.eda` on the real train.csv
(raw output saved in `eda_summary.txt`).

**EDA** = Exploratory Data Analysis. Before building any model, you look at the
data the way you'd look at a new city before opening a shop there.

## 1. Is the data clean?
- 0 missing values. Every one of the 500 store-item pairs has all 1,826 days (no gaps).
- Exactly 1 row with zero sales, no negatives.
- A **spike** is a day far above normal. Here, a "spike" means more than 3 standard
  deviations above that store-item's own average. (Standard deviation = how much
  daily sales typically wiggle around the average.) That's 2,373 rows, 0.26% of
  the data. We keep them: real demand has busy days, and a forecast that ignores
  them would under-stock.

## 2. Trend (chart 01): is demand going up?
**Trend** = the slow long-term direction. Total units sold per year:
2013: 7.94M, 2014: 9.14M, 2015: 9.54M, 2016: 10.36M, 2017: 10.73M.
Growth per year: +15.0%, +4.4%, +8.6%, +3.6%. So demand keeps growing, but the
growth is slowing. The grey lines are daily totals (the thick comb-like look is
the weekly cycle); the blue line is the monthly average, which smooths it out.

## 3. Weekly seasonality (chart 02)
**Seasonality** = a pattern that repeats on a calendar. Average units per
store-item per day: Mon 41.4, Tue 48.2, Wed 48.4, Thu 51.7, Fri 55.2, Sat 58.7,
Sun 62.1. Sunday is 50% busier than Monday. This is why the baseline in Stage 4
is "same as last week": same weekday last week is a much better guess than
yesterday.

## 4. Yearly seasonality (chart 03)
Average units per store-item per day by month: Jan 35.5, Jul 67.0 (peak),
Dec 39.4. July is 88.6% higher than January. Chart 03 draws each year as its own
line: the shape is the same every year (rise to July, dip in autumn, small bump
in November, drop in December) while the whole curve moves up each year. That is
trend + seasonality together, which is exactly what forecasting models are built
to capture.

## 5. Store and item differences (chart 04)
- Stores: Store 2 sells most (67.0 per item per day), Store 7 least (36.4).
  Best vs worst is 84.3% higher.
- Items: item 15 sells most (88.0 per store per day), item 5 least (18.4):
  4.8x difference.
- Why it matters for the business: the reorder quantity for a fast mover is
  completely different from a slow one, so we forecast each of the 500
  store-item pairs separately (or give the model store and item as inputs).

## 6. One store-item up close (chart 05)
Store 1, item 1: averages 20.0 units/day, ranges 4 to 50. Daily sales are
noisy even though the pattern is clear. That noise is the part no model can
predict, and it is what safety stock (Stage 7) is for.

## Honest caveats
- This is a public competition dataset that looks synthetic (several items have
  almost identical averages). Real company data would be messier: stockouts,
  promotions, price changes, new products.
- Sales is not the same as demand: if a shelf was empty, true demand was higher
  than recorded sales. We have no stock data, so we can't correct for that.
