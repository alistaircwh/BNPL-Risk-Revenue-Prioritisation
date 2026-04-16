# BNPL Merchant Risk & Revenue Prioritisation

A data-driven framework for prioritising Buy Now, Pay Later (BNPL) merchant partnerships by combining fraud risk assessment with revenue forecasting.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PySpark](https://img.shields.io/badge/PySpark-3.3%2B-orange) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)

---

## Overview

BNPL platforms must decide which merchants to partner with and how aggressively to pursue each relationship. Prioritising solely by revenue ignores fraud exposure; prioritising solely by fraud risk leaves money on the table. This project builds a four-stage analytical pipeline to rank merchants by **risk-adjusted expected present value (EPV)**.

**Pipeline:**
1. **ETL** — Clean and join 12.4M transactions, 3,212 merchants, and 34,747 consumers; impute geographic features (LGA codes, income metrics) via KNN; integrate ABS demographic data
2. **Exploratory Analysis** — Geospatial fraud distribution, revenue mix by merchant segment, temporal trends
3. **Fraud Modelling** — Separate Random Forest regression models for consumer-level and merchant-level fraud probability
4. **Merchant Ranking** — LSTM revenue forecasting (3 periods ahead) combined with fraud scores to produce a final risk-adjusted ranking across five industry segments

---

## Key Results

**Dataset scale:** 12.4M transactions · 3,212 merchants · 34,747 consumers

| Model | RMSE | R² | Top Feature |
|-------|------|----|-------------|
| Consumer fraud (RFR) | 6.81 | 0.43 | Transaction dollar value (42.5% importance) |
| Merchant fraud (RFR) | 2.43 | **0.85** | Monthly order volume (31.4% importance) |

**Revenue distribution:** Level 'a' merchants account for $47.8M (~56% of total revenue), yet share a similar fraud probability (~29%) to Level 'b' and 'c' merchants — indicating revenue level is a reliable proxy for lower fraud exposure. Level 'e' merchants average **69% fraud probability** vs 29% for Level 'a'.

**Merchant segments analysed (5):**

<p align="center">
  <img src="plots/donut_chart_segments.png" width="480" alt="Merchant segment distribution">
</p>

**Top-ranked merchant EPV:** ~$54,713 risk-adjusted EPV (fraud weights: 65% merchant, 35% consumer). Top 10 merchants by commission each generate >$490K in estimated commission from >$7.4M in revenue.

**Geospatial fraud spread:** Consumer fraud probability ranges from 8% to 53% by postcode, with relatively uniform state-level averages (14.4%–15.5%), suggesting fraud is driven by local demographic factors rather than state-wide patterns.

<p align="center">
  <img src="plots/average_fraud_prob_postcode.png" width="600" alt="Average fraud probability by postcode">
</p>

### Practical Outcomes

The final output is a ranked list of merchants ordered by risk-adjusted EPV — a single score combining forecasted revenue with fraud exposure. In practice, a BNPL platform could use this to:
- **Prioritise onboarding** for high-EPV, low-fraud merchants (Level 'a' revenue band, ~29% fraud probability)
- **Flag for review** any merchant with fraud probability above ~50% before offering them partnership terms
- **Target marketing spend** toward the Books/Media and Fashion segments, which show the highest aggregate EPV
- **Deprioritise** Level 'e' merchants (average 69% fraud probability, <0.1% of total revenue) until fraud controls are established

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ETL & ML pipelines | PySpark 3.3+, PySpark MLlib |
| Revenue forecasting | TensorFlow / Keras (LSTM) |
| Geospatial imputation | scikit-learn (KNeighborsRegressor), geopandas |
| Visualisation | matplotlib, seaborn, folium (interactive maps) |
| Data wrangling | pandas, numpy |

---

## Project Structure

```
notebooks/          # Run in order 1 → 5
  1_ETL_pipeline              # Data cleaning, joining, KNN imputation
  2.1_preliminary_analysis    # Missing values, distributions, data quality
  2.2_geospatial_analysis     # Postcode/LGA choropleth maps
  2.3_visualisation           # Feature distributions, segment breakdowns
  3.1a/b_consumer_fraud_model # Consumer fraud probability (v1 → v2)
  3.2_merchant_fraud_model    # Merchant fraud probability
  4.1b_ranking_model_v2       # LSTM forecasting + risk-adjusted ranking
  4.2_segments                # Segment profiling and top-merchant analysis
  5_summary                   # End-to-end review of findings

scripts/            # Reusable modules imported by notebooks
  etl_pipeline.py             # Core ETL functions and KNN imputation
  consumer_transaction_model.py  # Consumer fraud ML pipeline
  merchant_fraud.py           # Merchant fraud model with CrossValidator
  ranking_model_v2.py         # LSTM forecasting and weight generation
  geospatial_analysis.py      # Folium map utilities
  visualisation.py            # Chart helpers

data/
  raw/              # Tracked — shapefiles, postcodes, ABS income/fraud data
  curated/          # Gitignored — generated by notebook 1
  tables/           # Gitignored — generated by notebook 1

plots/              # Saved visualisation outputs (PNG)
```

---

## Setup

**Prerequisites:** Python 3.8+, Java 8 or 11 (required by PySpark)

```bash
pip install -r requirements.txt
```

> **Note:** `data/curated/` and `data/tables/` are not tracked in git. Run `notebooks/1_ETL_pipeline.ipynb` first to generate them before running any subsequent notebooks.

**Run order:**
```
1_ETL_pipeline → 2.1 → 2.2 → 2.3 → 3.1a → 3.1b → 3.2 → 4.1b → 4.2 → 5_summary
```

For dataset schema details (merchant revenue bands, transaction structure, consumer ID mapping), see [`data/README.md`](data/README.md).

---

## Limitations & Future Work

- **Consumer fraud R² (0.43) is moderate** — adding features such as consumer transaction history length, product category diversity, or device/session signals would likely improve prediction quality
- **LSTM forecasting is limited to merchants with complete sales records** — only 3,212 of the full merchant base have enough history for reliable sequence modelling; sparse-data merchants are excluded from the ranking
- **Fraud labels are simulated** — the dataset uses a fraud delta file applied to synthetic transaction data; real-world fraud distributions would differ in ways that could shift model behaviour significantly
- **Fixed fraud weighting** — the 65% merchant / 35% consumer split was chosen heuristically; a sensitivity analysis across weighting combinations was not performed and could reveal more robust configurations
- **KNN imputation for geospatial features** — LGA codes and income metrics for postcodes with missing data are imputed using nearest-neighbour geographic proximity, which may not reflect actual socioeconomic conditions in low-density areas

---

## Team

1. Do Nhat Anh Ha
2. Alistair Cheah Wern Hao
3. Sitao Qin
4. Shiping Song
