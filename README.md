# HRND Prediction Model Monitor

Monitoring dashboard for the HESI Likely Exit Score (LHES) prediction model — tracks predicted vs. actual performance, at-risk miss rate trend, threshold comparison, and prediction confidence by exams taken.

## How to refresh data

1. Make sure you're on the Elsevier network (or VPN + Zscaler)
2. Run the extraction script — a browser SSO tab will open:

```bash
python3 extract_data.py
```

3. Commit and push the updated `data/` files:

```bash
git add data/
git commit -m "chore: refresh dashboard data"
git push
```

The GitHub Pages site updates within ~1 minute of pushing.

## Dashboard panels

| Panel | What it shows |
|-------|--------------|
| KPI row | Miss rate, MedAE, MAE, % within ±50/±100 pts, student count |
| Error distribution | Count of students by (actual − predicted) in 10-pt bands |
| At-risk miss rate trend | Monthly false-negative rate over 24 months vs. 20% target line |
| Threshold comparison | Miss rate / recall / precision / false positives at thresholds 820–900 |
| Predicted vs. actual scatter | 5,000-student sample; diagonal = perfect prediction |
| Confidence by exam count | MedAE and IQR by number of specialty exams taken |

## Data sources

All tables in `ELS_SOURCE_SYSTEMS_DB_PROD.SRC_NHE_INSIGHTS`:

- `HESIDW_FACT_STUDENT_PREDICTION` — current snapshot (1.08M rows)
- `HESIDW_FACT_STUDENT_PREDICTION_HISTORY` — 24-month archive (47M rows)
- `HESIDW_FACT_TEST_TAKING_EDITION` — raw test-taking events (21M rows)

## Requirements

```
snowflake-connector-python>=4.5.0
pandas
```
