"""
HRND Monitoring Dashboard — Data Extraction Script

Run this locally (SSO auth will open a browser tab) to refresh the dashboard data.
Outputs JSON files to ./data/ which are read by index.html.

Usage:
    python3 extract_data.py

Requirements:
    pip install snowflake-connector-python pandas
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import snowflake.connector

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CONN_PARAMS = dict(
    account="elsevier-eu",
    user="TSAIH@SCIENCE.REGN.NET",
    authenticator="externalbrowser",
    warehouse="ELS_NHE_INSIGHTS_READER_WH_PROD",
    database="ELS_SOURCE_SYSTEMS_DB_PROD",
    schema="SRC_NHE_INSIGHTS",
    role="ELS_SNF_NHE_INSIGHTS_DEVELOPER_PROD",
)

DB = "ELS_SOURCE_SYSTEMS_DB_PROD.SRC_NHE_INSIGHTS"


def run_query(cur, sql: str, label: str) -> pd.DataFrame:
    print(f"  Running: {label}...")
    cur.execute(sql)
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    print(f"    → {len(df):,} rows")
    return df


def save(name: str, data) -> None:
    path = DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, default=str)
    print(f"  Saved {path.name}")


def main():
    print("Connecting to Snowflake (browser SSO will open)...")
    conn = snowflake.connector.connect(**CONN_PARAMS)
    cur = conn.cursor()
    print("Connected.\n")

    # ── 1. Error distribution (prediction vs actual) ─────────────────────────
    print("1/4  Error distribution")
    df_err = run_query(cur, f"""
        SELECT
            CURRENT_PREDICTED_EXITSCORE::FLOAT AS predicted,
            CURRENT_ACTUAL_EXITSCORE::FLOAT    AS actual,
            (CURRENT_ACTUAL_EXITSCORE - CURRENT_PREDICTED_EXITSCORE)::FLOAT AS error
        FROM {DB}.HESIDW_FACT_STUDENT_PREDICTION
        WHERE CURRENT_ACTUAL_EXITSCORE IS NOT NULL
          AND CURRENT_PREDICTED_EXITSCORE IS NOT NULL
    """, "error distribution")

    # Bucket errors into ±10-pt bands
    bins = list(range(-300, 310, 10))
    labels = [f"{b}" for b in bins[:-1]]
    df_err["band"] = pd.cut(df_err["ERROR"], bins=bins, labels=labels)
    error_dist = (
        df_err.groupby("band", observed=True)
        .size()
        .reset_index(name="count")
        .assign(band=lambda d: d["band"].astype(str))
    )

    # Scatter sample (max 5000 points for rendering)
    scatter_sample = (
        df_err[["PREDICTED", "ACTUAL"]]
        .sample(min(5000, len(df_err)), random_state=42)
        .round(0)
        .astype(int)
    )

    save("error_distribution", error_dist.to_dict(orient="records"))
    save("scatter_sample", scatter_sample.rename(columns=str.lower).to_dict(orient="records"))

    # Summary stats
    n = len(df_err)
    mae = df_err["ERROR"].abs().mean()
    medae = df_err["ERROR"].abs().median()
    bias = df_err["ERROR"].mean()
    pct_50 = (df_err["ERROR"].abs() <= 50).mean() * 100
    pct_100 = (df_err["ERROR"].abs() <= 100).mean() * 100

    # ── 2. At-risk miss rate over time ────────────────────────────────────────
    print("\n2/4  At-risk miss rate over time")
    df_hist = run_query(cur, f"""
        SELECT
            DATE_TRUNC('month', ARCHIVEDATE)   AS month,
            COUNT(*)                            AS total_with_actual,
            SUM(CASE
                WHEN CURRENT_ACTUAL_EXITSCORE < 850
                 AND CURRENT_PREDICTED_EXITSCORE >= 850 THEN 1 ELSE 0
            END)                                AS false_negatives,
            SUM(CASE
                WHEN CURRENT_ACTUAL_EXITSCORE < 850 THEN 1 ELSE 0
            END)                                AS true_at_risk
        FROM {DB}.HESIDW_FACT_STUDENT_PREDICTION_HISTORY
        WHERE CURRENT_ACTUAL_EXITSCORE IS NOT NULL
          AND CURRENT_PREDICTED_EXITSCORE IS NOT NULL
          AND ARCHIVEDATE >= DATEADD(month, -24, CURRENT_DATE())
        GROUP BY 1
        ORDER BY 1
    """, "miss rate trend (24 months)")

    df_hist["MISS_RATE_PCT"] = (
        df_hist["FALSE_NEGATIVES"] / df_hist["TRUE_AT_RISK"].replace(0, pd.NA) * 100
    ).round(2)
    save("miss_rate_trend", df_hist.assign(
        month=lambda d: d["MONTH"].astype(str)
    ).rename(columns=str.lower).to_dict(orient="records"))

    # ── 3. Threshold comparison ───────────────────────────────────────────────
    print("\n3/4  Threshold comparison")
    df_thresh_base = run_query(cur, f"""
        SELECT
            CURRENT_PREDICTED_EXITSCORE::FLOAT AS predicted,
            CURRENT_ACTUAL_EXITSCORE::FLOAT    AS actual
        FROM {DB}.HESIDW_FACT_STUDENT_PREDICTION
        WHERE CURRENT_ACTUAL_EXITSCORE IS NOT NULL
          AND CURRENT_PREDICTED_EXITSCORE IS NOT NULL
    """, "threshold base data")

    threshold_rows = []
    for t in range(820, 910, 10):
        tp = ((df_thresh_base["PREDICTED"] >= t) & (df_thresh_base["ACTUAL"] >= 850)).sum()
        fp = ((df_thresh_base["PREDICTED"] >= t) & (df_thresh_base["ACTUAL"] < 850)).sum()
        fn = ((df_thresh_base["PREDICTED"] < t)  & (df_thresh_base["ACTUAL"] >= 850)).sum()
        tn = ((df_thresh_base["PREDICTED"] < t)  & (df_thresh_base["ACTUAL"] < 850)).sum()
        at_risk_total = fp + tn  # actual <850
        miss_rate = round(fn / (tp + fn) * 100, 2) if (tp + fn) > 0 else None
        recall    = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else None
        precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else None
        threshold_rows.append({
            "threshold": int(t),
            "miss_rate": miss_rate,
            "recall": recall,
            "precision": precision,
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
        })
    save("threshold_comparison", threshold_rows)

    # ── 4. Error by number of specialty exams ────────────────────────────────
    print("\n4/4  Error by exam count")
    df_conf = run_query(cur, f"""
        WITH base AS (
            SELECT
                p.EVOLVEUSERNAME,
                p.CURRENT_PREDICTED_EXITSCORE::FLOAT AS predicted,
                p.CURRENT_ACTUAL_EXITSCORE::FLOAT    AS actual,
                COUNT(DISTINCT t.EXAMKEY)             AS specialty_exam_count
            FROM {DB}.HESIDW_FACT_STUDENT_PREDICTION p
            JOIN {DB}.HESIDW_FACT_TEST_TAKING_EDITION t
              ON p.EVOLVEUSERNAME = t.TESTTAKERKEY::VARCHAR
            WHERE p.CURRENT_ACTUAL_EXITSCORE IS NOT NULL
              AND p.CURRENT_PREDICTED_EXITSCORE IS NOT NULL
            GROUP BY 1, 2, 3
        )
        SELECT
            CASE
                WHEN specialty_exam_count <= 2  THEN '1–2 exams'
                WHEN specialty_exam_count <= 4  THEN '3–4 exams'
                WHEN specialty_exam_count <= 7  THEN '5–7 exams'
                ELSE '8+ exams'
            END AS exam_bucket,
            COUNT(*)                                             AS n_students,
            ROUND(MEDIAN(ABS(actual - predicted)), 1)           AS medae,
            ROUND(AVG(ABS(actual - predicted)), 1)              AS mae,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
                  (ORDER BY ABS(actual - predicted)), 1)        AS p25_error,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY ABS(actual - predicted)), 1)        AS p75_error
        FROM base
        GROUP BY 1
        ORDER BY MIN(specialty_exam_count)
    """, "error by exam count")
    save("error_by_exam_count", df_conf.rename(columns=str.lower).to_dict(orient="records"))

    conn.close()

    # ── Metadata ─────────────────────────────────────────────────────────────
    save("metadata", {
        "refreshed_at": datetime.utcnow().isoformat() + "Z",
        "n_students_with_actual": int(n),
        "mae": round(float(mae), 1),
        "medae": round(float(medae), 1),
        "bias": round(float(bias), 2),
        "pct_within_50": round(float(pct_50), 1),
        "pct_within_100": round(float(pct_100), 1),
    })

    print("\nAll done. Refresh the dashboard by opening index.html.")


if __name__ == "__main__":
    main()
