import glob
import os
import sys
from typing import List
import numpy as np
import polars as pl

BASE_FEATURE_NAMES: List[str] = [
    "Tot Fwd Pkts_sum",
    "Tot Bwd Pkts_sum",
    "TotLen Fwd Pkts_sum",
    "TotLen Bwd Pkts_sum",
    "Flow Duration_mean",
    "Flow Duration_std",
    "Flow IAT Mean_mean",
    "Dst Port_nunique",
    "SYN Flag Cnt_sum",
    "ACK Flag Cnt_sum",
    "RST Flag Cnt_sum",
    "FIN Flag Cnt_sum",
    "PSH Flag Cnt_sum",
    "flow_count",
    "bwd_fwd_pkt_ratio",
]

NUMERIC_SUM_COLUMNS: List[str] = [
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "RST Flag Cnt",
    "FIN Flag Cnt",
    "PSH Flag Cnt",
]


def aggregate_day_file(file_path: str) -> pl.DataFrame:
    print(f"  -> Scanning: {os.path.basename(file_path)}...", end="", flush=True)

    needed_cols = [
        "Timestamp",
        "Label",
        "Dst Port",
        "Flow Duration",
        "Flow IAT Mean",
    ] + NUMERIC_SUM_COLUMNS

    lf = pl.scan_csv(
        file_path,
        infer_schema_length=10000,
        ignore_errors=True,
    ).select(needed_cols)

    lf = lf.filter(pl.col("Timestamp") != "Timestamp")

    lf = lf.with_columns(
        pl.col("Timestamp")
        .str.to_datetime("%d/%m/%Y %H:%M:%S", strict=False)
        .alias("ts")
    ).filter(pl.col("ts").is_not_null() & (pl.col("ts").dt.year() == 2018))

    cast_exprs = [
        pl.col(c).cast(pl.Float64).fill_null(0.0) for c in NUMERIC_SUM_COLUMNS
    ] + [
        pl.col("Flow Duration").cast(pl.Float64).fill_null(0.0),
        pl.col("Flow IAT Mean").cast(pl.Float64).fill_null(0.0),
        pl.col("Dst Port").cast(pl.UInt32).fill_null(0),
        (pl.col("Label") != "Benign").cast(pl.UInt8).alias("is_attack"),
    ]
    lf = lf.with_columns(cast_exprs)

    lf = lf.with_columns(
        pl.col("ts").dt.truncate("10s").alias("window_start")
    )

    agg_exprs = [
        pl.col("Tot Fwd Pkts").sum().alias("Tot Fwd Pkts_sum"),
        pl.col("Tot Bwd Pkts").sum().alias("Tot Bwd Pkts_sum"),
        pl.col("TotLen Fwd Pkts").sum().alias("TotLen Fwd Pkts_sum"),
        pl.col("TotLen Bwd Pkts").sum().alias("TotLen Bwd Pkts_sum"),
        pl.col("Flow Duration").mean().alias("Flow Duration_mean"),
        pl.col("Flow Duration").std(ddof=1).alias("Flow Duration_std"),
        pl.col("Flow IAT Mean").mean().alias("Flow IAT Mean_mean"),
        pl.col("Dst Port").n_unique().cast(pl.Float64).alias("Dst Port_nunique"),
        pl.col("SYN Flag Cnt").sum().alias("SYN Flag Cnt_sum"),
        pl.col("ACK Flag Cnt").sum().alias("ACK Flag Cnt_sum"),
        pl.col("RST Flag Cnt").sum().alias("RST Flag Cnt_sum"),
        pl.col("FIN Flag Cnt").sum().alias("FIN Flag Cnt_sum"),
        pl.col("PSH Flag Cnt").sum().alias("PSH Flag Cnt_sum"),
        pl.len().cast(pl.Float64).alias("flow_count"),
        (pl.col("Tot Bwd Pkts").sum() / (pl.col("Tot Fwd Pkts").sum() + 1e-6)).alias(
            "bwd_fwd_pkt_ratio"
        ),
        pl.col("is_attack").max().alias("has_attack"),
    ]

    agg_df = lf.group_by("window_start").agg(agg_exprs).sort("window_start").collect()
    print(f" [done: {agg_df.height:,} windows]")
    return agg_df


def compute_temporal_features(base_df: pl.DataFrame) -> pl.DataFrame:
    df = base_df.sort("window_start")

    lag_exprs = []
    for f in BASE_FEATURE_NAMES:
        lag_exprs.append(pl.col(f).shift(1).alias(f"{f}_lag1"))
        lag_exprs.append(pl.col(f).shift(2).alias(f"{f}_lag2"))
        lag_exprs.append(pl.col(f).shift(3).alias(f"{f}_lag3"))
    df = df.with_columns(lag_exprs)

    delta_exprs = [
        (pl.col(f) - pl.col(f"{f}_lag1")).alias(f"{f}_delta1")
        for f in BASE_FEATURE_NAMES
    ]
    df = df.with_columns(delta_exprs)

    df = df.with_columns(
        [
            pl.col("Flow Duration_std").fill_null(0.0),
            pl.col("Flow Duration_std_lag1").fill_null(0.0),
            pl.col("Flow Duration_std_lag2").fill_null(0.0),
            pl.col("Flow Duration_std_lag3").fill_null(0.0),
            pl.col("Flow Duration_std_delta1").fill_null(0.0),
        ]
    )

    has_attack_arr = df["has_attack"].to_numpy()
    n = len(has_attack_arr)
    target_arr = np.zeros(n, dtype=np.float64)

    for i in range(n):
        end = min(n, i + 13)
        if np.any(has_attack_arr[i + 1 : end] == 1):
            target_arr[i] = 1.0

    df = df.with_columns(pl.Series("Future_Attack_Target", target_arr))

    filtered_df = df.filter(
        (pl.col("has_attack") == 0) & pl.col("Tot Fwd Pkts_sum_lag3").is_not_null()
    ).drop("has_attack")

    filtered_df = filtered_df.with_columns(
        pl.col("window_start")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .alias("window_start")
    )

    return filtered_df


def aggregate_raw_directory(
    input_pattern: str,
    output_path: str,
) -> pl.DataFrame:
    files = sorted(glob.glob(input_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern: {input_pattern}")

    print("=" * 65)
    print(f" VectorFlow Full Day Aggregation ({len(files)} files)")
    print("=" * 65)

    daily_results = []
    for f in files:
        daily_results.append(aggregate_day_file(f))

    print("\nConcatenating days and generating lags + forecasting targets...")
    combined_base = pl.concat(daily_results).sort("window_start")
    final_df = compute_temporal_features(combined_base)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.write_csv(output_path)
    print(f"\nSaved complete dataset: {final_df.height:,} windows -> {output_path}")
    return final_df


if __name__ == "__main__":
    raw_pattern = (
        sys.argv[1] if len(sys.argv) > 1 else "cic-ids2018-processed/*.csv"
    )
    dest_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "data/cic_ids2018_complete_dataset.csv"
    )
    aggregate_raw_directory(raw_pattern, dest_csv)
