from typing import List, Tuple
import pandas as pd

TRAIN_DATES = {
    "2018-02-14",
    "2018-02-16",
    "2018-02-20",
    "2018-02-22",
    "2018-02-28",
}
VAL_DATES = {"2018-02-15", "2018-02-23"}
TEST_DATES = {"2018-02-21", "2018-03-01", "2018-03-02"}

TARGET_COL = "Future_Attack_Target"
TIMESTAMP_COL = "window_start"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df["date"] = df[TIMESTAMP_COL].dt.date.astype(str)
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("date")[TARGET_COL]
        .agg(total="count", positives="sum")
        .assign(pos_rate=lambda x: x["positives"] / x["total"])
    )

    print("\n--- Complete 10-Day Session Audit ---")
    print(summary.to_string())

    suspicious = summary[summary["total"] < 50]
    if not suspicious.empty:
        print("\nFlagged small sessions:")
        print(suspicious.to_string())
    else:
        print("\nAll 10 capture sessions verified with sufficient sample volume.")

    return summary


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["date"].isin(TRAIN_DATES)].copy()
    val = df[df["date"].isin(VAL_DATES)].copy()
    test = df[df["date"].isin(TEST_DATES)].copy()

    print("\n--- Attack-Family-Diverse Split Breakdown ---")
    for name, s in [("Train", train), ("Val", val), ("Test", test)]:
        pos = s[TARGET_COL].sum()
        pct = (pos / len(s)) * 100 if len(s) > 0 else 0
        print(
            f"  {name:5s}: {len(s):6,} rows | {int(pos):5d} attack precursors ({pct:.2f}%)"
        )

    return train, val, test


def feature_cols(df: pd.DataFrame) -> List[str]:
    drop = {TIMESTAMP_COL, "date", TARGET_COL}
    return [c for c in df.columns if c not in drop]


def xy(split_df: pd.DataFrame, cols: List[str]):
    X = split_df[cols].values.astype(float)
    y = split_df[TARGET_COL].values.astype(int)
    return X, y
