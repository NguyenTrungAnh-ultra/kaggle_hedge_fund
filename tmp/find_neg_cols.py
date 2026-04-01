import polars as pl
import os

path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train.parquet'
df = pl.scan_parquet(path)
schema = pl.read_parquet_schema(path)
all_features = [col for col in schema.keys() if col.startswith("feature_") and not col.startswith("is_nan")]

stats = df.select([
    pl.struct([
        (pl.col(c).min()).alias("min"),
        (pl.col(c).max()).alias("max")
    ]).alias(c) for c in all_features
]).collect()

neg_cols = []
for c in all_features:
    s = stats[0, c]
    if s["max"] <= 0 and s["min"] < 0:
        neg_cols.append(c)

print(f"Strictly negative features count: {len(neg_cols)}")
for c in neg_cols:
    s = stats[0, c]
    print(f"{c:12} | Min: {s['min']:14.4f} | Max: {s['max']:14.4f}")
