import polars as pl
import os

path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\test.parquet'
df = pl.scan_parquet(path)

schema = pl.read_parquet_schema(path)
all_features = [col for col in schema.keys() if col.startswith("feature_") and not col.startswith("is_nan")]

# Lấy các cột có sẵn trong test
stats = df.select([
    pl.struct([
        (pl.col(c).min()).alias("min"),
        (pl.col(c).max()).alias("max"),
        (pl.col(c).mean()).alias("mean")
    ]).alias(c) for c in all_features
]).collect()

has_both = []
only_pos = []
only_neg = []

for c in all_features:
    s = stats[c][0]
    if s["min"] is not None and s["max"] is not None:
        if s["min"] < 0 and s["max"] > 0:
            has_both.append(c)
        elif s["max"] > 0:
            only_pos.append(c)
        elif s["min"] < 0:
            only_neg.append(c)

print(f"Tổng số feature kiểm tra trong test: {len(all_features)}")
print(f"Feature có cả (- & +): {len(has_both)}")
print(f"Feature chỉ có (+): {len(only_pos)}")
print(f"Feature chỉ có (-): {len(only_neg)}")

print("\nVí dụ các feature có cả hai (test):")
for c in has_both[:10]:
    s = stats[c][0]
    print(f"  {c:15} | Min: {s['min']:10.4f} | Max: {s['max']:10.4f} | Mean: {s['mean']:10.4f}")

print("\nVí dụ các feature chỉ có âm (test):")
for c in only_neg[:10]:
    s = stats[c][0]
    print(f"  {c:15} | Min: {s['min']:10.4f} | Max: {s['max']:10.4f} | Mean: {s['mean']:10.4f}")

# Kiểm tra nếu y_target có trong test (thường là không)
if "y_target" in schema:
    y_stats = df.select([pl.col("y_target").min(), pl.col("y_target").max()]).collect()
    print(f"\ny_target range in test: {y_stats[0,0]} to {y_stats[0,1]}")
else:
    print("\ntest.parquet không có cột y_target.")
