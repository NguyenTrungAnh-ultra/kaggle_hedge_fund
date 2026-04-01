import polars as pl
import os

path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train.parquet'
df = pl.scan_parquet(path)

schema = pl.read_parquet_schema(path)
all_features = [col for col in schema.keys() if col.startswith("feature_") and not col.startswith("is_nan")]

stats = df.select([
    pl.struct([
        (pl.col(c).min()).alias("min"),
        (pl.col(c).max()).alias("max"),
        (pl.col(c).mean()).alias("mean")
    ]).alias(c) for c in all_features + ["y_target"]
]).collect()

has_both = []
only_pos = []
only_neg = []

for c in all_features + ["y_target"]:
    s = stats[c][0]
    if s["min"] < 0 and s["max"] > 0:
        has_both.append(c)
    elif s["max"] > 0:
        only_pos.append(c)
    elif s["min"] < 0:
        only_neg.append(c)

print(f"Tổng số feature kiểm tra: {len(all_features)}")
print(f"Feature có cả (- & +): {len(has_both)}")
print(f"Feature chỉ có (+): {len(only_pos)}")
print(f"Feature chỉ có (-): {len(only_neg)}")

print("\nVí dụ các feature có cả hai:")
for c in has_both[:10]:
    s = stats[c][0]
    print(f"  {c:15} | Min: {s['min']:10.4f} | Max: {s['max']:10.4f} | Mean: {s['mean']:10.4f}")

print("\nVí dụ các feature chỉ có (+):")
for c in only_pos[:10]:
    s = stats[c][0]
    print(f"  {c:15} | Min: {s['min']:10.4f} | Max: {s['max']:10.4f} | Mean: {s['mean']:10.4f}")

if "y_target" in has_both:
    print("\ny_target có cả âm và dương.")
else:
    print(f"\ny_target chỉ có {'(+)' if 'y_target' in only_pos else '(-)'}.")
