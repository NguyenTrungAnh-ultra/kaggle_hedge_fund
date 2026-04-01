import polars as pl
import os

path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train.parquet'
df = pl.scan_parquet(path)

# Lấy danh sách các cột feature
features = [f"feature_{chr(97 + i)}" for i in range(26)] + \
           [f"feature_a{chr(97 + i)}" for i in range(26)] + \
           [f"feature_b{chr(97 + i)}" for i in range(26)] + \
           [f"feature_c{chr(97 + i)}" for i in range(10)] # feature_a to feature_cj, nhưng hãy lấy từ schema cho chắc

schema = pl.read_parquet_schema(path)
all_features = [col for col in schema.keys() if col.startswith("feature_") and not col.startswith("is_nan")]

stats = df.select([
    pl.struct([
        (pl.col(c).min() < 0).alias("has_negative"),
        (pl.col(c).max() > 0).alias("has_positive"),
        (pl.col(c).min() >= 0).alias("all_positive"),
        (pl.col(c).max() <= 0).alias("all_negative")
    ]).alias(c) for c in all_features
]).collect()

has_both = []
only_pos = []
only_neg = []

for c in all_features:
    s = stats[c][0]
    if s["has_negative"] and s["has_positive"]:
        has_both.append(c)
    elif s["has_positive"]:
        only_pos.append(c)
    elif s["has_negative"]:
        only_neg.append(c)

print(f"Số feature có cả âm và dương: {len(has_both)}")
print(f"Số feature chỉ có dương (hoặc 0): {len(only_pos)}")
print(f"Số feature chỉ có âm (hoặc 0): {len(only_neg)}")

if only_pos:
    print(f"Các feature chỉ có dương: {only_pos[:10]}...")
if only_neg:
    print(f"Các feature chỉ có âm: {only_neg[:10]}...")
