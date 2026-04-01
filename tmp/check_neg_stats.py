import polars as pl
import os

path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train.parquet'
df = pl.scan_parquet(path)
cols = ['feature_bs', 'feature_bt', 'feature_bz', 'feature_ca', 'feature_cb', 'feature_cc', 'feature_cd', 'feature_ce']

exprs = []
for c in cols:
    exprs.append(pl.col(c).min().alias(f"{c}_min"))
    exprs.append(pl.col(c).max().alias(f"{c}_max"))
    exprs.append(pl.col(c).mean().alias(f"{c}_mean"))

stats = df.select(exprs).collect()

for c in cols:
    print(f"{c:12} | Min: {stats[0, f'{c}_min']:12.4f} | Max: {stats[0, f'{c}_max']:12.4f} | Mean: {stats[0, f'{c}_mean']:12.4f}")
