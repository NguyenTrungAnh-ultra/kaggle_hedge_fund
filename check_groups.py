import polars as pl

file_path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'
df = pl.read_parquet(file_path)

print(f"Total rows: {len(df)}")
groups = df.group_by(['code', 'sub_code', 'horizon']).count()
print(f"Number of groups: {len(groups)}")
print(f"Average rows per group: {df['code'].len() / len(groups):.2f}")

# How many groups have >= 20 rows?
large_groups = groups.filter(pl.col('count') >= 20)
print(f"Number of groups with >= 20 rows: {len(large_groups)}")
print(f"Potential valid indices: {large_groups['count'].sum() - len(large_groups) * 19}")
