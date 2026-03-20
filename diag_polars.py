import polars as pl
import os

file_path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    print(f"File exists. Size: {os.path.getsize(file_path) / 1e9:.2f} GB")
    try:
        df = pl.scan_parquet(file_path)
        print("Lazy scan successful.")
        print(df.schema)
        # Try to collect just 5 rows
        print(df.head(5).collect())
        print("Collection of 5 rows successful.")
    except Exception as e:
        print(f"Error: {e}")
