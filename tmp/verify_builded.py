import pandas as pd
import os

file_path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_feature_builded.parquet'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    df = pd.read_parquet(file_path)
    print(f"Shape: {df.shape}")
    print(f"Total NAs: {df.isna().sum().sum()}")
    print(f"Last columns: {df.columns[-10:].tolist()}")
