import pandas as pd
import os

file_path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'

if not os.path.exists(file_path):
    print(f"Error: File {file_path} not found.")
else:
    print(f"Reading {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Shape: {df.shape}")
    
    na_counts = df.isna().sum()
    total_nas = na_counts.sum()
    
    if total_nas == 0:
        print("Success: No NA values found in the dataset.")
    else:
        print(f"Warning: Found {total_nas} NA values.")
        cols_with_nas = na_counts[na_counts > 0]
        print("Columns with NAs:")
        print(cols_with_nas)
