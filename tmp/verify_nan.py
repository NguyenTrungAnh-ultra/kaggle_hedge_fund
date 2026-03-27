import pandas as pd
import os

file_path = "f:/Projects/hege_fund_kaggle/ts-forecasting/inference_processed_full.parquet"
if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print("DataFrame shape:", df.shape)
    
    total_cells = df.shape[0] * df.shape[1]
    total_nans = df.isna().sum().sum()
    print(f"Total NaNs: {total_nans} ({total_nans / total_cells * 100:.2f}%)")
    
    # Check a specific stationary feature to see if NaNs are present but not exploding
    print("feature_a NaNs:", df['feature_a'].isna().sum())
    
    print("Verification complete.")
else:
    print("File not found.")
