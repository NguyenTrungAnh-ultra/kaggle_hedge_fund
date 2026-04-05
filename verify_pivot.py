import pandas as pd
import os
import numpy as np

DATA_DIR = r"f:\Projects\hege_fund_kaggle\ts-forecasting"
file_path = os.path.join(DATA_DIR, "train_feature_builded.parquet")
# Load a larger chunk to be sure
df = pd.read_parquet(file_path, columns=['ts_index', 'code', 'sub_code', 'sub_category', 'horizon', 'y_target', 'weight', 'feature_a'])

print(f"Original shape: {df.shape}")
print(f"Unique horizons: {df['horizon'].unique()}")

# Check grouping
group_cols = ['ts_index', 'code', 'sub_code', 'sub_category']
counts = df.groupby(group_cols).size().value_counts()
print("\nCounts of unique samples across horizons:")
print(counts)

# Check if feature_a is consistent within groups
# We expect std() to be 0 for a feature that is common to all horizons of a sample
feature_std = df.groupby(group_cols)['feature_a'].std().dropna()
print(f"\nFeature consistency (feature_a std across horizons):")
print(f"Mean std: {feature_std.mean()}")
print(f"Max std: {feature_std.max()}")

# If std is near 0, then features are duplicated in Long format, 
# and MTA (Wide format) will save massive memory.
