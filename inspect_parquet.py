import pandas as pd
import pyarrow.parquet as pq
import os

file_path = r"f:\Projects\hege_fund_kaggle\ts-forecasting\train_feature_builded.parquet"
file_size_gb = os.path.getsize(file_path) / (1024**3)
print(f"File size on disk: {file_size_gb:.2f} GB")

# Read metadata without reading the full file
metadata = pq.read_metadata(file_path)
print(f"Number of rows: {metadata.num_rows}")
print(f"Number of columns: {metadata.num_columns}")

schema = metadata.schema
types = {}
for i in range(metadata.num_columns):
    type_str = str(schema.column(i).physical_type)
    types[type_str] = types.get(type_str, 0) + 1

print("\nPhysical Data Types (in Parquet):")
for t, count in types.items():
    print(f"{t}: {count}")

# Look at schema types (logical)
logical_types = {}
for i in range(metadata.num_columns):
    type_str = str(schema.column(i).logical_type)
    logical_types[type_str] = logical_types.get(type_str, 0) + 1

print("\nLogical Data Types (in Parquet):")
for t, count in logical_types.items():
    print(f"{t}: {count}")

# Check for large columns (if many are doubles, it will be larger)
# We can also check if there's any compression used
print(f"\nCompression used in first row group:")
for i in range(metadata.num_columns):
    if i == 0:
        print(f"Example Compression: {metadata.row_group(0).column(i).compression}")
        break
