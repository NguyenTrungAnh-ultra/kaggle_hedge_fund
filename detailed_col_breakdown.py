import pandas as pd
import pyarrow.parquet as pq

file_path = r"f:\Projects\hege_fund_kaggle\ts-forecasting\train_feature_builded.parquet"
metadata = pq.read_metadata(file_path)
schema = metadata.schema

print(f"Total Columns: {metadata.num_columns}")
double_cols = []
float_cols = []
int_cols = []

for i in range(metadata.num_columns):
    col = schema.column(i)
    ptype = str(col.physical_type)
    if ptype == "DOUBLE":
        double_cols.append(col.name)
    elif ptype == "FLOAT":
        float_cols.append(col.name)
    else:
        int_cols.append((col.name, ptype))

print(f"\nDOUBLE columns ({len(double_cols)}):")
print(double_cols[:20], "... (truncated)")
print("\nFLOAT columns ({len(float_cols)}):")
print(float_cols[:20], "... (truncated)")
print("\nSome INT/Other columns:")
print(int_cols[:10])
