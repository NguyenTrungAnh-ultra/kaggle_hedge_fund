import pandas as pd
import pyarrow.parquet as pq

file_path = r"f:\Projects\hege_fund_kaggle\ts-forecasting\train_feature_builded.parquet"
metadata = pq.read_metadata(file_path)
schema = metadata.schema

print("Columns with physical type DOUBLE:")
for i in range(metadata.num_columns):
    col = schema.column(i)
    if str(col.physical_type) == "DOUBLE":
        print(f"  - {col.name}")
