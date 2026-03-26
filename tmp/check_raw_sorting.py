import pandas as pd
import os

file_path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train.parquet'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    df = pd.read_parquet(file_path, columns=['ts_index', 'code', 'sub_code', 'horizon'])
    print("Dữ liệu gốc (First 10 rows):")
    print(df.head(10))
    
    # Kiểm tra xem nó có được sắp xếp theo ts_index hay không
    is_sorted_ts = df['ts_index'].is_monotonic_increasing
    print(f"\nSắp xếp theo 'ts_index' tăng dần: {is_sorted_ts}")
    
    # Nếu không, kiểm tra các tổ hợp khác
    is_sorted_code_ts = df.sort_values(['code', 'ts_index']).equals(df) # Thường rất chậm cho 5m rows
    
    # Thử lấy head và check local sorting
    print("\nKiểm tra 50 dòng đầu tiên có thực sự được sắp xếp không?")
    print(df.head(50))
