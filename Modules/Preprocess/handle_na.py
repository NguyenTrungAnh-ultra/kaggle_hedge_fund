import pandas as pd
import numpy as np
import os
import time
import sys
import gc

# Paths
# BASE_DIR should be f:\Projects\hege_fund_kaggle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_handled_na.parquet')
REPORT_FILE = os.path.join(BASE_DIR, 'reports', 'preprocess_na_report.md')

class NaN_Handler:
    """
    Module Xử lý Dữ liệu khuyết thiếu (NaN Imputation) theo 4 lớp bảo vệ.
    Đảm bảo: Zero Data Leakage, Spatial Isolation, Longitudinal Integrity.
    """
    def process(self, df: pd.DataFrame, features_list: list) -> pd.DataFrame:
        print(f"\n--- BẮT ĐẦU XỬ LÝ NAN: NAN HANDLER ---")
        start_time = time.time()
        
        # Bước 1: Phân tích NaN ban đầu
        na_before = df[features_list].isna().sum()
        features_with_na = na_before[na_before > 0].index.tolist()
        
        if not features_with_na:
            print("Không tìm thấy NaNs trong features.")
            return df
            
        print(f"Phát hiện {len(features_with_na)} features chứa NaNs.")
        
        # Lớp 2: Đánh dấu Khuyết thiếu (Missingness Indicator)
        for col in features_with_na:
            indicator = df[col].isna().astype(np.int8)
            # Thêm bước: Nếu toàn 0 hoặc 1 (không có sự biến thiên) thì không lấy cột đó
            if indicator.nunique() > 1:
                df[f'is_nan_{col}'] = indicator
            
        # Lớp 3: Vá lỗi Trục Dọc (Longitudinal Forward Fill)
        # Đảm bảo sắp xếp chuẩn trước khi ffill (Memory efficient)
        df.sort_values(by=['code', 'horizon', 'ts_index'], inplace=True)
        gc.collect()
        
        df[features_with_na] = df.groupby(['code', 'horizon'])[features_with_na].ffill(limit=50)
        gc.collect()
        
        # Lớp 4: Khắc phục Cold Start bằng Trung vị (Cross-sectional Median Fill)
        na_after_l3 = df[features_with_na].isna().sum()
        features_still_na = na_after_l3[na_after_l3 > 0].index.tolist()
        
        if features_still_na:
            # Điền trung vị theo mặt cắt ts_index và horizon
            medians_ts_horizon = df.groupby(['ts_index', 'horizon'])[features_still_na].transform('median')
            df[features_still_na] = df[features_still_na].fillna(medians_ts_horizon)
            del medians_ts_horizon
            gc.collect()
            
        # Lớp 5: Hậu xử lý - Loại bỏ các cột hằng số (toàn 0 hoặc toàn 1)
        # Loại bỏ các features và indicators không mang thông tin
        cols_to_check = [c for c in df.columns if c in features_list or c.startswith('is_nan_')]
        constant_cols = [c for c in cols_to_check if df[c].nunique() <= 1]
        
        if constant_cols:
            print(f"Bổ sung: Đã loại bỏ {len(constant_cols)} cột hằng số (toàn 0 hoặc 1).")
            df = df.drop(columns=constant_cols)
            
        print(f"HOÀN THÀNH XỬ LÝ NAN: {time.time() - start_time:.2f}s")
        return df

def main():
    print(f"Reading raw data from {INPUT_FILE}...")
    start_time = time.time()
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find raw data at {INPUT_FILE}")
        sys.exit(1)
        
    df = pd.read_parquet(INPUT_FILE)
    
    # Metadata columns
    meta_cols = ['id', 'y_target', 'weight', 'ts_index', 'code', 'sub_code', 'sub_category', 'horizon']
    features = [c for c in df.columns if c not in meta_cols]
    
    handler = NaN_Handler()
    df = handler.process(df, features)
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Hoàn thành! {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
