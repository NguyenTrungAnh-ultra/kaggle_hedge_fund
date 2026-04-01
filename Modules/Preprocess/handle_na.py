import pandas as pd
import numpy as np
import os
import time
import sys
import gc
from joblib import Parallel, delayed

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
    def _generate_indicators(self, df_values, features_list):
        """Hàm helper để tạo indicators song song"""
        def process_col(col_idx):
            col_name = features_list[col_idx]
            vals = df_values[:, col_idx]
            mask = np.isnan(vals)
            if np.any(mask) and not np.all(mask): # Chỉ tạo nếu có NaN và không phải toàn bộ là NaN
                return f'is_nan_{col_name}', mask.astype(np.int8)
            return None

        print(f"   - Đang tạo Missingness Indicators song song...")
        results = Parallel(n_jobs=-1)(delayed(process_col)(i) for i in range(len(features_list)))
        return {res[0]: res[1] for res in results if res is not None}

    def _impute_values(self, df: pd.DataFrame, features_list: list) -> pd.DataFrame:
        """Hàm riêng để xử lý điền giá trị (Imputation) - Có thể bật tắt"""
        # Lớp 3: Longitudinal Forward Fill
        print(f"   - [IMPUTE] Lớp 3: Đang thực hiện Forward Fill theo Group...")
        df.sort_values(by=['code', 'sub_code', 'horizon', 'ts_index'], inplace=True)
        df[features_list] = df.groupby(['code', 'sub_code', 'horizon'], sort=False)[features_list].ffill(limit=20)
        gc.collect()

        # Lớp 4: Cross-sectional Median Fill
        print(f"   - [IMPUTE] Lớp 4: Đang thực hiện Median Fill theo mặt cắt thời gian...")
        na_mask = df[features_list].isna()
        if na_mask.any().any():
            medians = df.groupby(['ts_index', 'horizon'])[features_list].median()
            df.set_index(['ts_index', 'horizon'], inplace=True)
            df[features_list] = df[features_list].fillna(medians)
            df.reset_index(inplace=True)
            del medians
            gc.collect()
        return df

    def process(self, df: pd.DataFrame, features_list: list, impute: bool = False) -> pd.DataFrame:
        print(f"\n--- BẮT ĐẦU XỬ LÝ NAN: NAN HANDLER (Parallel Optimized) ---")
        start_time = time.time()
        
        # Bước 1: Phân tích và tạo Indicators song song (Luôn chạy để model biết chỗ nào bị NaN)
        indicators_dict = self._generate_indicators(df[features_list].values, features_list)
        if indicators_dict:
            # Dùng concat để tránh SettingWithCopyWarning và nhanh hơn
            df = pd.concat([df, pd.DataFrame(indicators_dict, index=df.index)], axis=1)
        
        # Bước 2: Chỉ thực hiện điền giá trị nếu được yêu cầu (Mặc định là False)
        if impute:
            df = self._impute_values(df, features_list)
        else:
            print(f"   - Bỏ qua bước Imputation (Lớp 3 & 4) theo yêu cầu người dùng.")

        # Bước 3: Hậu xử lý - Loại bỏ cột hằng số nhanh bằng NumPy
        print(f"   - Lớp 5: Loại bỏ các cột không mang thông tin...")
        cols_to_check = [c for c in df.columns if c in features_list or c.startswith('is_nan_')]
        
        def is_constant(col_data):
            # Dùng numpy check nhanh hơn nunique
            c_data = col_data.values
            if c_data.dtype.kind in 'fi': # float or int
                non_nan = c_data[~np.isnan(c_data)]
                if len(non_nan) == 0: return True
                return np.min(non_nan) == np.max(non_nan)
            return len(np.unique(c_data)) <= 1

        results = Parallel(n_jobs=-1)(delayed(is_constant)(df[c]) for c in cols_to_check)
        constant_cols = [cols_to_check[i] for i, const in enumerate(results) if const]
        
        if constant_cols:
            print(f"     -> Đã loại bỏ {len(constant_cols)} cột hằng số (phát hiện động).")
            df.drop(columns=constant_cols, inplace=True)
            
        print(f"HOÀN THÀNH XỬ LÝ NAN: {time.time() - start_time:.2f}s")
        print(f"Datashape: {df.shape}")
        print(f"Tổng NaN còn lại: {df.isna().sum().sum()}")
        gc.collect()
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
