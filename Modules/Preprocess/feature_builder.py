import pandas as pd
import numpy as np
import time
import sys
import os
import gc
from joblib import Parallel, delayed  # Used by inject_cross_sectional_zscore

# Paths
# BASE_DIR should be f:\Projects\hege_fund_kaggle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train_logreturn.parquet')
RAW_DATA = os.path.join(DATA_DIR, 'train.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_feature_builded.parquet')

class AlphaGenerator:
    """
    Module tạo Phái sinh Alpha (Feature Engineering).
    Bảo đảm tuyệt đối nguyên tắc Nhân quả (Causality) và Không Rò rỉ (Zero-Leakage).
    """
    def __init__(self, lag_features=['feature_al_logret', 'feature_am_logret', 'feature_u_logret'], lags=[1, 3, 5, 10, 25]):
        self.lag_features = lag_features
        self.lags = lags

    def inject_lagged_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n--- BẮT ĐẦU: CẤY GHÉP TRÍ NHớ TRỤC DỌC (LAGGED MOMENTUM) ---")
        start_time = time.time()
        
        df = df.sort_values(['sub_category', 'code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
        
        # Dùng groupby.shift() chuẩn của pandas: đúng về mặt toán học, không có rủi ro race condition
        grouped = df.groupby(['sub_category', 'code', 'sub_code', 'horizon'], sort=False)
        
        new_cols = {}
        valid_cols = [col for col in self.lag_features if col in df.columns]
        missing = set(self.lag_features) - set(valid_cols)
        if missing:
            print(f"   ⚠ CẢNH BÁO: Không tìm thấy các cột lag: {missing}")
        
        total_tasks = len(valid_cols) * len(self.lags)
        done = 0
        for col in valid_cols:
            for lag in self.lags:
                shifted = grouped[col].shift(lag).astype(np.float32)
                new_cols[f"{col}_lag_{lag}"] = shifted.values
                done += 1
                if done % 5 == 0 or done == total_tasks:
                    print(f"   [{done:2}/{total_tasks}] lag features computed")
        
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            # NaN Report: phân biệt NaN ranh giới (bình thường) vs NaN bất thường
            total_nan = new_df.isna().sum().sum()
            n_groups = grouped.ngroups
            expected_boundary_nan = sum(lag * n_groups for lag in self.lags) * len(valid_cols)
            extra_nan = max(0, total_nan - expected_boundary_nan)
            print(f"   NaN Report (Lagged Momentum):")
            print(f"     - Tổng NaN trong {len(new_cols)} cột lag: {total_nan:,}")
            print(f"     - Ước tính NaN ranh giới group: ~{expected_boundary_nan:,} (bình thường)")
            if extra_nan > 0:
                print(f"     - ⚠ NaN bất thường ngoài ranh giới: {extra_nan:,}")
            df = pd.concat([df, new_df], axis=1)
            del new_cols, new_df
            
        print(f" Thời gian thực thi: {time.time() - start_time:.2f}s")
        gc.collect()
        return df
    
    def inject_cross_sectional_zscore(self, df: pd.DataFrame, target_features=['feature_k_logret', 'feature_at_logret'], rawdata=None) -> pd.DataFrame:
        """
        Module Cấy ghép Sức mạnh Trục ngang (Sector-Neutral Cross-Sectional Z-Score).
        
        Công năng (Functionality):
        - Chuyển đổi giá trị tuyệt đối của các đặc trưng (features) thành Điểm Z (Z-Score) tương đối.
        - Việc chuẩn hóa được cô lập nghiêm ngặt tại cùng một mốc thời gian (ts_index) và 
        cùng một nhóm ngành (sub_category).
        - Tự động bọc lỗi toán học (Zero-Division/NaN) cho các nhóm ngành thiếu thanh khoản (chỉ có 1 mã).
        
        Ý nghĩa Định lượng (Quantitative Significance):
        1. Khử nhiễu Vĩ mô (Macro-Neutralization): Rửa sạch tác động của xu hướng thị trường chung (Beta), 
        ngăn chặn hiện tượng mọi mã đều tăng/giảm do tin tức vĩ mô.
        2. Tách lọc Alpha (Alpha Isolation): Ép thuật toán Cây Quyết định (LightGBM) ngừng học thuộc lòng 
        các giá trị tuyệt đối vô nghĩa, chuyển sang học "Sức mạnh Tương đối" (Relative Strength) của 
        tài sản so với các đối thủ cùng cân nặng.
        3. Bảo chứng Không Rò rỉ (Zero-Leakage Guarantee): Phép biến đổi mang tính Nhân quả (Causal) 100%, 
        chỉ sử dụng thông tin lát cắt ngang tại thời điểm t, tuyệt đối không nhìn trộm tương lai.
        """
        print("\n--- BẮT ĐẦU: CẤY GHÉP SỨC MẠNH TRỤC NGANG (CROSS-SECTIONAL Z-SCORE - Parallel) ---")
        start_time = time.time()
        
        def calculate_zscore(col_name):
            # Tính toán mean/std bằng agg rồi merge (nhanh hơn transform cho dữ liệu lớn)
            stats = df.groupby(['ts_index', 'sub_category'])[col_name].agg(['mean', 'std'])
            
            # Map lại dữ liệu vào dataframe gốc qua index
            # Dùng join/map để gán giá trị nhanh
            merged_stats = df[['ts_index', 'sub_category']].join(stats, on=['ts_index', 'sub_category'])
            
            g_mean = merged_stats['mean'].values
            g_std = merged_stats['std'].values
            
            # Xử lý std an toàn
            safe_std = np.where(np.isnan(g_std) | (g_std == 0), 1.0, g_std)
            z_score = (df[col_name].values - g_mean) / safe_std
            
            return f"{col_name}_sector_zscore", z_score.astype(np.float32)

        print(f"   - Đang tính Z-Score song song cho {len(target_features)} features...")
        z_results = Parallel(n_jobs=-1)(delayed(calculate_zscore)(c) for c in target_features if c in df.columns)
        
        if z_results:
            new_z_cols = {name: res for name, res in z_results}
            df = pd.concat([df, pd.DataFrame(new_z_cols, index=df.index)], axis=1)
            del new_z_cols, z_results

        # Merge Raw Data (Optimized)
        if rawdata and os.path.exists(rawdata):
            print(f"   - Nối với raw data từ: {rawdata}")
            raw_columns = pd.read_parquet(rawdata, columns=None).columns # Get column list first
            cols_to_add = [c for c in raw_columns if c not in df.columns]
            
            if cols_to_add:
                raw_df = pd.read_parquet(rawdata, columns=['id'] + cols_to_add)
                df = pd.merge(df, raw_df, on='id', how='left')
                del raw_df
                
            if all(col in df.columns for col in ['code', 'sub_code', 'horizon', 'ts_index']):
                df.sort_values(['code', 'sub_code', 'horizon', 'ts_index'], inplace=True)
                df.reset_index(drop=True, inplace=True)
            
        print(f" Thời gian thực thi: {time.time() - start_time:.2f}s")
        gc.collect()
        return df

def main(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Could not find raw data at {file_path}")
        sys.exit(1)
        
    df = pd.read_parquet(file_path)
    print(f"Bắt đầu xử lý dữ liệu từ {file_path}. Shape: {df.shape}")

    # Khởi tạo AlphaGenerator và thực hiện Feature Engineering
    alpha_gen = AlphaGenerator()
    
    # 1. Cấy ghép Trí nhớ Trục dọc (Lagged Momentum)
    df = alpha_gen.inject_lagged_momentum(df)
    
    # 2. Cấy ghép Sức mạnh Trục ngang (Sector-Neutral Z-Score)
    # df = alpha_gen.inject_cross_sectional_zscore(df)

    # Kiểm tra NA sau khi build features
    print("\n--- KIỂM TRA GIÁ TRỊ TRỐNG (NA CHECK) ---")
    na_counts = df.isna().sum()
    total_nas = na_counts.sum()

    if total_nas == 0:
        print("Success: Không có giá trị NA sau khi xử lý.")
    else:
        print(f"Cảnh báo: Phát hiện {total_nas} giá trị NA.")
        cols_with_nas = na_counts[na_counts > 0]
        print("Các cột chứa NA:")
        print(cols_with_nas)
        print("Lưu ý: Các giá trị NA ở cột 'lag' là bình thường do phép trễ đầu chuỗi.")

    print(f"\nLưu dữ liệu đã xử lý vào {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    print("Hoàn tất!")

if __name__ == '__main__':
    main(INPUT_FILE)