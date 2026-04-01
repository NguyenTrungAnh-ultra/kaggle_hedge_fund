import pandas as pd
import numpy as np
import os
import time
import gc

class LogReturn_Handler:
    """
    Module thực hiện Log Return (Log-Differencing) cho các biến toàn dương.
    Đảm bảo: Zero Data Leakage, Cross-sectional Isolation, Vectorization.
    Công thức: X'_t = log(X_t + 1) - log(X_{t-1} + 1)
    """
    def __init__(self, lag=1):
        self.lag = lag

    def process(self, df: pd.DataFrame, features_list: list) -> pd.DataFrame:
        print(f"\n--- BẮT ĐẦU XỬ LÝ LOG RETURN: {len(features_list)} FEATURES ---")
        start_time = time.time()
        
        # 1. Sắp xếp để đảm bảo tính tuần tự của chuỗi thời gian cho mỗi nhóm (Crucial for Shift)
        # Sắp xếp theo cấu trúc phân cấp: sub_category -> code -> sub_code -> horizon -> ts_index
        df = df.sort_values(['sub_category', 'code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
        
        # 2. Tạo Grouped object để thực hiện shift an toàn (tránh rò rỉ giữa các mã cổ phiếu)
        # Sử dụng groupby để thực hiện shift trên toàn bộ dataframe một lần
        grouped = df.groupby(['sub_category', 'code', 'sub_code', 'horizon'], sort=False)
        
        # Danh sách các cột mới để tối ưu việc gán
        new_cols_dict = {}
        
        nan_clipped_cols = []
        for idx, col in enumerate(features_list, 1):
            col_start = time.time()
            
            raw_vals = df[col].values.astype(np.float32)
            
            # Guard #3: Kiểm tra và clip giá trị âm trước log1p
            # log1p(x) = log(x+1): nếu x < -1 → NaN im lặng; nếu x < 0 → kết quả âm (không hợp lệ cho biến "toàn dương")
            neg_mask = (raw_vals < 0) & ~np.isnan(raw_vals)
            if neg_mask.any():
                n_neg = neg_mask.sum()
                nan_clipped_cols.append((col, int(n_neg)))
                raw_vals = np.where(neg_mask, 0.0, raw_vals)  # Clip về 0 thay vì để NaN
            
            log_curr = np.log1p(raw_vals)
            
            # Tính Log(X_{t-1} + 1) bằng cách shift theo nhóm
            # Phải shift TRƯỚC khi lấy log để đảm bảo logic thời gian chuẩn
            shifted_vals = grouped[col].shift(self.lag).values.astype(np.float32)
            # Guard: clip shifted cũng có thể âm sau ffill hoặc impute
            neg_shifted = (shifted_vals < 0) & ~np.isnan(shifted_vals)
            if neg_shifted.any():
                shifted_vals = np.where(neg_shifted, 0.0, shifted_vals)
            log_prev = np.log1p(shifted_vals)
            
            # Log Return (NaN tại ranh giới group từ shift là bình thường)
            new_cols_dict[f"{col}_logret"] = (log_curr - log_prev)
            
            if idx % 20 == 0 or idx == len(features_list):
                print(f"   [{idx:2}/{len(features_list)}] Created {col}_logret ({time.time() - col_start:.2f}s)")
        
        if nan_clipped_cols:
            print(f"   ⚠ CẢNH BÁO: {len(nan_clipped_cols)} cột có giá trị âm bị clip về 0 trước log1p:")
            for cname, cnt in nan_clipped_cols:
                print(f"     - {cname}: {cnt} giá trị âm")
        
        # Gán tất cả các cột mới một lượt để tránh fragmented dataframe và tiết kiệm RAM
        print(f"   Đang gán {len(new_cols_dict)} cột mới vào DataFrame...")
        new_df = pd.DataFrame(new_cols_dict, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
        # Report #4: NaN propagation summary sau log return
        logret_cols = list(new_cols_dict.keys())
        nan_counts = new_df.isna().sum()
        total_nan = nan_counts.sum()
        n_boundary_nan = len(features_list) * df.groupby(['sub_category', 'code', 'sub_code', 'horizon'], sort=False).ngroups * self.lag
        print(f"   NaN Report (Log Return):")
        print(f"     - Tổng NaN trong {len(logret_cols)} cột _logret: {total_nan:,}")
        print(f"     - Ước tính NaN ranh giới group (lag={self.lag}): ~{n_boundary_nan:,} (bình thường)")
        extra_nan = max(0, total_nan - n_boundary_nan)
        if extra_nan > 0:
            print(f"     - ⚠ NaN bất thường ngoài ranh giới: {extra_nan:,} → kiểm tra dữ liệu đầu vào")
        
        # Dọn dẹp RAM
        del new_cols_dict, new_df
        gc.collect()
        
        print(f"HOÀN THÀNH XỬ LÝ LOG RETURN: {time.time() - start_time:.2f}s")
        return df

if __name__ == "__main__":
    # Script chuẩn hóa đầu ra giống make_stationary.py
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
    INPUT_FILE = os.path.join(DATA_DIR, 'train_handled_na.parquet')
    OUTPUT_FILE = os.path.join(DATA_DIR, 'train_logreturn.parquet')
    
    if os.path.exists(INPUT_FILE):
        print(f"Đọc dữ liệu từ: {INPUT_FILE}")
        df = pd.read_parquet(INPUT_FILE)
        
        # Tự động nhận diện 69 biến toàn dương (min >= 0 và max > 0)
        # Loại trừ các cột meta và is_nan
        meta_cols = ['id', 'y_target', 'weight', 'ts_index', 'is_test', 'code', 'sub_code', 'sub_category', 'horizon']
        pos_features = [c for c in df.columns if c.startswith('feature_') 
                        and not c.startswith('is_nan')
                        and c not in meta_cols
                        and df[c].min() >= 0]
        
        print(f"Phát hiện {len(pos_features)} biến toàn dương để xử lý Log Return.")
        
        handler = LogReturn_Handler(lag=1)
        df_processed = handler.process(df, pos_features)
        
        print(f"Đang lưu kết quả vào: {OUTPUT_FILE}")
        df_processed.to_parquet(OUTPUT_FILE, index=False)
        print(f"Hoàn thành! New shape: {df_processed.shape}")
    else:
        print(f"Không tìm thấy file: {INPUT_FILE}")
