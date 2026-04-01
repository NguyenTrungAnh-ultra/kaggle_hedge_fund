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
        
        for idx, col in enumerate(features_list, 1):
            col_start = time.time()
            
            # Tính Log(X_t + 1) - Sử dụng float32 để tiết kiệm RAM
            # Thêm epsilon nhỏ hoặc 1 để tránh log(0)
            log_curr = np.log1p(df[col].values.astype(np.float32))
            
            # Tính Log(X_{t-1} + 1) bằng cách shift theo nhóm
            # Phải shift TRƯỚC khi lấy log để đảm bảo logic thời gian chuẩn
            shifted_vals = grouped[col].shift(self.lag).values.astype(np.float32)
            log_prev = np.log1p(shifted_vals)
            
            # Log Return
            new_cols_dict[f"{col}_logret"] = (log_curr - log_prev)
            
            if idx % 20 == 0 or idx == len(features_list):
                print(f"   [{idx:2}/{len(features_list)}] Created {col}_logret ({time.time() - col_start:.2f}s)")
        
        # Gán tất cả các cột mới một lượt để tránh fragmented dataframe và tiết kiệm RAM
        print(f"   Đang gán {len(new_cols_dict)} cột mới vào DataFrame...")
        df = pd.concat([df, pd.DataFrame(new_cols_dict, index=df.index)], axis=1)
        
        # Dọn dẹp RAM
        del new_cols_dict
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
