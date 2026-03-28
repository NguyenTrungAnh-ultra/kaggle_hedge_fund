import pandas as pd
import numpy as np
import time
import sys
import os

# Paths
# BASE_DIR should be f:\Projects\hege_fund_kaggle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train_stationary.parquet')
RAW_DATA = os.path.join(DATA_DIR, 'train.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_feature_builded.parquet')

class AlphaGenerator:
    """
    Module tạo Phái sinh Alpha (Feature Engineering).
    Bảo đảm tuyệt đối nguyên tắc Nhân quả (Causality) và Không Rò rỉ (Zero-Leakage).
    """
    def __init__(self, lag_features=['feature_al_sta', 'feature_am_sta', 'feature_cg_sta', 'feature_by_sta'], lags=[1, 3, 5, 10, 25]):
        self.lag_features = lag_features
        self.lags = lags

    def inject_lagged_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Đại loại nó tạo thêm cột ['feature_al', 'feature_am', 'feature_cg', 'feature_by'] nhưng shift lên [1, 3, 5, 10, 25]
        """
        print("\n--- BẮT ĐẦU: CẤY GHÉP TRÍ NHỚ TRỤC DỌC (LAGGED MOMENTUM) ---")
        start_time = time.time()
        
        # BƯỚC 1: Khóa chặt Tọa độ Không-Thời gian (Absolute Sorting)
        # Bắt buộc phải sắp xếp để .shift() không bị trượt nhầm mã
        df = df.sort_values(['code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
        
        # BƯỚC 2: Khởi tạo ranh giới vách ngăn (Groupby Boundaries)
        grouped = df.groupby(['code', 'sub_code', 'horizon'])
        
        # BƯỚC 3: Kích hoạt Phép toán Nhân quả (Causal Shifting)
        new_cols = {}
        for col in self.lag_features:
            if col not in df.columns:
                print(f" Cảnh báo: {col} không tồn tại trong tập dữ liệu.")
                continue
            
            for lag in self.lags:
                new_col_name = f"{col}_lag_{lag}"
                # Lấy giá trị của 'lag' bước trước đó đắp vào hiện tại
                new_cols[new_col_name] = grouped[col].shift(lag).astype(np.float32)
                
        # Nối các cột mới vào dataframe gốc (dùng pd.concat để tối ưu RAM thay vì gán từng cột)
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
                
        print(f" Đã tạo thành công {len(new_cols)} đặc trưng trễ. Tối ưu hóa float32.")
        print(f" Tổng NaN tại cuối lớp Trí Nhớ Trục Dọc: {df.isna().sum().sum()}")
        print(f" Thời gian thực thi: {time.time() - start_time:.2f}s")
        
        return df
    
    def inject_cross_sectional_zscore(self, df: pd.DataFrame, target_features=['feature_k_sta', 'feature_at_sta', 'feature_by_sta'], rawdata=RAW_DATA) -> pd.DataFrame:
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
        print("\n--- BẮT ĐẦU: CẤY GHÉP SỨC MẠNH TRỤC NGANG (SECTOR-NEUTRAL Z-SCORE) ---")
        start_time = time.time()
        
        # BƯỚC 1: Khởi tạo vách ngăn Không-Thời gian (Spatio-Temporal Boundaries)
        # Nhóm theo CÙNG MỘT thời điểm (ts_index) và CÙNG MỘT nhóm ngành (sub_category)
        grouped = df.groupby(['ts_index', 'sub_category'])
        
        new_cols = {}
        for col in target_features:
            if col not in df.columns:
                continue
                
            new_col_name = f"{col}_sector_zscore"
            
            # BƯỚC 2: Tính toán Thống kê Cục bộ (Local Statistics)
            # Lưu ý quan trọng: Pandas .transform('mean') và .transform('std') TỰ ĐỘNG BỎ QUA NaN.
            group_mean = grouped[col].transform('mean')
            group_std = grouped[col].transform('std')
            
            # BƯỚC 3: Chuẩn hóa Z-Score và Xử lý Lỗi Chia cho 0 (Zero-Division Safe)
            # Nhóm 1 phần tử thì std = NaN -> Gán thành 1.0 để tránh phép chia cho NaN
            safe_std = group_std.fillna(1.0)
            # Nếu tất cả giá trị bằng nhau thì std = 0 -> Gán thành 1.0
            safe_std = np.where(safe_std == 0, 1.0, safe_std)
            
            # Tính z-score: Nếu df[col] vốn là NaN thì z-score sẽ giữ nguyên là NaN (Bảo toàn Native NaN)
            z_score = (df[col] - group_mean) / safe_std
            
            new_cols[new_col_name] = z_score.astype(np.float32)
            
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            
        if rawdata and os.path.exists(rawdata):
            print(f" Nối với raw data (phục hồi các cột bị rớt) từ: {rawdata}")
            raw_df = pd.read_parquet(rawdata)
            
            cols_to_add = [c for c in raw_df.columns if c not in df.columns]
            
            if 'id' in df.columns and 'id' in raw_df.columns:
                # Dùng left join để giữ nguyên dữ liệu gốc của pipeline df hiện tại (gồm cả modified features)
                # và chỉ lấy thêm các cột missing từ raw_df tương ứng.
                df = pd.merge(df, raw_df[['id'] + cols_to_add], on='id', how='left')
            else:
                raw_df = raw_df.sort_values(['code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
                # Fallback ghép nối trực tiếp nếu không thể nối qua id.
                df = pd.concat([df, raw_df[cols_to_add]], axis=1)
                
            # Đảm bảo trục thời gian không bị lệch sau khi merge
            if all(col in df.columns for col in ['code', 'sub_code', 'horizon', 'ts_index']):
                df.sort_values(['code', 'sub_code', 'horizon', 'ts_index'], inplace=True)
                df.reset_index(drop=True, inplace=True)
            
        print(f" Đã tạo thành công {len(new_cols)} đặc trưng Z-Score chéo nhóm ngành.")
        print(f" Tổng NaN tại cuối lớp Z-Score & Merge Raw Data: {df.isna().sum().sum()}")
        print(f" Data Shape: {df.shape}")
        print(f" Thời gian thực thi: {time.time() - start_time:.2f}s")
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
    df = alpha_gen.inject_cross_sectional_zscore(df)

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