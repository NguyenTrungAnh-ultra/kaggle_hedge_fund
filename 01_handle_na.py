import pandas as pd
import numpy as np
import os
import time
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_handled_na.parquet')
REPORT_FILE = os.path.join(BASE_DIR, 'preprocess_na_report.md')

def main():
    print(f"Reading raw data from {INPUT_FILE}...")
    start_time = time.time()
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find raw data at {INPUT_FILE}")
        sys.exit(1)
        
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns in {time.time() - start_time:.2f}s")
    
    # Define metadata columns so we only fill NaNs on features
    meta_cols = ['id', 'y_target', 'weight', 'ts_index', 'code', 'sub_code', 'sub_category', 'horizon']
    features = [c for c in df.columns if c not in meta_cols]
    
    print("\n--- Phân tích NaN ban đầu ---")
    na_before = df[features].isna().sum()
    features_with_na = na_before[na_before > 0].index.tolist()
    
    print(f"Found {len(features_with_na)} features with NaNs out of {len(features)} total features.")
    
    if not features_with_na:
        print("No NaNs found in features. Exiting without changes.")
        return
        
    # --- Lớp 1: Cách ly Không gian (Spatial Isolation) ---
    print("\n--- Lớp 1: Cách ly Không gian (Spatial Isolation) ---")
    print("Xác nhận nguyên tắc: Các hành động groupby sẽ luôn có 'horizon' đóng vai trò phân mảnh ngang.")

    # --- Lớp 2: Đánh dấu Khuyết thiếu (Missingness Indicator / Shadow Matrix) ---
    print("\n--- Lớp 2: Đánh dấu Khuyết thiếu (Missingness Indicator) ---")
    start_layer2 = time.time()
    for col in features_with_na:
        df[f'is_nan_{col}'] = df[col].isna().astype(np.int8)
    print(f"Tạo {len(features_with_na)} cột cờ nhị phân 'is_nan_...' mất {time.time() - start_layer2:.2f}s")
    
    # --- Lớp 3: Vá lỗi Trục Dọc (Longitudinal Forward Fill) ---
    print("\n--- Lớp 3: Vá lỗi Trục Dọc (Longitudinal Forward Fill) ---")
    start_layer3 = time.time()
    # Sort data directly before groupby-ffill
    print("Sắp xếp theo 'code', 'horizon', 'ts_index'...")
    df = df.sort_values(by=['code', 'horizon', 'ts_index'])
    
    print("Thực hiện ffill() với giới hạn limit=50...")
    # Apply ffill per 'code' and 'horizon' safely
    df[features_with_na] = df.groupby(['code', 'horizon'])[features_with_na].ffill(limit=50)
    print(f"Thực hiện Lớp 3 mất {time.time() - start_layer3:.2f}s")
    
    # --- Lớp 4: Khắc phục Cold Start bằng Trung vị (Cross-sectional Median Fill) ---
    print("\n--- Lớp 4: Khắc phục Cold Start (Median Fill) ---")
    start_layer4 = time.time()
    
    # Find features that STILL have NaNs after Layer 3 to optimize computation
    na_after_l3 = df[features_with_na].isna().sum()
    features_still_na = na_after_l3[na_after_l3 > 0].index.tolist()
    
    if features_still_na:
        print(f"Còn {len(features_still_na)} features chứa NaN. Khắc phục bằng Groupby(['ts_index', 'horizon']) median...")
        
        # We need to fill remaining NaNs in each target column with the spatial median for that ['ts_index', 'horizon']
        medians_ts_horizon = df.groupby(['ts_index', 'horizon'])[features_still_na].transform('median')
        df[features_still_na] = df[features_still_na].fillna(medians_ts_horizon)
        
        # Verify if any entire cross-sections were empty
        na_after_l4_1 = df[features_still_na].isna().sum()
        features_still_na_2 = na_after_l4_1[na_after_l4_1 > 0].index.tolist()
        
        if features_still_na_2:
            print("CẢNH BÁO LUỒNG LOGIC: Vẫn còn nhóm NaN diện rộng (entire cross-section bị NaN toàn bộ).")
            print("Do yêu cầu CHỐNG RÒ RỈ TƯƠNG LAI, hệ thống KHÔNG TỰ ĐỘNG DÙNG MEDIAN TOÀN CỤC để lấp nội suy.")
            print("Các giá trị này sẽ được giữ nguyên là NaN để chờ người dùng cùng xử lý.")
    
    print(f"Thực hiện Lớp 4 mất {time.time() - start_layer4:.2f}s")
    
    # --- Final Check ---
    na_final = df[features_with_na].isna().sum().sum()
    print(f"\nFinal NaN count in original features: {na_final}")
    if na_final > 0:
        print("CẢNH BÁO: Vẫn còn dòng NaN trong dữ liệu. Quy trình có lỗi logic!")
    else:
        print("THÀNH CÔNG: Đã làm sạch toàn bộ NaN theo cấu trúc thời gian khép kín.")
        
    print(f"\nSắp xếp lại theo [ts_index, code, sub_code] và cấu trúc ban đầu...")
    df = df.sort_values(['ts_index', 'code', 'sub_code', 'horizon']).reset_index(drop=True)
    
    # Kiểm tra metadata và shape
    print(f"Cấu trúc dữ liệu sau xử lý: {len(df)} dòng, {len(df.columns)} cột.")
    
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"Hoàn thành toàn bộ pipeline! Tổng thời gian: {time.time() - start_time:.2f}s")
    
    # --- Report Output ---
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Báo Cáo Preprocessing Pipeline (Xử lý NaN chuẩn mực)\n\n")
        f.write("## Hành Động Đã Thực Hiện Theo 4 Lớp\n\n")
        f.write("Dựa trên các chỉ định nghiêm ngặt, dữ liệu raw đã được rà soát với thuật toán chống rò rỉ dữ liệu.\n")
        f.write("### Lớp 1: Cách ly Không gian (Spatial Isolation)\n")
        f.write("- Mọi tính toán tập hợp đều sử dụng `horizon` làm biến phân mảnh ngang. Đảm bảo dữ liệu các chân trời dự báo không bị nội suy chéo cho nhau.\n")
        f.write(f"### Lớp 2: Đánh dấu Khuyết thiếu (Missingness Indicator)\n")
        f.write(f"- Đã tạo thành công {len(features_with_na)} cột cờ nhị phân (tiền tố `is_nan_`) cho toàn bộ các đặc trưng bị khuyết để giúp LightGBM học được mẫu hình đứt gãy hệ thống.\n")
        f.write("### Lớp 3: Vá lỗi Trục Dọc (Longitudinal Forward Fill)\n")
        f.write("- Sử dụng `df.sort_values(by=['code', 'horizon', 'ts_index'])` và áp dụng `groupby(['code', 'horizon']).ffill(limit=50)`.\n")
        f.write("- Giải quyết triệt để các nan-gap ở phạm vi giữa mà không gây rò rỉ hiện trạng tương lai (mỗi giá trị chỉ nối tiếp quá khứ 50 node).\n")
        f.write("### Lớp 4: Khắc phục Cold Start (Median Fill)\n")
        f.write("- Sử dụng `groupby(['ts_index', 'horizon']).transform('median')` để lấy trung vị từng mặt cắt theo thời gian, điền cho các NaN còn sót ở các đoạn đứt gãy đầu chuỗi.\n")
        f.write("- Trong trường hợp mặt cắt thời gian bị rỗng NaN 100%, hệ thống sẽ giữ nguyên là NaN và cảnh báo ra màn hình để tránh Data Leakage (không dùng median toàn cục).\n\n")
        f.write("## Kết Quả Chống Kiểm Thống Kê\n")
        f.write(f"- **Volume Khởi Thủy**: {len(df)} dòng.\n")
        f.write(f"- **Features Bị Khuyết Thiết Ban Đầu**: {len(features_with_na)} cột.\n")
        f.write(f"- **Khối Lượng Features Sinh Thêm**: {len(features_with_na)} cờ nhị phân `is_nan_*`.\n")
        f.write(f"- **NaN Tồn Đọng Cuối Cùng (Original Features)**: {na_final}.\n")
        f.write(f"- **Kích Thước Dữ Liệu Sau Xử Lý**: {len(df.columns)} cột (đã bảo toàn nguyên trạng dữ liệu và thêm shadow indicators).\n")
        f.write(f"- **File Xuất**: `{OUTPUT_FILE}`.\n")

if __name__ == '__main__':
    main()
