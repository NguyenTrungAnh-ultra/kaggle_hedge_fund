import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

# BASE_DIR should be f:\Projects\hege_fund_kaggle
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train_handled_na.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_stationary.parquet')
REPORT_FILE = os.path.join(BASE_DIR, 'reports', 'report_COMPARISON.csv')


class Seasonal_Handler:
    """
    Module Tiền xử lý (Pre-processing) xử lý Mùa vụ (Seasonal Stationarity).
    Tuân thủ quy tắc: Zero Data Leakage, Cross-sectional Isolation, Vectorization.
    """
    def __init__(self, m_candidates=[5, 22], acf_threshold=0.85):
        self.m_candidates = m_candidates
        self.acf_threshold = acf_threshold

    def process(self, df: pd.DataFrame, features_list: list) -> pd.DataFrame:
        print(f"\n--- BẮT ĐẦU XỬ LÝ MÙA VỤ: SEASONAL HANDLER ---")
        start_time = time.time()
        
        # Bước 1: Cách ly Không gian (Cross-sectional Isolation) & Đảm bảo thứ tự thời gian chuẩn
        df = df.sort_values(['code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
        
        print("Ensuring AFML compliance: Calculating ACF ONLY on the 80% chronological train set to prevent Data Leak...")
        unique_ts = np.sort(df['ts_index'].unique())
        split_ts_idx = int(len(unique_ts) * 0.80)
        split_time = unique_ts[split_ts_idx]
        train_mask_global = (df['ts_index'] < split_time).values
        
        grouped = df.groupby(['code', 'sub_code', 'horizon'], sort=False)
        group_sizes = grouped.size()
        g_ids = grouped.ngroup().values
        
        # Calculate sizes only on train data for valid gates
        train_group_sizes = pd.Series(np.bincount(g_ids[train_mask_global], minlength=len(group_sizes)), index=group_sizes.index)
        
        for idx, col in enumerate(features_list, 1):
            col_start = time.time()
            
            # Khởi tạo lưu trữ m* tốt nhất và ACF cao nhất cho từng nhóm
            best_m_for_group = pd.Series(0, index=group_sizes.index, dtype=int)
            best_acf_for_group = pd.Series(-1.0, index=group_sizes.index, dtype=float)
            
            for m in self.m_candidates:
                # Bước 2: Cổng Kiểm định Độ dài (Length Gate) DỰA TRÊN TẬP TRAIN
                valid_groups_mask = train_group_sizes > (3 * m)
                valid_groups = group_sizes[valid_groups_mask].index
                
                if len(valid_groups) == 0:
                    continue
                
                # Bước 3: Kiểm định Tự tương quan (ACF Test) sử dụng Vectorization
                # Dịch chuyển nhanh toàn bộ dữ liệu
                shifted_col = grouped[col].shift(m)
                
                # Tính ACF siêu tốc bằng NumPy, CHỈ TRÊN TẬP TRAIN MASK
                col_vals = df[col].values
                shifted_vals = shifted_col.values
                
                mask = ~np.isnan(col_vals) & ~np.isnan(shifted_vals) & train_mask_global
                v_ids = g_ids[mask]
                x = col_vals[mask]
                y = shifted_vals[mask]
                
                num_groups = len(group_sizes)
                # Đếm số phần tử và lấy tổng cho công thức Pearson
                n_count = np.bincount(v_ids, minlength=num_groups)
                sum_x = np.bincount(v_ids, weights=x, minlength=num_groups)
                sum_y = np.bincount(v_ids, weights=y, minlength=num_groups)
                sum_x2 = np.bincount(v_ids, weights=x**2, minlength=num_groups)
                sum_y2 = np.bincount(v_ids, weights=y**2, minlength=num_groups)
                sum_xy = np.bincount(v_ids, weights=x*y, minlength=num_groups)
                
                num = n_count * sum_xy - sum_x * sum_y
                den = np.sqrt(np.maximum((n_count * sum_x2 - sum_x**2) * (n_count * sum_y2 - sum_y**2), 0))
                
                corr_arr = np.where(den > 1e-8, num / den, -1.0)
                
                # Ngroup index map trực tiếp 1-1 với group_sizes.index do sort=False
                corr_series = pd.Series(corr_arr, index=group_sizes.index)
                
                # Lọc ra các nhóm thỏa mãn Length Gate
                corr_series = corr_series.loc[corr_series.index.intersection(valid_groups)]
                
                # Bước 4: Ngưỡng Kích hoạt & Thực thi
                # Chọn m có độ tương quan ACF cao nhất và vượt ngưỡng threshold
                update_mask = (corr_series > self.acf_threshold) & (corr_series > best_acf_for_group.loc[corr_series.index])
                
                groups_to_update = corr_series[update_mask].index
                best_acf_for_group.loc[groups_to_update] = corr_series.loc[groups_to_update]
                best_m_for_group.loc[groups_to_update] = m
                
            # Thực thi Toán tử Dịch chuyển Lùi Mùa vụ (Seasonal Backshift)
            final_col = df[col].copy()
            
            # O(1) Array Mapping: Ánh xạ kết quả m* siêu tốc qua g_ids thay vì Index Mapping chậm chạp
            m_array = best_m_for_group.values[g_ids]
            
            for m in self.m_candidates:
                m_mask = (m_array == m)
                if m_mask.sum() > 0:
                    # Lấy lại array đã shift
                    shifted_m = grouped[col].shift(m)
                    # Phép trừ lùi mùa vụ: X'_t = X_t - X_{t-m*}
                    final_col.loc[m_mask] = df.loc[m_mask, col] - shifted_m.loc[m_mask]
                    
            df[col] = final_col
            
            num_groups_shifted = (best_m_for_group > 0).sum()
            print(f"[{idx}/{len(features_list)}] {col:15} | Nhóm áp dụng Dịch chuyển Mùa vụ: {num_groups_shifted}/{len(group_sizes)} ({time.time() - col_start:.2f}s)")
            
        print(f"HOÀN THÀNH XỬ LÝ MÙA VỤ: {time.time() - start_time:.2f}s")
        return df


class Fractional_Differencer:
    """
    Module Xử lý Dừng phân số (Fractional Differencing) sử dụng Modified Fixed-Width Window (FFW).
    Ước lượng tham số d trên toàn bộ tập dữ liệu để tối ưu độ chính xác và RAM.
    """
    @staticmethod
    def get_weights(d, size):
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] / k * (d - k + 1)
            w.append(w_k)
        w = np.array(w[::-1]).reshape(-1, 1)
        return w

    @staticmethod
    def find_optimal_d(series):
        """
        Tìm tham số d tối ưu (0 -> 1) sao cho chuỗi dữ liệu đạt tính dừng (ADF < 0.05).
        Sử dụng toàn bộ dữ liệu được cung cấp.
        """
        series = series.dropna().values
        if len(series) < 20: 
            return 1.0 
        
        # Thử các giá trị d từ 0 đến 1 với bước nhảy 0.1
        for d in np.linspace(0, 1, 11):
            if d == 0.0:
                diff_series = series
            else:
                w_full = Fractional_Differencer.get_weights(d, len(series))
                w_cumsum = np.cumsum(np.abs(w_full))
                
                # Cắt trọng số nhiễu (FFW - Fixed-Width Window)
                skip = np.searchsorted(w_cumsum / w_cumsum[-1], 1e-2)
                if (len(series) - skip) > 50:
                    skip = len(series) - 50
                    
                w = w_full[skip:]
                if len(w) == 0:
                    w = w_full[-1:]
                    
                w_for_conv = w[::-1].flatten() 
                diff_series = np.convolve(series, w_for_conv, mode='valid')
                
            if len(diff_series) > 10:
                try:
                    # Kiểm định ADF
                    p_val = adfuller(diff_series)[1]
                    if p_val <= 0.05:
                        return float(d)
                except:
                    continue
        return 1.0

    def process(self, df: pd.DataFrame, features_list: list) -> pd.DataFrame:
        import gc
        print(f"\n--- BẮT ĐẦU XỬ LÝ DỪNG PHÂN SỐ: FRACTIONAL DIFFERENCER (RAM Optimized) ---")
        start_time = time.time()
        
        # 1. Đảm bảo sắp xếp chuẩn thời gian
        df = df.sort_values(['code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
        
        # 2. Ước tính d trên TOÀN BỘ tập dữ liệu (Không split)
        print("Đang tìm group lớn nhất trên toàn bộ dataset để ước lượng d...")
        
        # Nhóm dữ liệu để lấy thông tin size
        grouped = df.groupby(['code', 'sub_code', 'horizon'], sort=False)
        group_sizes = grouped.size()
        top_group_keys = group_sizes.idxmax()
        max_size = group_sizes.max()
        
        # Lấy indices của group lớn nhất để tránh copy dataframe lớn
        test_idx = df[(df['code'] == top_group_keys[0]) & 
                      (df['sub_code'] == top_group_keys[1]) & 
                      (df['horizon'] == top_group_keys[2])].index
        
        print(f"Group đại diện: {top_group_keys} | Size: {len(test_idx)}")
        
        # 3. Tính toán trọng số và áp dụng cho từng cột
        for idx, col in enumerate(features_list, 1):
            start_col = time.time()
            
            # 3. Tính toán trọng số và áp dụng cho từng cột
        for idx, col in enumerate(features_list, 1):
            start_col = time.time()
            
            # --- FIX DATA LEAKAGE ---
            # Chỉ ước tính d_opt trên phần dữ liệu thuộc tập Train (is_test == 0)
            if 'is_test' in df.columns:
                train_idx = df[(df['code'] == top_group_keys[0]) & 
                               (df['sub_code'] == top_group_keys[1]) & 
                               (df['horizon'] == top_group_keys[2]) &
                               (df['is_test'] == 0)].index
                test_series = df.loc[train_idx, col]
            else:
                # Fallback nếu đang chạy độc lập trên tập Train
                test_series = df.loc[test_idx, col]
                
            opt_d = self.find_optimal_d(test_series)
            # ------------------------
            
            print(f"[{idx}/{len(features_list)}] {col:15} | d_opt: {opt_d:.1f} ", end='', flush=True)
            
            if opt_d > 0.0:
                w_full = self.get_weights(opt_d, max_size)
                w_cumsum = np.cumsum(np.abs(w_full))
                skip = np.searchsorted(w_cumsum / w_cumsum[-1], 1e-2)
                if (max_size - skip) > 50: skip = max_size - 50
                
                w = w_full[skip:]
                if len(w) > 0:
                    w_for_conv = w[::-1].flatten()
                    window_len = len(w)

                    # Transform vectorization - ép kiểu float32 để tiết kiệm RAM
                    df[col] = grouped[col].transform(lambda g: 
                        np.convolve(g.values, w_for_conv, mode='full')[:len(g)].astype(np.float32) 
                        if len(g) >= window_len else np.full(len(g), np.nan, dtype=np.float32)
                    )
                else:
                    print(f"| Skipping (Empty weights) ", end='')
            
            # Quản lý RAM: dọn dẹp thường xuyên hơn
            if idx % 10 == 0:
                gc.collect()
            
            print(f"| Done ({time.time() - start_col:.2f}s)")
            
        gc.collect()
        print(f"HOÀN THÀNH XỬ LÝ DỪNG PHÂN SỐ: {time.time() - start_time:.2f}s")
        return df


class StationarityPipeline:
    """
    Pipeline chính để điều phối quá trình xử lý Stationarity qua Fractional Differencing. 
    Ước lượng tham số trên toàn bộ tập dữ liệu và tối ưu RAM.
    """
    def __init__(self, input_file, output_file, report_file):
        self.input_file = input_file
        self.output_file = output_file
        self.report_file = report_file

    def extract_target_features(self, df: pd.DataFrame) -> list:
        if not os.path.exists(self.report_file):
            print(f"Error: Could not find {self.report_file}")
            sys.exit(1)
            
        report = pd.read_csv(self.report_file)
        to_fix = report[report['Final_Verdict'].str.contains("Non-Stationary|Nghi ngờ", na=False)]
        cols_to_fix = to_fix['feature'].unique().tolist()
        
        # Validation: Ensure features actually exist in DataFrame
        NON_FEATURE_COLS = {'id', 'y_target', 'weight', 'ts_index', 'code', 'sub_code', 'horizon', 't1_end', 'sub_category'}
        cols_to_fix = [col for col in cols_to_fix if col in df.columns and col not in NON_FEATURE_COLS]
        return cols_to_fix

    def execute(self):
        import gc
        print(f"====== KHỞI ĐỘNG PIPELINE STATIONARITY (Full Data / RAM Optimized) ======")
        start_all = time.time()
        
        # 1. Đọc dữ liệu và Tối ưu kiểu dữ liệu sớm
        print(f"STEP 1: Reading data from {self.input_file}...")
        df = pd.read_parquet(self.input_file)
        initial_rows = len(df)
        
        # Chuyển đổi float64 sang float32 ngay lập tức
        float64_cols = df.select_dtypes(include=['float64']).columns
        if len(float64_cols) > 0:
            print(f"   Memory Optimization: Chuyển {len(float64_cols)} cột float64 sang float32.")
            df[float64_cols] = df[float64_cols].astype('float32')
        
        gc.collect()
        
        # 2. Xác định các tham số cần xử lý từ báo cáo
        cols_to_fix = self.extract_target_features(df)
        print(f"Ghi nhận {len(cols_to_fix)} tính năng cần xử lý (Non-Stationary / Nghi ngờ).")

        # 3. Xử lý Fractional Differencing
        if cols_to_fix:
            print("\n--- BẮT ĐẦU LAYER: FRACTIONAL DIFFERENCER ---")
            fractional_differencer = Fractional_Differencer()
            df = fractional_differencer.process(df, cols_to_fix)
        else:
            print("Không có cột nào cần xử lý.")

        # 4. Kiểm tra NaNs và dọn dẹp
        print(f"\nSTEP 4: Kiểm tra NaNs và loại bỏ dòng Invalid do Windowing...")
        meta_cols = {'id', 'y_target', 'weight', 'ts_index', 'code', 'sub_code', 'horizon', 'is_test'}
        features_to_check = [c for c in df.columns if c not in meta_cols]
        
        # Drop những dòng bị NaN do Differencing (đầu chuỗi của mỗi nhóm)
        df.dropna(subset=features_to_check, inplace=True)
        
        final_rows = len(df)
        remaining_pct = (final_rows / initial_rows) * 100
        print(f"Original rows: {initial_rows} -> Remaining: {final_rows} ({remaining_pct:.2f}%)")

        # 5. Lưu kết quả
        print(f"\nSTEP 5: Sắp xếp dữ liệu và Lưu Parquet...")
        df.sort_values(['ts_index', 'code', 'sub_code'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        df.to_parquet(self.output_file, index=False)
        
        print(f"\n==========================================")
        print(f"QUÁ TRÌNH HOÀN THÀNH TOÀN BỘ")
        print(f"Tổng thời gian: {time.time() - start_all:.2f}s | Output: {self.output_file}")
        print(f"==========================================")
        
        del df
        gc.collect()


if __name__ == "__main__":
    pipeline = StationarityPipeline(INPUT_FILE, OUTPUT_FILE, REPORT_FILE)
    pipeline.execute()
