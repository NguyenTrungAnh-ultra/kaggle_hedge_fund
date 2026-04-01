"""
sai bài toán
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os
import sys
import time
import warnings
import gc
from joblib import Parallel, delayed

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
    Module Xử lý Dừng phân số (Fractional Differencing) chuẩn mực AFML.
    Sử dụng Fixed-Width Window theo ngưỡng cắt (tau).
    """
    @staticmethod
    def get_weights_ffw(d, tau=1e-4, max_size=2000):
        """Tạo chuỗi trọng số w_0, w_1, ..., w_k"""
        w = [1.0]
        for k in range(1, max_size):
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < tau:
                break
            w.append(w_k)
        # QUAN TRỌNG: Không đảo ngược weights khi dùng np.convolve
        return np.array(w, dtype=np.float32)

    @staticmethod
    def find_optimal_d(series):
        series = series.dropna().values
        if len(series) < 100: 
            return 1.0 
        
        # Thử d từ 0.1 đến 0.9
        for d in np.arange(0.1, 1.0, 0.1):
            w_for_conv = Fractional_Differencer.get_weights_ffw(d)
            if len(w_for_conv) >= len(series):
                continue
                
            # Mode 'valid' bỏ qua các phần tử không đủ cửa sổ
            diff_series = np.convolve(series, w_for_conv, mode='valid')
            
            if len(diff_series) > 50:
                try:
                    p_val = adfuller(diff_series)[1]
                    if p_val <= 0.05:
                        return float(d)
                except:
                    continue
        return 1.0

    def _process_single_col(self, col, series_values, group_offsets, is_test_col=None, test_idx=None):
        """Hàm xử lý cho một cột đơn lẻ, được thiết kế để chạy song song"""
        start_col = time.time()
        
        # 1. Tìm d_opt trên tập train của group đại diện
        if is_test_col is not None and test_idx is not None:
            # Lọc group data (test_idx) và chỉ lấy tập train (is_test == 0)
            mask = (is_test_col == 0)
            # test_idx là array các index của group đại diện
            test_series = series_values[test_idx[mask[test_idx]]]
        else:
            test_series = series_values[test_idx] if test_idx is not None else series_values
            
        opt_d = self.find_optimal_d(pd.Series(test_series))
        
        # 2. Thực hiện Differencing
        res_col = np.full_like(series_values, np.nan, dtype=np.float32)
        
        if opt_d < 1.0:
            w = self.get_weights_ffw(opt_d)
            w_len = len(w)
            
            # Xử lý từng group theo offset đã tính trước (Nhanh hơn groupby.transform)
            for i in range(len(group_offsets)-1):
                start, end = group_offsets[i], group_offsets[i+1]
                g_vals = series_values[start:end]
                
                if len(g_vals) >= w_len:
                    # Fill NaNs đơn giản bằng 0 cho convolve (Dữ liệu đã qua handled_na)
                    # Sau đó sẽ trả lại NaNs thực tế sau
                    g_filled = np.copy(g_vals)
                    nan_mask = np.isnan(g_filled)
                    if np.any(nan_mask):
                        g_filled[nan_mask] = 0.0
                    
                    # Convolve
                    diff_vals = np.convolve(g_filled, w, mode='full')[:len(g_vals)]
                    diff_vals[:w_len-1] = np.nan
                    diff_vals[nan_mask] = np.nan
                    res_col[start:end] = diff_vals.astype(np.float32)
                    
        elif opt_d == 1.0:
            for i in range(len(group_offsets)-1):
                start, end = group_offsets[i], group_offsets[i+1]
                g_vals = series_values[start:end]
                if len(g_vals) > 1:
                    res_col[start+1:end] = np.diff(g_vals).astype(np.float32)
        
        return col, res_col, opt_d, (time.time() - start_col)

    def process(self, df: pd.DataFrame, features_list: list) -> pd.DataFrame:
        print(f"\n--- BẮT ĐẦU XỬ LÝ DỪNG PHÂN SỐ: AFML FRACTIONAL DIFFERENCER (Parallel Optimized) ---")
        start_time = time.time()

        # 1. Dự toán Group Boundaries (offsets) một lần duy nhất
        df = df.sort_values(['code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
        group_sizes = df.groupby(['code', 'sub_code', 'horizon'], sort=False).size().values
        group_offsets = np.insert(np.cumsum(group_sizes), 0, 0)
        
        # 2. Tìm Group đại diện lớn nhất
        top_group_idx_in_sizes = np.argmax(group_sizes)
        start_top, end_top = group_offsets[top_group_idx_in_sizes], group_offsets[top_group_idx_in_sizes+1]
        test_idx = np.arange(start_top, end_top)
        
        is_test_col = df['is_test'].values if 'is_test' in df.columns else None
        
        # 3. Chạy Song song cho các Features
        print(f"Xử lý song song {len(features_list)} features sử dụng all CPU cores...")
        
        results = Parallel(n_jobs=-1)(
            delayed(self._process_single_col)(
                col, 
                df[col].values, 
                group_offsets, 
                is_test_col, 
                test_idx
            ) for col in features_list
        )
        
        # 4. Gộp kết quả
        for col, res_col, opt_d, duration in results:
            df[f"{col}_sta"] = res_col
            print(f"  - {col:15} | d_opt: {opt_d:.1f} | Done ({duration:.2f}s)")
            
        print(f"Datashape: {df.shape}")
        print(f"Tổng NaN sau xử lý Dừng Phân Số (Fractional Differencing): {df.isna().sum().sum()}")
        print(f"HOÀN THÀNH XỬ LÝ: {time.time() - start_time:.2f}s")
        gc.collect()
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
        
        # # Drop những dòng bị NaN do Differencing (đầu chuỗi của mỗi nhóm)
        # df.dropna(subset=features_to_check, inplace=True)
        
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
        print(f"Datashape: {df.shape}")
        print(f"==========================================")
        
        del df
        gc.collect()


if __name__ == "__main__":
    pipeline = StationarityPipeline(INPUT_FILE, OUTPUT_FILE, REPORT_FILE)
    pipeline.execute()
