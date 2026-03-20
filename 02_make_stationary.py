import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
        series = series.dropna().values
        if len(series) < 20: 
            return 1.0 
        for d in np.linspace(0, 1, 11):
            if d == 0.0:
                diff_series = series
            else:
                w_full = Fractional_Differencer.get_weights(d, len(series))
                w_cumsum = np.cumsum(np.abs(w_full))
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
                    p_val = adfuller(diff_series)[1]
                    if p_val <= 0.05:
                        return float(d)
                except:
                    continue
        return 1.0

    def process(self, df: pd.DataFrame, features_list: list) -> pd.DataFrame:
        print(f"\n--- BẮT ĐẦU XỬ LÝ DỪNG PHÂN SỐ: FRACTIONAL DIFFERENCER ---")
        start_time = time.time()
        
        # Đảm bảo sắp xếp chuẩn thời gian
        df = df.sort_values(['code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
        
        print("Ensuring AFML compliance: Estimating d ONLY on the 80% chronological train set to prevent Data Leak...")
        unique_ts = np.sort(df['ts_index'].unique())
        split_ts_idx = int(len(unique_ts) * 0.80)
        split_time = unique_ts[split_ts_idx]
        train_df_for_est = df[df['ts_index'] < split_time]
        
        print("Finding the largest group [code, sub_code, horizon] in Train set for d estimation...")
        top_group_keys = train_df_for_est.groupby(['code', 'sub_code', 'horizon']).size().idxmax()
        mask_test = (train_df_for_est['code'] == top_group_keys[0]) & (train_df_for_est['sub_code'] == top_group_keys[1]) & (train_df_for_est['horizon'] == top_group_keys[2])
        test_group = train_df_for_est[mask_test]

        print(f"Bắt đầu FFW cho {len(features_list)} cột...")
        max_size = df.groupby(['code', 'sub_code', 'horizon'], sort=False).size().max()
        grouped = df.groupby(['code', 'sub_code', 'horizon'], sort=False)

        for idx, col in enumerate(features_list, 1):
            start_col = time.time()
            
            test_series = test_group.set_index('ts_index')[col]
            opt_d = self.find_optimal_d(test_series)
            
            print(f"[{idx}/{len(features_list)}] {col:15} | d_opt: {opt_d:.1f} | ", end='', flush=True)
            
            if opt_d > 0.0:
                w_full = self.get_weights(opt_d, max_size)
                w_cumsum = np.cumsum(np.abs(w_full))
                skip = np.searchsorted(w_cumsum / w_cumsum[-1], 1e-2)
                
                if (max_size - skip) > 50:
                    skip = max_size - 50
                    
                w = w_full[skip:]
                if len(w) == 0:
                    print(f"Warning: weights empty, keeping original. Done ({time.time() - start_col:.2f}s)")
                    continue

                w_for_conv = w[::-1].flatten()
                window_len = len(w)

                def apply_ffw(g_vals):
                    if len(g_vals) == 0:
                        return g_vals
                    conv = np.convolve(g_vals, w_for_conv, mode='full')[:len(g_vals)]
                    if len(conv) >= window_len:
                        conv[:window_len-1] = np.nan
                    else:
                        conv[:] = np.nan
                    return conv
                    
                df[col] = grouped[col].transform(lambda g: apply_ffw(g.values))
            
            print(f"Done ({time.time() - start_col:.2f}s)")
            
        print(f"HOÀN THÀNH XỬ LÝ DỪNG PHÂN SỐ: {time.time() - start_time:.2f}s")
        return df


class StationarityPipeline:
    """
    Pipeline chính để điều phối quá trình xử lý Stationarity qua nhiều lớp bảo vệ (AFML). 
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
        print(f"====== KHỞI ĐỘNG PIPELINE STATIONARITY ======")
        start_all = time.time()
        
        print(f"STEP 1: Reading raw data from {self.input_file}...")
        df = pd.read_parquet(self.input_file)
        initial_rows = len(df)
        
        cols_to_fix = self.extract_target_features(df)
        print(f"Ghi nhận {len(cols_to_fix)} tính năng cần xử lý (Non-Stationary / Nghi ngờ).")

        # Đóng gói và Xử lý qua 2 Layer Dừng Hóa (Stationarity) độc lập
        # Lớp 1: Khử tính mùa vụ bằng Backshift Operator Vectorized
        # print("\n--- LAYER 1: SEASONAL HANDLER (Memory Preservation Priority) ---")
        # seasonal_handler = Seasonal_Handler(m_candidates=[5, 22], acf_threshold=0.85)
        
        # Lưu vết các cột đã được xử lý mùa vụ để tránh Over-differencing ở lớp sau
        # Chúng ta sẽ kiểm tra xem cột đó có được transform nhiều không
        # df_old = df[cols_to_fix].copy()
        # df = seasonal_handler.process(df, cols_to_fix)
        
        # Chỉ những cột KHÔNG bị biến đổi đáng kể bởi Seasonal Handler mới được đi tiếp vào Fractional Diff
        # Nếu cột đã bị trừ lùi (Seasonal), coi như nó đã Stationarity hoặc đã mất memory bậc 1, không nên FracDiff thêm.
        # diff_mask = (df[cols_to_fix] != df_old).any()
        # seasonally_fixed_cols = diff_mask[diff_mask].index.tolist()
        # remaining_cols_for_frac = [c for c in cols_to_fix if c not in seasonally_fixed_cols]
        
        # print(f"Ghi nhận {len(seasonally_fixed_cols)} cột đã khử mùa vụ. Bỏ qua FracDiff để tránh Over-differencing.")
        # print(f"Còn lại {len(remaining_cols_for_frac)} cột tiếp tục vào Layer 2.")

        # Lớp 2: Khử tính không dừng phân số theo FFW Windowing
        print("\n--- LAYER 2: FRACTIONAL DIFFERENCER (Memory Preservation) ---")
        
        # Nếu không dùng Seasonal_Handler, toàn bộ cols_to_fix sẽ được FracDiff
        remaining_cols_for_frac = cols_to_fix
        
        if remaining_cols_for_frac:
            fractional_differencer = Fractional_Differencer()
            df = fractional_differencer.process(df, remaining_cols_for_frac)
        else:
            print("Không có cột nào cần xử lý FracDiff.")

        print(f"\nSTEP 4: Kiểm tra NaNs và loại bỏ dòng Invalid do Windowing/Shift...")
        features_to_check = [c for c in df.columns if c not in ['id', 'y_target']]
        df = df.dropna(subset=features_to_check)
        
        final_rows = len(df)
        remaining_pct = (final_rows / initial_rows) * 100
        print(f"Original rows: {initial_rows}")
        print(f"Remaining rows sau khi drop NaNs (Clean Slice): {final_rows}")
        print(f"Tỷ lệ giữ lại dữ liệu: {remaining_pct:.2f}%")

        if df[features_to_check].isna().sum().sum() > 0:
            print("CẢNH BÁO: Vẫn còn NaNs trong feature set!")
        else:
            print("THÀNH CÔNG: Dữ liệu sạch, không còn NaNs!")

        print(f"\nSTEP 5: Sắp xếp dữ liệu và Lưu Output...")
        df = df.sort_values(['ts_index', 'code', 'sub_code']).reset_index(drop=True)
        df.to_parquet(self.output_file, index=False)
        
        print(f"\n==========================================")
        print(f"QUÁ TRÌNH HOÀN THÀNH TOÀN BỘ")
        print(f"Tổng thực thi Dataset 5.3M Dòng: {time.time() - start_all:.2f}s")
        print(f"Output lưu tại: {self.output_file}")
        print(f"==========================================")


if __name__ == "__main__":
    pipeline = StationarityPipeline(INPUT_FILE, OUTPUT_FILE, REPORT_FILE)
    pipeline.execute()
