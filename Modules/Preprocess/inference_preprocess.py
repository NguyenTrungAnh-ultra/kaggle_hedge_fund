import pandas as pd
import numpy as np
import time
import os
import sys
import gc
import numpy as np

# Đảm bảo có thể import các module anh em khi chạy script trực tiếp
try:
    from Modules.Preprocess.handle_na import NaN_Handler
    from Modules.Preprocess.make_stationary import Fractional_Differencer
    from Modules.Preprocess.feature_builder import AlphaGenerator
except ImportError:
    from handle_na import NaN_Handler
    from make_stationary import Fractional_Differencer
    from feature_builder import AlphaGenerator

class InferencePreprocess:
    """
    Module hỗ trợ nạp dữ liệu hợp nhất (Unified Loading) và Tiền xử lý phục vụ Inference.
    Đảm bảo tính liên tục của dữ liệu (Memory) từ Train sang Test.
    """
    @staticmethod
    def full_pipeline(train_path, test_path, report_path):
        print("\n" + "="*60)
        print("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ HỢP NHẤT (MEMORY PRESERVATION)")
        print("="*60)
        start_time = time.time()
        
        # 1. Loading
        print("1. Loading raw TRAIN and TEST data directly into contiguous timeline...")
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        print(f"   Nạp xong. Train: {len(train)} dòng, Test: {len(test)} dòng.")
        
        # 2. Labeling
        train['is_test'] = 0
        test['is_test'] = 1
        
        # 3. Concatenation (Timeline Continuity)
        df = pd.concat([train, test], ignore_index=True)
        del train, test
        gc.collect()
        
        # 4. Memory Optimization (Crucial for 6M+ rows)
        # Chuyển đổi float64 sang float32 để tiết kiệm 1/2 bộ nhớ
        float64_cols = df.select_dtypes(include=['float64']).columns
        df[float64_cols] = df[float64_cols].astype('float32')
        print(f"   Memory Optimization: Chuyển {len(float64_cols)} cột sang float32.")
        gc.collect()
        
        # 5. Identify Features
        meta_cols = ['id', 'y_target', 'weight', 'ts_index', 'is_test']
        features = [c for c in df.columns if c not in meta_cols]
        
        # 5. NaN Imputation (Restoring Memory)
        print("2. Khởi tạo NaN Handler để phục hồi Memory (Layer 1-4)...")
        nan_handler = NaN_Handler()
        df = nan_handler.process(df, features)
        
        # 6. Stationarity (Fractional Differencing)
        if not os.path.exists(report_path):
            print(f"CẢNH BÁO: Không thấy file báo cáo dừng {report_path}. Bỏ qua FracDiff.")
        else:
            print(f"3. Bắt đầu xử lý Stationarity dựa trên báo cáo: {report_path}")
            report = pd.read_csv(report_path)
            to_fix = report[report['Final_Verdict'].str.contains("Non-Stationary|Nghi ngờ", na=False)]
            cols_to_fix = to_fix['feature'].unique().tolist()
            # Chỉ xử lý các cột thực sự có trong dữ liệu
            cols_to_fix = [c for c in cols_to_fix if c in df.columns]
            
            print(f"   Phát hiện {len(cols_to_fix)} tính năng cần xử lý FracDiff.")
            
            # Xử lý Stationarity sử dụng class Fractional_Differencer đã được tối ưu
            diff_handler = Fractional_Differencer()
            df = diff_handler.process(df, cols_to_fix)
        # 7. AlphaGenerator
        print("4. Bắt đầu tạo Alpha Features (Lags & Z-Score)...")
        alpha_gen = AlphaGenerator()
        df = alpha_gen.inject_lagged_momentum(df)
        df = alpha_gen.inject_cross_sectional_zscore(df)
        gc.collect()
        
        # 8. Final Sorting for Time-Series logic
        print("4. Sắp xếp dữ liệu và xuất file...")
        df.sort_values(['code', 'sub_code', 'horizon', 'ts_index'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        gc.collect()
        
        # 8. Export to ts-forecasting directory
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ts-forecasting')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_file = os.path.join(output_dir, 'inference_processed_full.parquet')
        print(f"Datashape: {df.shape}")
        print(f"4. Exporting fully processed DF to: {output_file}")
        df.to_parquet(output_file, index=False)
        print(f"Đã lưu tại {output_file}")
        
        print(f"--- HOÀN THÀNH PIPELINE TIỀN XỬ LÝ: {time.time() - start_time:.2f}s ---\n")
        print("="*60 + "\n")
        return df, features

if __name__ == "__main__":
    # Test run logic
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
    REPORT_PATH = os.path.join(BASE_DIR, 'reports', 'report_COMPARISON.csv')
    TRAIN_RAW = os.path.join(DATA_DIR, 'train.parquet')
    TEST_RAW = os.path.join(DATA_DIR, 'test.parquet')
    
    if os.path.exists(TRAIN_RAW) and os.path.exists(TEST_RAW):
        df, features = InferencePreprocess.full_pipeline(TRAIN_RAW, TEST_RAW, REPORT_PATH)
        print(f"Test run success: {df.shape}")
    else:
        print(f"Skipping test run (files missing: {TRAIN_RAW})")