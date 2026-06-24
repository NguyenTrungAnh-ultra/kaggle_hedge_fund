import pandas as pd
import numpy as np
import time
import os
import sys
import gc

try:
    from Modules.Preprocess.handle_na import NaN_Handler
except ImportError:
    from handle_na import NaN_Handler

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
        
        # Sắp xếp sơ bộ theo phân cấp mới (Sector -> Ticker -> SubTicker)
        df.sort_values(['sub_category', 'code', 'sub_code', 'horizon', 'ts_index'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # 4. Memory Optimization (Crucial for 6M+ rows)
        float64_cols = df.select_dtypes(include=['float64']).columns
        df[float64_cols] = df[float64_cols].astype('float32')
        print(f"   Memory Optimization: Chuyển {len(float64_cols)} cột sang float32.")
        gc.collect()
        
        # 5. Identify Features
        meta_cols = ['id', 'y_target', 'weight', 'ts_index', 'is_test', 'code', 'sub_code', 'sub_category', 'horizon']
        features = [c for c in df.columns if c not in meta_cols]
        
        # 6. NaN Imputation (Restoring Memory)
        print("2. Khởi tạo NaN Handler để phục hồi Memory (Layer 1-4)...")
        nan_handler = NaN_Handler()
        df = nan_handler.process(df, features)
        gc.collect()

        # Removed LogReturn and AlphaGenerator (Architectural Debt - We use RAW ONLY features now)
        
        # 7. Final Sorting for Time-Series logic (Synced with new Hierarchy)
        print("4. Sắp xếp dữ liệu và xuất file...")
        df.sort_values(['sub_category', 'code', 'sub_code', 'horizon', 'ts_index'], inplace=True)
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