import polars as pl
import os
import argparse

def detect_static_dynamic(df_path: str, group_col: str = 'code'):
    """
    Máy dò Biến Tĩnh vs. Biến Động (Static vs. Dynamic Radar)
    
    Cơ chế: 
    Nhóm (Groupby) dữ liệu theo code và tính Độ lệch chuẩn (Standard Deviation - std) của từng feature.
    
    Luật suy luận: 
    - Nếu std == 0 cho mọi code: Đó là Biến Tĩnh (Static). Thường là thuộc tính phân loại ẩn.
    - Nếu std > 0: Đó là Biến Động (Dynamic).
    """
    print(f"Reading data from {df_path}...")
    
    # Sử dụng lazy scan để tối ưu RAM
    lf = pl.scan_parquet(df_path)
    
    # Lấy toàn bộ tên cột
    all_cols = lf.columns
    
    # Loại bỏ các cột định danh và target
    exclude_cols = {'id', 'ts_index', 'y_target', 'weight', 'sub_code', 'sub_category', 'horizon', group_col}
    # Loại bỏ các cột is_nan_ (nếu có)
    features = [c for c in all_cols if c not in exclude_cols and not c.startswith('is_nan_')]
    
    print(f"Analyzing {len(features)} features grouped by '{group_col}'...")
    
    # Tính std per group cho từng feature
    # Lưu ý: pl.col(f).std() sẽ trả về null nếu group chỉ có 1 dòng hoặc toàn null
    agg_exprs = [pl.col(f).std().alias(f) for f in features]
    
    # Thực thi aggregation
    std_df = lf.group_by(group_col).agg(agg_exprs).collect()
    
    static_features = []
    dynamic_features = []
    
    for f in features:
        # Lấy giá trị std lớn nhất tìm thấy trong tất cả các nhóm
        # Nếu bất kỳ nhóm nào có std > 0 -> Biến Động
        # Nếu mọi nhóm có std == 0 (hoặc null) -> Biến Tĩnh
        max_std = std_df[f].max()
        
        if max_std is None or max_std == 0:
            static_features.append(f)
        else:
            dynamic_features.append(f)
            
    return static_features, dynamic_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Static vs. Dynamic Feature Radar')
    parser.add_argument('--input', type=str, default=r'f:\Projects\hege_fund_kaggle\ts-forecasting\train.parquet', help='Path to parquet file')
    parser.add_argument('--group', type=str, default='code', help='Column to group by')
    args = parser.parse_args()
    
    # Kiểm tra file tồn tại, nếu không thử bản backup/processed
    input_path = args.input
    if not os.path.exists(input_path):
        alternatives = [
            r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_handled_na.parquet',
            r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                input_path = alt
                break
        else:
            print(f"Error: File not found {args.input}")
            exit(1)
            
    static, dynamic = detect_static_dynamic(input_path, args.group)
    
    print("\n" + "="*60)
    print(f"📊 KẾT QUẢ PHÂN TÍCH (Nhóm theo {args.group})")
    print("="*60)
    print(f"✅ Biến Tĩnh - Static Features ({len(static)}):")
    if static:
        # In theo dạng cột cho đẹp
        for i in range(0, len(static), 4):
            print("  " + " | ".join(f"{s:15}" for s in static[i:i+4]))
    else:
        print("  (Không tìm thấy)")
        
    print(f"\n\n🔥 Biến Động - Dynamic Features ({len(dynamic)}):")
    print(f"  Tổng số: {len(dynamic)}")
    if len(dynamic) > 0:
        sample_dynamic = dynamic[:20]
        print(f"  Ví dụ 20 biến đầu: {', '.join(sample_dynamic)}...")
    print("="*60)
    
    # Lưu kết quả ra file trong thư mục reports (nếu có) hoặc cùng thư mục data
    output_dir = r'f:\Projects\hege_fund_kaggle\reports'
    if not os.path.exists(output_dir):
        output_dir = os.path.dirname(input_path)
        
    report_file = os.path.join(output_dir, 'feature_dynamics_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"STATIC VS DYNAMIC RADAR REPORT\n")
        f.write(f"Source: {input_path}\n")
        f.write(f"Grouped by: {args.group}\n")
        f.write(f"="*30 + "\n\n")
        f.write(f"STATIC FEATURES ({len(static)}):\n")
        f.write(",".join(static) + "\n\n")
        f.write(f"DYNAMIC FEATURES ({len(dynamic)}):\n")
        f.write(",".join(dynamic) + "\n")
        
    print(f"Báo cáo chi tiết đã được lưu tại: {report_file}")
