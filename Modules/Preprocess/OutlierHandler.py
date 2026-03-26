import pandas as pd

class OutlierHandler:
    """
    Module xử lý các giá trị ngoại lai (Outliers) để bảo vệ tính ổn định của mô hình.
    """
    @staticmethod
    def clip_target(df: pd.DataFrame, target_col='y_target', lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
        """
        BẢO VỆ HÀM MẤT MÁT: TARGET CLIPPING (CẮT TỈA NGOẠI LAI)
        
        Nhiệm vụ: Cắt bỏ 1% (hoặc tùy chọn) điểm bất thường ở hai đầu phân phối của Target.
        Giúp thuật toán Gradient Boosting (LightGBM/XGBoost) không bị 'kéo' quá mạnh theo 
        các điểm nhiễu cực đoan.
        """
        print("\n--- BẮT ĐẦU: CẮT TỈA NGOẠI LAI TARGET (CLIPPING) ---")
        
        if target_col not in df.columns:
            print(f"   Cảnh báo: Không tìm thấy cột {target_col} trong DataFrame.")
            return df

        # Tính toán ranh giới dựa trên phân vị (Quantiles)
        lower_bound = df[target_col].quantile(lower_q)
        upper_bound = df[target_col].quantile(upper_q)
        
        # Thực hiện ép giá trị (Clamping)
        df[target_col] = df[target_col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"   [Quant Security] Target Clipping activated.")
        print(f"   > {target_col} được giới hạn trong khoảng [{lower_bound:.6f}, {upper_bound:.6f}]")
        
        return df