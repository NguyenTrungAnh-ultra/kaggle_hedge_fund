import numpy as np

def create_pseudo_huber_objective(delta=15.0):
    """
    Tạo Hàm Mục tiêu Weighted Pseudo-Huber (Smooth L1/Charbonnier) cho LightGBM.
    Cấu trúc Closure (Hàm bao đóng) cho phép Optuna tinh chỉnh tham số delta linh hoạt.
    """
    def pseudo_huber_obj(preds, train_data):
        # 1. Trích xuất Thực tế và Trọng số
        y_true = train_data.get_label()
        w = train_data.get_weight()
        
        # Bọc lỗi phòng hờ (Mặc dù tập dữ liệu của sếp luôn có weight)
        if w is None:
            w = np.ones_like(y_true)
            
        # 2. Tính sai số tuyệt đối (d = y_hat - y)
        d = preds - y_true
        
        # 3. Ép không gian toán học (Ngăn chặn Overflow)
        # Bắt buộc dùng np.float64 vì weight có thể lên tới 10^13
        z = d / delta
        scale = np.sqrt(1.0 + z**2, dtype=np.float64)
        
        # 4. Tính Đạo hàm Bậc 1 (Gradient) và Bậc 2 (Hessian) theo C^2 Continuity
        grad_base = d / scale
        hess_base = 1.0 / (scale ** 3)
        
        # 5. Khớp trọng số (Weighting) - Phục vụ cho hàm weighted_rmse của Kaggle
        grad = grad_base * w
        hess = hess_base * w
        
        return grad, hess
        
    return pseudo_huber_obj