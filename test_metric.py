import numpy as np
import pandas as pd

def _clip01(x):
    return max(0.0, min(1.0, x))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    print(f"Denom: {denom}, Numerator: {np.sum(w * (y_target - y_pred) ** 2)}, Ratio: {ratio}")
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))

# Generate dummy test data replicating the validation scales
np.random.seed(42)

# w: mean ~ 1, max ~ 1e13. But let's say mostly 1e4 with a few mega weights.
w_raw = np.random.lognormal(mean=10.0, sigma=2.0, size=100000).astype(np.float64)
# Max weight is huge
w_raw[0] = 1e13

# Target targets originally mean=-0.67, std=32.5
y_orig = np.random.normal(loc=-0.67, scale=32.5, size=100000).astype(np.float64)

# Scale by 10000
y_scaled = y_orig * 10000.0

# Network initially predicts the mean perfectly (the best zero-info guess)
y_pred_mean = np.full_like(y_scaled, y_scaled.mean())

score = weighted_rmse_score(y_scaled.astype(np.float64), y_pred_mean.astype(np.float64), w_raw)
print(f"Score when predicting the global mean (64-bit): {score:.6f}")

# Now suppose it predicts just zeroes
y_pred_zero = np.zeros_like(y_scaled)
score_zero = weighted_rmse_score(y_scaled.astype(np.float64), y_pred_zero.astype(np.float64), w_raw)
print(f"Score when predicting exact zero (64-bit): {score_zero:.6f}")

# Now suppose network predicts some noisy linear combination
y_pred_model = y_scaled * 0.1 + np.random.normal(scale=1000.0, size=100000)
score_model = weighted_rmse_score(y_scaled.astype(np.float64), y_pred_model.astype(np.float64), w_raw)
print(f"Score when predicting with some signal (64-bit): {score_model:.6f}")

print("\n--- Now simulating 32-bit float calculation ---")
y_scaled_32 = y_scaled.astype(np.float32)
y_pred_model_32 = y_pred_model.astype(np.float32)
w_raw_32 = w_raw.astype(np.float64) # Weights kept float64
# In python evaluate:
all_preds = y_pred_model_32
all_y = y_scaled_32
all_w = w_raw_32

denom = np.sum(all_w * all_y ** 2)
num = np.sum(all_w * (all_y - all_preds) ** 2)
ratio = num / denom
print(f"32-bit sum - Denom: {denom}, Numerator: {num}, Ratio: {ratio}")
