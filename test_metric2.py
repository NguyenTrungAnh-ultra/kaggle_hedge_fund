import numpy as np

def _clip01(x):
    return max(0.0, min(1.0, x))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))

np.random.seed(42)
w_raw = np.random.lognormal(mean=10.0, sigma=2.0, size=100000).astype(np.float64)
# Max weight is huge
w_raw[0] = 1e13

y_orig = np.random.normal(loc=-0.67, scale=32.5, size=100000).astype(np.float64)

# Try unscaled target
y_pred_mean = np.full_like(y_orig, y_orig.mean())
score = weighted_rmse_score(y_orig, y_pred_mean, w_raw)
print(f"UNSCALED: Score when predicting the unweighted mean: {score:.6f}")

w_mean = np.sum(w_raw * y_orig) / np.sum(w_raw)
y_pred_wmean = np.full_like(y_orig, w_mean)
score2 = weighted_rmse_score(y_orig, y_pred_wmean, w_raw)
print(f"UNSCALED: Score when predicting the weighted mean: {score2:.6f}")

score3 = weighted_rmse_score(y_orig, np.zeros_like(y_orig), w_raw)
print(f"UNSCALED: Score when predicting exactly zero: {score3:.6f}")
